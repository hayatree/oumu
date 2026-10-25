//! VoxCPM2 on the GPU: weights, prefill and one decoding step (capturable as a CUDA graph).
//!
//! One step, for B slots at once: the DiT once per diffusion step (Euler + CFG-Zero*; 9 with the default 10
//! timesteps) -> local encoder on the new patch -> stop head -> base LM step -> FSQ -> residual LM step.
use crate::gpu::{A::*, Arena, Gpu, Ptr, R, b2f, f2b, rb};
use crate::{json, st};
use cudarc::driver::sys;

// VoxCPM2 dimensions (checked against config.json by check_config)
const LM: usize = 2048; // language model width
const LM_FFN: usize = 6144;
const LOC: usize = 1024; // local encoder and DiT width
const LOC_FFN: usize = 4096;
const FSQ: usize = 512;
const NQ: usize = 16;
const NK: usize = 2;
const HD: usize = 128;
const QKV: usize = (NQ + 2 * NK) * HD;
const QW: usize = NQ * HD;
const PATCH: usize = 4 * 64; // one patch: 4 latent frames x 64 channels
const ENC_SEQ: usize = 5; // local encoder input: special token + 4 frames
const DIT_SEQ: usize = 11; // DiT input: mu (2), time, 4 condition frames, 4 noisy frames
const DIT_X0: usize = 7; // first noisy frame in the DiT sequence
const ATT_CH: usize = 64; // KV positions per attn_decode block (ATT_CH in the kernels)
const MAX_SLOTS: usize = 256; // step_end runs one thread per slot in a 256-thread block

pub struct Lin {
    w: Ptr,
    b: Option<Ptr>,
    n: usize,
    k: usize,
}

impl Lin {
    fn fwd(&self, g: &Gpu, x: Ptr, ldx: usize, y: Ptr, ldy: usize, m: usize) -> R<()> {
        let beta = match self.b {
            Some(b) => {
                g.k1("fill_bias", m * self.n, &[P(y), P(b), I(m as i32), I(self.n as i32), I(ldy as i32)])?;
                1.0
            }
            None => 0.0,
        };
        g.gemm(x, ldx, self.w, y, ldy, m, self.n, self.k, beta)
    }
}

struct Layer {
    ln1: Ptr,
    qkv: Lin,
    o: Lin,
    ln2: Ptr,
    gu: Lin,
    down: Lin,
}

pub struct Tf {
    layers: Vec<Layer>,
    norm: Ptr,
    h: usize,
    ffn: usize,
    rope: bool,
    kc: Vec<Ptr>,
    vc: Vec<Ptr>,
}

/// Shared activations for every transformer pass.
struct Scratch {
    xn: Ptr,
    qkv: Ptr,
    att: Ptr,
    tmp: Ptr,
    gu: Ptr,
    act: Ptr,
}

/// Places the weights in one device arena, streamed from the memory-mapped checkpoint.
struct Loader<'a> {
    f: &'a st::File,
    a: Arena,
}

impl Loader<'_> {
    /// Uploads tensors concatenated along dim 0 (fused qkv / gate-up weights); `want` is the shape of the result.
    fn cat(&mut self, g: &mut Gpu, names: &[String], want: &[usize]) -> R<Ptr> {
        let mut parts = vec![];
        let mut rows = 0;
        for n in names {
            let t = self.f.get(n)?;
            if t.dtype != "BF16" {
                return Err(format!("{n}: expected BF16, got {}", t.dtype).into());
            }
            if t.shape.len() != want.len() || t.shape[1..] != want[1..] {
                return Err(format!("{n}: shape {:?} does not fit {want:?}", t.shape).into());
            }
            rows += t.shape[0];
            parts.push(t.data);
        }
        if rows != want[0] {
            return Err(format!("{}: {rows} rows, expected {}", names.join(" + "), want[0]).into());
        }
        self.a.put(g, &parts)
    }
    fn one(&mut self, g: &mut Gpu, n: &str, want: &[usize]) -> R<Ptr> {
        self.cat(g, &[n.to_string()], want)
    }
    /// A linear layer with an [n, k] weight (several concatenated along n) and, for one weight, an optional bias.
    fn lin(&mut self, g: &mut Gpu, names: &[String], n: usize, k: usize, bias: bool) -> R<Lin> {
        let w = self.cat(g, names, &[n, k])?;
        let b = if bias { Some(self.one(g, &names[0].replace(".weight", ".bias"), &[n])?) } else { None };
        Ok(Lin { w, b, n, k })
    }
    fn lin1(&mut self, g: &mut Gpu, name: &str, n: usize, k: usize, bias: bool) -> R<Lin> {
        self.lin(g, &[format!("{name}.weight")], n, k, bias)
    }
    #[allow(clippy::too_many_arguments)]
    fn tf(&mut self, g: &mut Gpu, p: &str, n: usize, h: usize, ffn: usize, rope: bool, cache: Option<(usize, usize)>) -> R<Tf> {
        let mut layers = vec![];
        let (mut kc, mut vc) = (vec![], vec![]);
        for i in 0..n {
            let l = format!("{p}.layers.{i}");
            let a = format!("{l}.self_attn");
            layers.push(Layer {
                ln1: self.one(g, &format!("{l}.input_layernorm.weight"), &[h])?,
                qkv: self.lin(g, &[format!("{a}.q_proj.weight"), format!("{a}.k_proj.weight"), format!("{a}.v_proj.weight")], QKV, h, false)?,
                o: self.lin1(g, &format!("{a}.o_proj"), h, QW, false)?,
                ln2: self.one(g, &format!("{l}.post_attention_layernorm.weight"), &[h])?,
                gu: self.lin(g, &[format!("{l}.mlp.gate_proj.weight"), format!("{l}.mlp.up_proj.weight")], 2 * ffn, h, false)?,
                down: self.lin1(g, &format!("{l}.mlp.down_proj"), h, ffn, false)?,
            });
            if let Some((slots, maxlen)) = cache {
                kc.push(g.alloc_bf16(slots * NK * maxlen * HD)?);
                vc.push(g.alloc_bf16(slots * NK * maxlen * HD)?);
            }
        }
        let norm = self.one(g, &format!("{p}.norm.weight"), &[h])?;
        Ok(Tf { layers, norm, h, ffn, rope, kc, vc })
    }
}

pub struct Engine {
    pub g: Gpu,
    base: Tf,
    res: Tf,
    enc: Tf,
    dit: Tf,
    embed: Ptr,
    enc_in: Lin,
    enc_special: Ptr,
    enc_to_lm: Lin,
    dit_in: Lin,
    dit_cond: Lin,
    dit_out: Lin,
    fsq_in: Lin,
    fsq_out: Lin,
    fusion: Lin,
    lm_to_dit: Lin,
    res_to_dit: Lin,
    stop_proj: Lin,
    stop_head: Lin,
    cos: Ptr,
    sin: Ptr,
    eps: f32,
    sc: Scratch,
    pub b: usize,
    pub maxlen: usize,
    tmax: usize,
    vocab: usize,
    sched: Vec<(f32, f32)>,
    temb: Ptr,
    pub cfg: f32,
    s: StepBufs,
    pf: PrefillBufs,
    /// generated patches [b][max_steps][PATCH]
    patches: Ptr,
    patches_cap: usize,
    max_steps: usize,
    /// replayed reference noise (verification), or 0 for the seeded generator
    noise_tab: Ptr,
    noise_bytes: usize,
    pub min_len: i32,
}

/// Per-slot state and activations of one decoding step.
struct StepBufs {
    x: Ptr,
    cond: Ptr,
    mu: Ptr,
    lm_h: Ptr,
    lm_raw: Ptr,
    res_h: Ptr,
    base_x: Ptr,
    res_x: Ptr,
    fus: Ptr,
    q: Ptr,
    condp: Ptr,
    xp: Ptr,
    dit_seq: Ptr,
    dit_hout: Ptr,
    dit_tok: Ptr,
    dit_o: Ptr,
    enc_seq: Ptr,
    enc_hout: Ptr,
    enc_cls: Ptr,
    stop_h: Ptr,
    stop_l: Ptr,
    att_part: Ptr,
    /// [position, step, 0, 0], read by the kernels so one graph serves every step
    ctr: Ptr,
    stop: Ptr,
    seeds: Ptr,
}

impl StepBufs {
    fn new(g: &mut Gpu, b: usize, maxlen: usize) -> R<Self> {
        Ok(StepBufs {
            x: g.alloc_bf16(b * PATCH)?,
            cond: g.alloc_bf16(b * PATCH)?,
            mu: g.alloc_bf16(b * LM)?,
            lm_h: g.alloc_bf16(b * LM)?,
            lm_raw: g.alloc_bf16(b * LM)?,
            res_h: g.alloc_bf16(b * LM)?,
            base_x: g.alloc_bf16(b * LM)?,
            res_x: g.alloc_bf16(b * LM)?,
            fus: g.alloc_bf16(b * 2 * LM)?,
            q: g.alloc_bf16(b * FSQ)?,
            condp: g.alloc_bf16(b * 4 * LOC)?,
            xp: g.alloc_bf16(b * 4 * LOC)?,
            dit_seq: g.alloc_bf16(2 * b * DIT_SEQ * LOC)?,
            dit_hout: g.alloc_bf16(2 * b * DIT_SEQ * LOC)?,
            dit_tok: g.alloc_bf16(2 * b * 4 * LOC)?,
            dit_o: g.alloc_bf16(2 * b * PATCH)?,
            enc_seq: g.alloc_bf16(b * ENC_SEQ * LOC)?,
            enc_hout: g.alloc_bf16(b * ENC_SEQ * LOC)?,
            enc_cls: g.alloc_bf16(b * LOC)?,
            stop_h: g.alloc_bf16(b * LM)?,
            stop_l: g.alloc_bf16(b * 2)?,
            att_part: g.alloc(4 * b * NQ * (maxlen / ATT_CH) * (HD + 2))?,
            ctr: g.alloc(16)?,
            stop: g.alloc(4 * b)?,
            seeds: g.alloc(4 * b)?,
        })
    }
}

/// Prompt inputs and activations of the prefill, for prompts of up to `tmax` positions.
struct PrefillBufs {
    tok: Ptr,
    tm: Ptr,
    am: Ptr,
    feat: Ptr,
    xp: Ptr,
    eseq: Ptr,
    eout: Ptr,
    cls: Ptr,
    fe: Ptr,
    x: Ptr,
    out: Ptr,
    q: Ptr,
    fsq: Ptr,
    mix: Ptr,
    resin: Ptr,
    rout: Ptr,
}

impl PrefillBufs {
    fn new(g: &mut Gpu, tmax: usize) -> R<Self> {
        Ok(PrefillBufs {
            tok: g.alloc(4 * tmax)?,
            tm: g.alloc(4 * tmax)?,
            am: g.alloc(4 * tmax)?,
            feat: g.alloc_bf16(tmax * PATCH)?,
            xp: g.alloc_bf16(tmax * 4 * LOC)?,
            eseq: g.alloc_bf16(tmax * ENC_SEQ * LOC)?,
            eout: g.alloc_bf16(tmax * ENC_SEQ * LOC)?,
            cls: g.alloc_bf16(tmax * LOC)?,
            fe: g.alloc_bf16(tmax * LM)?,
            x: g.alloc_bf16(tmax * LM)?,
            out: g.alloc_bf16(tmax * LM)?,
            q: g.alloc_bf16(tmax * FSQ)?,
            fsq: g.alloc_bf16(tmax * LM)?,
            mix: g.alloc_bf16(tmax * LM)?,
            resin: g.alloc_bf16(tmax * 2 * LM)?,
            rout: g.alloc_bf16(tmax * LM)?,
        })
    }
}

fn off(p: Ptr, elems: usize) -> Ptr {
    p + 2 * elems as u64
}

/// Diffusion schedule of UnifiedCFM (sway sampling, bf16 arithmetic): (t, dt) of every step that runs the DiT.
fn schedule(n: usize) -> Vec<(f32, f32)> {
    let steps = n + 1;
    let step = -1.0f32 / n as f32;
    let lin: Vec<f32> = (0..steps).map(|i| rb(if i < steps / 2 { 1.0 + step * i as f32 } else { 0.0 - step * (steps - 1 - i) as f32 })).collect();
    let ts: Vec<f32> = lin
        .iter()
        .map(|&t| {
            let a = rb(std::f32::consts::FRAC_PI_2 * t);
            let e = rb(rb(rb(a.cos()) - 1.0) + t);
            rb(t + e)
        })
        .collect();
    let zero_init = 1usize.max((steps as f64 * 0.04) as usize);
    let (mut t, mut dt) = (ts[0], rb(ts[0] - ts[1]));
    let mut out = vec![];
    for s in 1..steps {
        if s > zero_init {
            out.push((t, dt));
        }
        t = rb(t - dt);
        if s < steps - 1 {
            dt = rb(t - ts[s + 1]);
        }
    }
    out
}

/// SinusoidalPosEmb(1024) of a bf16 timestep, rounding like the bf16 PyTorch ops.
fn sinusoid(t: f32, dim: usize) -> Vec<u16> {
    let half = dim / 2;
    let c = -((10000f64).ln() / (half as f64 - 1.0)) as f32;
    let x = rb(1000.0 * t);
    let mut v = vec![0u16; dim];
    for i in 0..half {
        let e = rb(rb(rb(i as f32) * c).exp());
        let a = rb(x * e);
        v[i] = f2b(a.sin());
        v[half + i] = f2b(a.cos());
    }
    v
}

/// Rotary cos/sin tables [maxlen][HD / 2] with the LongRoPE short factors, rounded to bf16 because the reference
/// casts its cos/sin buffers to bf16.
fn rotary_tables(maxlen: usize, ext: &[f64]) -> (Vec<f32>, Vec<f32>) {
    let half = HD / 2;
    let (mut cs, mut sn) = (vec![0f32; maxlen * half], vec![0f32; maxlen * half]);
    for p in 0..maxlen {
        for i in 0..half {
            let inv = 1.0 / 10000f32.powf((2 * i) as f32 / HD as f32);
            let fr = (p as f32 * (1.0 / ext[i] as f32)) * inv;
            cs[p * half + i] = rb(fr.cos());
            sn[p * half + i] = rb(fr.sin());
        }
    }
    (cs, sn)
}

/// Time embedding of every DiT step on the GPU: time_mlp(sin(t)) + delta_time_mlp(sin(0)), [steps][LOC].
fn time_embedding(g: &mut Gpu, sched: &[(f32, f32)], [time1, time2, dtime1, dtime2]: [&Lin; 4]) -> R<Ptr> {
    let ns = sched.len();
    let n = ns * LOC;
    let tin: Vec<u16> = sched.iter().flat_map(|&(t, _)| sinusoid(t, LOC)).collect();
    let zin: Vec<u16> = (0..ns).flat_map(|_| sinusoid(0.0, LOC)).collect();
    let (tin_d, zin_d) = (g.upload_new(&tin)?, g.upload_new(&zin)?);
    let (h1, dt) = (g.alloc_bf16(n)?, g.alloc_bf16(n)?);
    let temb = g.alloc_bf16(n)?;
    time1.fwd(g, tin_d, LOC, h1, LOC, ns)?;
    g.k1("silu_inplace", n, &[P(h1), L(n as i64)])?;
    time2.fwd(g, h1, LOC, temb, LOC, ns)?;
    dtime1.fwd(g, zin_d, LOC, h1, LOC, ns)?;
    g.k1("silu_inplace", n, &[P(h1), L(n as i64)])?;
    dtime2.fwd(g, h1, LOC, dt, LOC, ns)?;
    g.k1("add_inplace", n, &[P(temb), P(dt), L(n as i64)])?;
    for p in [tin_d, zin_d, h1, dt] {
        g.free(p, 2 * n)?;
    }
    Ok(temb)
}

/// The kernels and buffers are laid out for VoxCPM2's dimensions; refuse other checkpoints with a clear message.
fn check_config(c: &json::Json) -> R<()> {
    let expect: &[(&[&str], f64)] = &[
        (&["patch_size"], 4.0),
        (&["feat_dim"], 64.0),
        (&["scalar_quantization_latent_dim"], FSQ as f64),
        (&["scalar_quantization_scale"], 9.0),
        (&["lm_config", "hidden_size"], LM as f64),
        (&["lm_config", "intermediate_size"], LM_FFN as f64),
        (&["lm_config", "num_attention_heads"], NQ as f64),
        (&["lm_config", "num_key_value_heads"], NK as f64),
        (&["lm_config", "kv_channels"], HD as f64),
        (&["encoder_config", "hidden_dim"], LOC as f64),
        (&["encoder_config", "ffn_dim"], LOC_FFN as f64),
        (&["encoder_config", "num_heads"], 16.0),
        (&["encoder_config", "kv_channels"], 128.0),
        (&["dit_config", "hidden_dim"], LOC as f64),
        (&["dit_config", "ffn_dim"], LOC_FFN as f64),
        (&["dit_config", "num_heads"], 16.0),
        (&["dit_config", "kv_channels"], 128.0),
    ];
    for (path, want) in expect {
        let got = path.iter().try_fold(c, |j, k| j.get(k)).and_then(|j| j.num());
        if got != Some(*want) {
            return Err(format!("config.json: {} is {got:?}, this engine supports VoxCPM2 ({want})", path.join(".")).into());
        }
    }
    let flag = |path: &[&str]| path.iter().try_fold(c, |j, k| j.get(k)).and_then(|j| j.bool());
    if flag(&["lm_config", "use_mup"]) != Some(false) || flag(&["residual_lm_no_rope"]) != Some(true) {
        return Err("config.json: this engine supports VoxCPM2 (lm_config.use_mup false, residual_lm_no_rope true)".into());
    }
    Ok(())
}

impl Engine {
    /// Loads the weights and allocates buffers for `b` slots, a KV cache of `maxlen` positions (rounded up to a
    /// multiple of 64) and prompts of up to `tmax` positions.
    pub fn new(model_dir: &str, b: usize, maxlen: usize, tmax: usize, timesteps: usize) -> R<Self> {
        if !(1..=MAX_SLOTS).contains(&b) || !(2..=100).contains(&timesteps) || !(1..=1 << 16).contains(&maxlen) || !(1..=1 << 14).contains(&tmax) {
            return Err(format!("slots must be 1..={MAX_SLOTS}, timesteps 2..=100, KV length 1..=65536, prompt length 1..=16384").into());
        }
        let maxlen = maxlen.next_multiple_of(ATT_CH);
        let mut g = Gpu::new()?;
        let cfg_json = json::parse_json(&std::fs::read(format!("{model_dir}/config.json"))?)?;
        check_config(&cfg_json)?;
        let num =
            |path: &[&str]| path.iter().try_fold(&cfg_json, |j, k| j.get(k)).and_then(|j| j.num()).ok_or_else(|| format!("config.json: no {}", path.join(".")));
        let eps = num(&["lm_config", "rms_norm_eps"]).unwrap_or(1e-5) as f32;
        let ext = cfg_json.get("lm_config").and_then(|j| j.get("rope_scaling")).and_then(|j| j.get("short_factor")).and_then(|j| j.floats());
        let ext = ext.filter(|v| v.len() == HD / 2).ok_or("config.json: lm_config.rope_scaling.short_factor")?;
        let count = |path: &[&str]| -> R<usize> {
            let v = num(path)?;
            if v.fract() != 0.0 || !(1.0..=256.0).contains(&v) {
                return Err(format!("config.json: {} must be an integer in 1..=256", path.join(".")).into());
            }
            Ok(v as usize)
        };
        let base_layers = count(&["lm_config", "num_hidden_layers"])?;
        let res_layers = count(&["residual_lm_num_layers"])?;
        let enc_layers = count(&["encoder_config", "num_layers"])?;
        let dit_layers = count(&["dit_config", "num_layers"])?;
        let vocab = num(&["lm_config", "vocab_size"])?;
        if vocab.fract() != 0.0 || !(1.0..=(1u32 << 24) as f64).contains(&vocab) {
            return Err("config.json: lm_config.vocab_size".into());
        }
        let vocab = vocab as usize;

        let f = st::File::open(&format!("{model_dir}/model.safetensors"))?;
        let arena = Arena::new(&mut g, f.data_bytes(256))?;
        let mut ld = Loader { f: &f, a: arena };
        let cache = Some((b, maxlen));
        let base = ld.tf(&mut g, "base_lm", base_layers, LM, LM_FFN, true, cache)?;
        let res = ld.tf(&mut g, "residual_lm", res_layers, LM, LM_FFN, false, cache)?;
        let enc = ld.tf(&mut g, "feat_encoder.encoder", enc_layers, LOC, LOC_FFN, true, None)?;
        let dit = ld.tf(&mut g, "feat_decoder.estimator.decoder", dit_layers, LOC, LOC_FFN, true, None)?;
        let d = "feat_decoder.estimator";
        let time1 = ld.lin1(&mut g, &format!("{d}.time_mlp.linear_1"), LOC, LOC, true)?;
        let time2 = ld.lin1(&mut g, &format!("{d}.time_mlp.linear_2"), LOC, LOC, true)?;
        let dtime1 = ld.lin1(&mut g, &format!("{d}.delta_time_mlp.linear_1"), LOC, LOC, true)?;
        let dtime2 = ld.lin1(&mut g, &format!("{d}.delta_time_mlp.linear_2"), LOC, LOC, true)?;
        let embed = ld.one(&mut g, "base_lm.embed_tokens.weight", &[vocab, LM])?;
        let enc_in = ld.lin1(&mut g, "feat_encoder.in_proj", LOC, 64, true)?;
        let enc_special = ld.one(&mut g, "feat_encoder.special_token", &[1, 1, 1, LOC])?;
        let enc_to_lm = ld.lin1(&mut g, "enc_to_lm_proj", LM, LOC, true)?;
        let dit_in = ld.lin1(&mut g, &format!("{d}.in_proj"), LOC, 64, true)?;
        let dit_cond = ld.lin1(&mut g, &format!("{d}.cond_proj"), LOC, 64, true)?;
        let dit_out = ld.lin1(&mut g, &format!("{d}.out_proj"), 64, LOC, true)?;
        let fsq_in = ld.lin1(&mut g, "fsq_layer.in_proj", FSQ, LM, true)?;
        let fsq_out = ld.lin1(&mut g, "fsq_layer.out_proj", LM, FSQ, true)?;
        let fusion = ld.lin1(&mut g, "fusion_concat_proj", LM, 2 * LM, true)?;
        let lm_to_dit = ld.lin1(&mut g, "lm_to_dit_proj", LOC, LM, true)?;
        let res_to_dit = ld.lin1(&mut g, "res_to_dit_proj", LOC, LM, true)?;
        let stop_proj = ld.lin1(&mut g, "stop_proj", LM, LM, true)?;
        let stop_head = ld.lin1(&mut g, "stop_head", 2, LM, false)?;
        ld.a.finish(&g)?;
        drop(f);

        let (cs, sn) = rotary_tables(maxlen, &ext);
        let (cos, sin) = (g.upload_new(&cs)?, g.upload_new(&sn)?);

        // scratch sized for the largest pass: the local encoder over every prefill patch
        let rows = (ENC_SEQ * tmax).max(2 * b * DIT_SEQ);
        let sc = Scratch {
            xn: g.alloc_bf16(rows * LM)?,
            qkv: g.alloc_bf16(rows * QKV)?,
            att: g.alloc_bf16(rows * QW)?,
            tmp: g.alloc_bf16(rows * LM)?,
            gu: g.alloc_bf16(rows * 2 * LM_FFN)?,
            act: g.alloc_bf16(rows * LM_FFN)?,
        };

        let sched = schedule(timesteps);
        let temb = time_embedding(&mut g, &sched, [&time1, &time2, &dtime1, &dtime2])?;

        let s = StepBufs::new(&mut g, b, maxlen)?;
        let pf = PrefillBufs::new(&mut g, tmax)?;
        g.sync()?;
        Ok(Engine {
            base,
            res,
            enc,
            dit,
            embed,
            enc_in,
            enc_special,
            enc_to_lm,
            dit_in,
            dit_cond,
            dit_out,
            fsq_in,
            fsq_out,
            fusion,
            lm_to_dit,
            res_to_dit,
            stop_proj,
            stop_head,
            cos,
            sin,
            eps,
            sc,
            b,
            maxlen,
            tmax,
            vocab,
            sched,
            temb,
            cfg: 2.0,
            s,
            pf,
            patches: 0,
            patches_cap: 0,
            max_steps: 0,
            noise_tab: 0,
            noise_bytes: 0,
            min_len: 2,
            g,
        })
    }

    /// Full-sequence pass over `rows` tokens made of independent sequences of length `seq`.
    /// x is the residual stream (overwritten); the final-normed result goes to `out`.
    /// With `cache`, k/v of every position go to slot 0 of the layer caches.
    #[allow(clippy::too_many_arguments)]
    fn full(&self, tf: &Tf, x: Ptr, rows: usize, seq: usize, causal: bool, out: Ptr, cache: bool) -> R<()> {
        let (g, sc, h) = (&self.g, &self.sc, tf.h);
        let r32 = rows as u32;
        let scale = 1.0 / (HD as f32).sqrt();
        g.k("rmsnorm", (r32, 1, 1), 256, 0, &[P(x), P(tf.layers[0].ln1), P(sc.xn), I(h as i32), F(self.eps)])?;
        for (li, l) in tf.layers.iter().enumerate() {
            l.qkv.fwd(g, sc.xn, h, sc.qkv, QKV, rows)?;
            if tf.rope {
                g.k(
                    "rope",
                    (r32, (NQ + NK) as u32, 1),
                    64,
                    0,
                    &[P(sc.qkv), P(self.cos), P(self.sin), P(0), I(rows as i32), I(seq as i32), I(NQ as i32), I(NK as i32), I(HD as i32)],
                )?;
            }
            if cache {
                g.k(
                    "kv_write",
                    (r32, 1, NK as u32),
                    128,
                    0,
                    &[P(sc.qkv), P(tf.kc[li]), P(tf.vc[li]), P(0), I(rows as i32), I(NQ as i32), I(NK as i32), I(HD as i32), I(self.maxlen as i32)],
                )?;
            }
            g.k(
                "attn_full",
                (r32, NQ as u32, 1),
                128,
                ((HD + seq) * 4) as u32,
                &[P(sc.qkv), P(sc.att), I(seq as i32), I(causal as i32), I(NQ as i32), I(NK as i32), I(HD as i32), F(scale)],
            )?;
            self.post_attn(tf, li, x, rows, out)?;
        }
        Ok(())
    }

    /// One decoding position for every slot; positions come from ctr[0] on the device.
    fn step_tf(&self, tf: &Tf, x: Ptr, out: Ptr) -> R<()> {
        let (g, sc, h, b) = (&self.g, &self.sc, tf.h, self.b);
        let scale = 1.0 / (HD as f32).sqrt();
        g.k("rmsnorm", (b as u32, 1, 1), 256, 0, &[P(x), P(tf.layers[0].ln1), P(sc.xn), I(h as i32), F(self.eps)])?;
        for (li, l) in tf.layers.iter().enumerate() {
            l.qkv.fwd(g, sc.xn, h, sc.qkv, QKV, b)?;
            if tf.rope {
                g.k(
                    "rope",
                    (b as u32, (NQ + NK) as u32, 1),
                    64,
                    0,
                    &[P(sc.qkv), P(self.cos), P(self.sin), P(self.s.ctr), I(b as i32), I(0), I(NQ as i32), I(NK as i32), I(HD as i32)],
                )?;
            }
            g.k(
                "kv_write",
                (1, b as u32, NK as u32),
                128,
                0,
                &[P(sc.qkv), P(tf.kc[li]), P(tf.vc[li]), P(self.s.ctr), I(1), I(NQ as i32), I(NK as i32), I(HD as i32), I(self.maxlen as i32)],
            )?;
            let nsp = self.maxlen / ATT_CH;
            g.k(
                "attn_decode",
                (b as u32, NQ as u32, nsp as u32),
                128,
                0,
                &[
                    P(sc.qkv),
                    P(tf.kc[li]),
                    P(tf.vc[li]),
                    P(self.s.att_part),
                    P(self.s.ctr),
                    I(NQ as i32),
                    I(NK as i32),
                    I(HD as i32),
                    I(self.maxlen as i32),
                    F(scale),
                ],
            )?;
            g.k("attn_combine", (b as u32, NQ as u32, 1), HD as u32, 0, &[P(self.s.att_part), P(sc.att), I(NQ as i32), I(HD as i32), I(nsp as i32)])?;
            self.post_attn(tf, li, x, b, out)?;
        }
        Ok(())
    }

    /// o_proj, residual, MLP, residual; the trailing norm is the next layer's input norm (or the final norm into `out`).
    fn post_attn(&self, tf: &Tf, li: usize, x: Ptr, rows: usize, out: Ptr) -> R<()> {
        let (g, sc, h, l) = (&self.g, &self.sc, tf.h, &tf.layers[li]);
        let r32 = rows as u32;
        l.o.fwd(g, sc.att, QW, sc.tmp, h, rows)?;
        g.k("add_rmsnorm", (r32, 1, 1), 256, 0, &[P(x), P(sc.tmp), P(l.ln2), P(sc.xn), I(h as i32), F(self.eps)])?;
        l.gu.fwd(g, sc.xn, h, sc.gu, 2 * tf.ffn, rows)?;
        g.k1("silu_mul", rows * tf.ffn, &[P(sc.gu), P(sc.act), I(rows as i32), I(tf.ffn as i32)])?;
        l.down.fwd(g, sc.act, tf.ffn, sc.tmp, h, rows)?;
        let (w, o) = if li + 1 == tf.layers.len() { (tf.norm, out) } else { (tf.layers[li + 1].ln1, sc.xn) };
        g.k("add_rmsnorm", (r32, 1, 1), 256, 0, &[P(x), P(sc.tmp), P(w), P(o), I(h as i32), F(self.eps)])
    }

    /// Local encoder over n patches [n, 4, 64] -> CLS embeddings projected to the LM width (n x LM).
    #[allow(clippy::too_many_arguments)]
    fn encode(&self, feat: Ptr, n: usize, xp: Ptr, seqb: Ptr, hout: Ptr, cls: Ptr, y: Ptr) -> R<()> {
        let g = &self.g;
        self.enc_in.fwd(g, feat, 64, xp, LOC, n * 4)?;
        g.k("enc_build", (n as u32, ENC_SEQ as u32, 1), 256, 0, &[P(seqb), P(self.enc_special), P(xp), I(LOC as i32)])?;
        self.full(&self.enc, seqb, n * ENC_SEQ, ENC_SEQ, false, hout, false)?;
        g.k("gather_rows", (n as u32, 1, 1), 256, 0, &[P(hout), P(cls), I(ENC_SEQ as i32), I(0), I(1), I(LOC as i32)])?;
        self.enc_to_lm.fwd(g, cls, LOC, y, LM, n)
    }

    /// Prefill the prompt (shared by all slots), then copy its state into every slot.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(&mut self, tokens: &[i32], tm: &[i32], am: &[i32], feat: &[f32], noise_table: Option<&[f32]>, seeds: &[u32], max_steps: usize) -> R<()> {
        let t = tokens.len();
        if t == 0 || t > self.tmax || t + max_steps > self.maxlen {
            return Err(format!("prompt {t} + {max_steps} steps does not fit (max prompt {}, KV length {})", self.tmax, self.maxlen).into());
        }
        if tm.len() != t || am.len() != t || feat.len() != t * PATCH || seeds.is_empty() || seeds.len() > self.b {
            return Err("prefill: inconsistent prompt arrays".into());
        }
        if let Some(&id) = tokens.iter().find(|&&id| id < 0 || id as usize >= self.vocab) {
            return Err(format!("prefill: token id {id} is outside the vocabulary of {}", self.vocab).into());
        }
        self.max_steps = max_steps;
        if max_steps > self.patches_cap {
            if self.patches_cap > 0 {
                self.g.free(self.patches, 2 * self.b * self.patches_cap * PATCH)?;
            }
            self.patches = self.g.alloc_bf16(self.b * max_steps * PATCH)?;
            self.patches_cap = max_steps;
        }
        if self.noise_tab != 0 {
            self.g.free(self.noise_tab, self.noise_bytes)?;
            self.noise_tab = 0;
        }
        if let Some(tab) = noise_table {
            if tab.len() < max_steps * PATCH {
                return Err("prefill: the noise table is shorter than the generation".into());
            }
            self.noise_tab = self.g.upload_new(tab)?;
            self.noise_bytes = std::mem::size_of_val(tab);
        }
        let g = &self.g;
        g.upload(self.pf.tok, tokens)?;
        g.upload(self.pf.tm, tm)?;
        g.upload(self.pf.am, am)?;
        let fb: Vec<u16> = feat.iter().map(|&v| f2b(v)).collect();
        g.upload(self.pf.feat, &fb)?;
        let s: Vec<u32> = (0..self.b).map(|i| seeds[i % seeds.len()]).collect();
        g.upload(self.s.seeds, &s)?;
        let t32 = t as u32;
        self.encode(self.pf.feat, t, self.pf.xp, self.pf.eseq, self.pf.eout, self.pf.cls, self.pf.fe)?;
        g.k("prefill_combine", (t32, 1, 1), 256, 0, &[P(self.pf.x), P(self.pf.tok), P(self.embed), P(self.pf.fe), P(self.pf.tm), P(self.pf.am), I(LM as i32)])?;
        self.full(&self.base, self.pf.x, t, t, true, self.pf.out, true)?;
        self.fsq_in.fwd(g, self.pf.out, LM, self.pf.q, FSQ, t)?;
        g.k1("fsq_act", t * FSQ, &[P(self.pf.q), L((t * FSQ) as i64), F(9.0)])?;
        self.fsq_out.fwd(g, self.pf.q, FSQ, self.pf.fsq, LM, t)?;
        g.k(
            "prefill_mix",
            (t32, 1, 1),
            256,
            0,
            &[P(self.pf.out), P(self.pf.fsq), P(self.pf.fe), P(self.pf.tm), P(self.pf.am), P(self.pf.mix), P(self.pf.resin), I(LM as i32)],
        )?;
        g.k("broadcast_row", (self.b as u32, 1, 1), 256, 0, &[P(self.pf.mix), P(self.s.lm_h), I(t as i32), I(LM as i32), I(LM as i32)])?;
        self.fusion.fwd(g, self.pf.resin, 2 * LM, self.pf.x, LM, t)?;
        self.full(&self.res, self.pf.x, t, t, true, self.pf.rout, true)?;
        g.k("broadcast_row", (self.b as u32, 1, 1), 256, 0, &[P(self.pf.rout), P(self.s.res_h), I(t as i32), I(LM as i32), I(LM as i32)])?;
        let last: Vec<u16> = (0..self.b).flat_map(|_| fb[(t - 1) * PATCH..t * PATCH].to_vec()).collect();
        g.upload(self.s.cond, &last)?;
        if self.b > 1 {
            for tf in [&self.base, &self.res] {
                for c in tf.kc.iter().chain(tf.vc.iter()) {
                    g.k(
                        "kv_replicate",
                        (t32, NK as u32, (self.b - 1) as u32),
                        128,
                        0,
                        &[P(*c), I(self.b as i32), I(NK as i32), I(self.maxlen as i32), I(HD as i32), I(t as i32)],
                    )?;
                }
            }
        }
        g.upload(self.s.ctr, &[t as i32, 0, 0, 0])?;
        g.upload(self.s.stop, &vec![-1i32; self.b])?;
        g.sync()
    }

    /// One decoding step for all slots. Pure device work, so it can be captured into a CUDA graph.
    /// `dit0` (debug) copies the first DiT output of the step there.
    pub fn step(&self, dit0: Option<Ptr>) -> R<()> {
        let (g, b) = (&self.g, self.b);
        let b32 = b as u32;
        // DiT conditioning: mu = [lm_to_dit(lm_h) | res_to_dit(res_h)], cond = previous patch
        self.lm_to_dit.fwd(g, self.s.lm_h, LM, self.s.mu, LM, b)?;
        self.res_to_dit.fwd(g, self.s.res_h, LM, off(self.s.mu, LOC), LM, b)?;
        self.dit_cond.fwd(g, self.s.cond, 64, self.s.condp, LOC, b * 4)?;
        g.k("noise", (b32, 1, 1), 256, 0, &[P(self.s.x), P(self.noise_tab), P(self.s.ctr + 4), P(self.s.seeds), I(b as i32)])?;
        for (i, &(_, dt)) in self.sched.iter().enumerate() {
            self.dit_in.fwd(g, self.s.x, 64, self.s.xp, LOC, b * 4)?;
            g.k(
                "dit_build",
                (2 * b32, DIT_SEQ as u32, 1),
                256,
                0,
                &[P(self.s.dit_seq), P(self.s.mu), P(off(self.temb, i * LOC)), P(self.s.condp), P(self.s.xp), I(b as i32), I(LOC as i32)],
            )?;
            self.full(&self.dit, self.s.dit_seq, 2 * b * DIT_SEQ, DIT_SEQ, false, self.s.dit_hout, false)?;
            g.k("gather_rows", (2 * b32, 4, 1), 256, 0, &[P(self.s.dit_hout), P(self.s.dit_tok), I(DIT_SEQ as i32), I(DIT_X0 as i32), I(4), I(LOC as i32)])?;
            self.dit_out.fwd(g, self.s.dit_tok, LOC, self.s.dit_o, 64, 2 * b * 4)?;
            if let (0, Some(p)) = (i, dit0) {
                g.copy(p, self.s.dit_o, 2 * b * PATCH * 2)?;
            }
            g.k("cfg_euler", (b32, 1, 1), 256, 0, &[P(self.s.x), P(self.s.dit_o), I(b as i32), F(dt), F(self.cfg)])?;
        }
        // the new patch through the local encoder -> next LM input (also the right half of the fusion input)
        self.encode(self.s.x, b, self.s.xp, self.s.enc_seq, self.s.enc_hout, self.s.enc_cls, self.s.base_x)?;
        g.k1("copy2d", b * LM, &[P(self.s.base_x), I(LM as i32), P(off(self.s.fus, LM)), I((2 * LM) as i32), I(b as i32), I(LM as i32)])?;
        // stop decision uses the hidden state that produced this patch
        self.stop_proj.fwd(g, self.s.lm_h, LM, self.s.stop_h, LM, b)?;
        g.k1("silu_inplace", b * LM, &[P(self.s.stop_h), L((b * LM) as i64)])?;
        self.stop_head.fwd(g, self.s.stop_h, LM, self.s.stop_l, 2, b)?;
        // base LM -> FSQ -> residual LM
        self.step_tf(&self.base, self.s.base_x, self.s.lm_raw)?;
        self.fsq_in.fwd(g, self.s.lm_raw, LM, self.s.q, FSQ, b)?;
        g.k1("fsq_act", b * FSQ, &[P(self.s.q), L((b * FSQ) as i64), F(9.0)])?;
        self.fsq_out.fwd(g, self.s.q, FSQ, self.s.lm_h, LM, b)?;
        g.k1("copy2d", b * LM, &[P(self.s.lm_h), I(LM as i32), P(self.s.fus), I((2 * LM) as i32), I(b as i32), I(LM as i32)])?;
        self.fusion.fwd(g, self.s.fus, 2 * LM, self.s.res_x, LM, b)?;
        self.step_tf(&self.res, self.s.res_x, self.s.res_h)?;
        g.copy(self.s.cond, self.s.x, b * PATCH * 2)?;
        g.k(
            "step_end",
            (1, 1, 1),
            256,
            0,
            &[P(self.s.stop_l), P(self.s.x), P(self.patches), P(self.s.stop), P(self.s.ctr), I(b as i32), I(self.min_len), I(self.max_steps as i32)],
        )
    }

    /// Runs decoding steps until every slot has stopped or `max_steps` is reached (`done` steps already ran);
    /// returns the number of patches of every slot.
    pub fn decode(&mut self, graph: bool, done: usize) -> R<Vec<usize>> {
        let mut cg = None;
        if graph {
            self.g.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            self.step(None)?;
            cg = self.g.stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?;
        }
        let mut stops = self.g.download::<i32>(self.s.stop, self.b)?;
        for _ in done..self.max_steps {
            if stops.iter().all(|&s| s >= 0) {
                break;
            }
            match &cg {
                Some(gr) => gr.launch()?,
                None => self.step(None)?,
            }
            stops = self.g.download::<i32>(self.s.stop, self.b)?;
        }
        Ok(stops.iter().map(|&s| if s >= 0 { s as usize + 1 } else { self.max_steps }).collect())
    }

    /// The first `lens[s]` generated patches of every slot s, as f32 [n][4][64].
    pub fn latents(&self, lens: &[usize]) -> R<Vec<Vec<f32>>> {
        let all = self.g.download::<u16>(self.patches, self.b * self.max_steps * PATCH)?;
        Ok(lens.iter().enumerate().map(|(s, &n)| all[s * self.max_steps * PATCH..(s * self.max_steps + n) * PATCH].iter().map(|&h| b2f(h)).collect()).collect())
    }

    /// Hidden states of the last prompt position after prefill: (base LM, residual LM).
    pub fn prefill_last(&self, t: usize) -> R<(Vec<f32>, Vec<f32>)> {
        let at = |p: Ptr| self.g.download_bf16(off(p, (t - 1) * LM), LM);
        Ok((at(self.pf.out)?, at(self.pf.rout)?))
    }
}
