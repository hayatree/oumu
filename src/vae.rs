//! VoxCPM2's AudioVAE (V2) on the GPU in fp32, like the reference: encoder (16 kHz audio -> 64-dim latents at 25 Hz)
//! and decoder (latents -> 48 kHz audio). Convolutions are cuBLAS GEMMs over [channels][time] activations (bias as an
//! extra row of ones, im2col for strided ones, one GEMM per transposed convolution); depthwise convolutions and their
//! Snake activations are one fused kernel.
use crate::gpu::{A, Gpu, Ptr, R};
use crate::json::{Json, parse_json};
use crate::pth::Pth;

pub struct Config {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub sr_bin_boundaries: Vec<usize>,
    pub sample_rate: usize,
    pub out_sample_rate: usize,
}

impl Config {
    pub fn from_model_config(j: &Json) -> R<Self> {
        let c = j.get("audio_vae_config").ok_or("config.json: no audio_vae_config")?;
        let bad = |k: &str| format!("config.json: audio_vae_config.{k}");
        let n = |k: &str, d: usize| c.get(k).map(|v| v.uint().filter(|&n| n > 0).ok_or_else(|| bad(k))).transpose().map(|v| v.unwrap_or(d));
        let v = |k: &str, d: &[usize]| {
            c.get(k).map(|v| v.uints().filter(|v| !v.is_empty()).ok_or_else(|| bad(k))).transpose().map(|v| v.unwrap_or_else(|| d.to_vec()))
        };
        let cfg = Config {
            encoder_dim: n("encoder_dim", 128)?,
            encoder_rates: v("encoder_rates", &[2, 5, 8, 8])?,
            latent_dim: n("latent_dim", 64)?,
            decoder_dim: n("decoder_dim", 2048)?,
            decoder_rates: v("decoder_rates", &[8, 6, 5, 2, 2, 2])?,
            sr_bin_boundaries: v("sr_bin_boundaries", &[20000, 30000, 40000])?,
            sample_rate: n("sample_rate", 16000)?,
            out_sample_rate: n("out_sample_rate", 48000)?,
        };
        // the buffers are sized for VoxCPM2's AudioVAE; other shapes are refused rather than half supported
        let shape = (cfg.encoder_dim, &cfg.encoder_rates[..], cfg.latent_dim, cfg.decoder_dim, &cfg.decoder_rates[..]);
        if shape != (128, &[2, 5, 8, 8][..], 64, 2048, &[8, 6, 5, 2, 2, 2][..]) {
            return Err(bad("dimensions (this engine supports VoxCPM2's AudioVAE)").into());
        }
        if ![cfg.sample_rate, cfg.out_sample_rate].iter().all(|r| (1000..=768_000).contains(r)) {
            return Err(bad("sample_rate / out_sample_rate").into());
        }
        Ok(cfg)
    }
    /// Input samples per latent frame.
    pub fn hop(&self) -> usize {
        self.encoder_rates.iter().product()
    }
    /// Output samples per latent frame.
    pub fn decode_hop(&self) -> usize {
        self.decoder_rates.iter().product()
    }
}

/// Causal residual unit: x + conv1x1(snake(dwconv7_dil(snake(x)))).
struct Ru {
    dil: usize,
    a1: Ptr,
    dw_w: Ptr,
    dw_b: Ptr,
    a2: Ptr,
    pw: Ptr, // [C][C + 1], bias in the last column
}

struct DecBlock {
    cin: usize,
    cout: usize,
    s: usize,
    scale: Ptr,
    shift: Ptr,
    alpha: Ptr,
    tw: Ptr, // [s * cout][2 * cin]
    tb: Ptr,
    ru: Vec<Ru>,
}

struct EncBlock {
    cin: usize,
    cout: usize,
    s: usize,
    ru: Vec<Ru>,
    alpha: Ptr,
    w: Ptr, // [cout][cin * 2s + 1]
}

pub struct Vae {
    pub cfg: Config,
    dec_dw_w: Ptr,
    dec_dw_b: Ptr,
    dec_pw: Ptr, // [decoder_dim][latent + 1]
    dec: Vec<DecBlock>,
    out_alpha: Ptr,
    out_w: Ptr,
    out_b: Ptr,
    out_c: usize,
    enc_in: Ptr, // [encoder_dim][8]
    enc: Vec<EncBlock>,
    mu: Ptr, // [latent][enc_out * 3 + 1]
    enc_out: usize,
    ws: [(Ptr, usize); 3], // reusable work buffers (pointer, f32 capacity)
}

/// Appends the bias as a last column: [rows][cols] -> [rows][cols + 1].
fn with_bias(w: &[f32], b: &[f32], rows: usize) -> Vec<f32> {
    let cols = w.len() / rows;
    let mut o = Vec::with_capacity(rows * (cols + 1));
    for r in 0..rows {
        o.extend_from_slice(&w[r * cols..(r + 1) * cols]);
        o.push(b[r]);
    }
    o
}

fn f32s(p: &Pth, name: &str, numel: usize) -> R<Vec<f32>> {
    let (v, _) = p.f32s(name)?;
    if v.len() != numel {
        return Err(format!("audiovae: {name} has {} values, expected {numel}", v.len()).into());
    }
    Ok(v)
}

fn wn(p: &Pth, prefix: &str, shape: &[usize]) -> R<Vec<f32>> {
    let (w, s) = p.weight_norm(prefix)?;
    if s != shape {
        return Err(format!("audiovae: {prefix} has shape {s:?}, expected {shape:?}").into());
    }
    Ok(w)
}

fn load_ru(g: &mut Gpu, p: &Pth, pre: &str, c: usize, dil: usize) -> R<Ru> {
    let dw = wn(p, &format!("{pre}.block.1"), &[c, 1, 7])?;
    let pw = wn(p, &format!("{pre}.block.3"), &[c, c, 1])?;
    Ok(Ru {
        dil,
        a1: g.upload_new(&f32s(p, &format!("{pre}.block.0.alpha"), c)?)?,
        dw_w: g.upload_new(&dw)?,
        dw_b: g.upload_new(&f32s(p, &format!("{pre}.block.1.bias"), c)?)?,
        a2: g.upload_new(&f32s(p, &format!("{pre}.block.2.alpha"), c)?)?,
        pw: g.upload_new(&with_bias(&pw, &f32s(p, &format!("{pre}.block.3.bias"), c)?, c))?,
    })
}

impl Vae {
    pub fn load(g: &mut Gpu, model_dir: &str) -> R<Self> {
        let cj = parse_json(&std::fs::read(format!("{model_dir}/config.json"))?)?;
        let cfg = Config::from_model_config(&cj)?;
        let p = Pth::open(&format!("{model_dir}/audiovae.pth"))?;
        let lat = cfg.latent_dim;
        let dil = [1, 3, 9];

        // decoder
        let dec_dw_w = g.upload_new(&wn(&p, "decoder.model.0", &[lat, 1, 7])?)?;
        let dec_dw_b = g.upload_new(&f32s(&p, "decoder.model.0.bias", lat)?)?;
        let pw = wn(&p, "decoder.model.1", &[cfg.decoder_dim, lat, 1])?;
        let dec_pw = g.upload_new(&with_bias(&pw, &f32s(&p, "decoder.model.1.bias", cfg.decoder_dim)?, cfg.decoder_dim))?;
        let bounds: Vec<usize> = cfg.sr_bin_boundaries.clone();
        let sr_idx = bounds.iter().filter(|&&b| b < cfg.out_sample_rate).count(); // torch.bucketize, right=False
        let mut dec = vec![];
        for (i, &s) in cfg.decoder_rates.iter().enumerate() {
            let (cin, cout) = (cfg.decoder_dim >> i, cfg.decoder_dim >> (i + 1));
            let m = 2 + i;
            let row = |name: &str| -> R<Vec<f32>> {
                let v = f32s(&p, &format!("decoder.sr_cond_model.{m}.{name}.weight"), (bounds.len() + 1) * cin)?;
                Ok(v[sr_idx * cin..(sr_idx + 1) * cin].to_vec())
            };
            // transposed conv weight [cin][cout][2s] -> [(r, o)][(h, c)] = w[c][o][r + h s], rearranged on the GPU
            let w = g.upload_new(&wn(&p, &format!("decoder.model.{m}.block.1"), &[cin, cout, 2 * s])?)?;
            let tw = g.alloc(s * cout * 2 * cin * 4)?;
            g.k1("vae_tconv_w", s * cout * 2 * cin, &[A::P(w), A::P(tw), A::I(cin as i32), A::I(cout as i32), A::I(s as i32)])?;
            g.free(w, cin * cout * 2 * s * 4)?;
            let mut ru = vec![];
            for (j, &d) in dil.iter().enumerate() {
                ru.push(load_ru(g, &p, &format!("decoder.model.{m}.block.{}", j + 2), cout, d)?);
            }
            dec.push(DecBlock {
                cin,
                cout,
                s,
                scale: g.upload_new(&row("scale_embed")?)?,
                shift: g.upload_new(&row("bias_embed")?)?,
                alpha: g.upload_new(&f32s(&p, &format!("decoder.model.{m}.block.0.alpha"), cin)?)?,
                tw,
                tb: g.upload_new(&f32s(&p, &format!("decoder.model.{m}.block.1.bias"), cout)?)?,
                ru,
            });
        }
        let out_c = cfg.decoder_dim >> cfg.decoder_rates.len();
        if out_c > 32 {
            return Err("audiovae: more than 32 channels before the output layer are not supported".into());
        }
        let m = 2 + cfg.decoder_rates.len();
        let out_alpha = g.upload_new(&f32s(&p, &format!("decoder.model.{m}.alpha"), out_c)?)?;
        let out_w = g.upload_new(&wn(&p, &format!("decoder.model.{}", m + 1), &[1, out_c, 7])?)?;
        let out_b = g.upload_new(&f32s(&p, &format!("decoder.model.{}.bias", m + 1), 1)?)?;

        // encoder
        let e0 = cfg.encoder_dim;
        let w = wn(&p, "encoder.block.0", &[e0, 1, 7])?;
        let enc_in = g.upload_new(&with_bias(&w, &f32s(&p, "encoder.block.0.bias", e0)?, e0))?;
        let mut enc = vec![];
        let mut c = e0;
        for (j, &s) in cfg.encoder_rates.iter().enumerate() {
            let (cin, cout) = (c, 2 * c);
            let pre = format!("encoder.block.{}", j + 1);
            let mut ru = vec![];
            for (k, &d) in dil.iter().enumerate() {
                ru.push(load_ru(g, &p, &format!("{pre}.block.{k}"), cin, d)?);
            }
            let w = wn(&p, &format!("{pre}.block.4"), &[cout, cin, 2 * s])?;
            enc.push(EncBlock {
                cin,
                cout,
                s,
                ru,
                alpha: g.upload_new(&f32s(&p, &format!("{pre}.block.3.alpha"), cin)?)?,
                w: g.upload_new(&with_bias(&w, &f32s(&p, &format!("{pre}.block.4.bias"), cout)?, cout))?,
            });
            c = cout;
        }
        let w = wn(&p, "encoder.fc_mu", &[lat, c, 3])?;
        let mu = g.upload_new(&with_bias(&w, &f32s(&p, "encoder.fc_mu.bias", lat)?, lat))?;
        Ok(Vae { cfg, dec_dw_w, dec_dw_b, dec_pw, dec, out_alpha, out_w, out_b, out_c, enc_in, enc, mu, enc_out: c, ws: [(0, 0); 3] })
    }

    /// Work buffer i with room for n floats (grown, never shrunk).
    fn buf(&mut self, g: &mut Gpu, i: usize, n: usize) -> R<Ptr> {
        if self.ws[i].1 < n {
            if self.ws[i].1 > 0 {
                g.free(self.ws[i].0, self.ws[i].1 * 4)?;
            }
            self.ws[i] = (g.alloc(n * 4)?, n);
        }
        Ok(self.ws[i].0)
    }

    /// x[C][L] += conv1x1(snake(dwconv(snake(x)))), h a work buffer of (C + 1) * L floats.
    fn ru(g: &Gpu, u: &Ru, x: Ptr, h: Ptr, c: usize, l: usize) -> R<()> {
        g.k(
            "vae_dw",
            (l.div_ceil(512) as u32, c as u32 + 1, 1),
            512,
            0,
            &[A::P(x), A::P(h), A::P(u.dw_w), A::P(u.dw_b), A::P(u.a1), A::P(u.a2), A::I(c as i32), A::I(l as i32), A::I(u.dil as i32), A::I(1)],
        )?;
        g.sgemm(l, c, c + 1, h, l, u.pw, c + 1, x, l, 1.0)
    }

    /// Latents [frames][latent_dim] (time-major) -> audio at out_sample_rate, decode_hop() samples per frame.
    pub fn decode(&mut self, g: &mut Gpu, z: &[f32]) -> R<Vec<f32>> {
        let lat = self.cfg.latent_dim;
        let t = z.len() / lat;
        if t == 0 {
            return Ok(vec![]);
        }
        // largest activation, transposed-conv input and GEMM output, in floats
        let (mut act, mut inp, mut len) = (self.cfg.decoder_dim * t, (lat + 1) * t, t);
        for b in &self.dec {
            inp = inp.max(2 * b.cin * len).max((b.cout + 1) * len * b.s);
            len *= b.s;
            act = act.max(b.cout * len);
        }
        let x = self.buf(g, 0, act)?;
        let p = self.buf(g, 1, act)?;
        let h = self.buf(g, 2, inp)?;
        let mut zt = vec![0f32; lat * t];
        for (i, f) in z.chunks(lat).enumerate() {
            for (c, &v) in f.iter().enumerate() {
                zt[c * t + i] = v;
            }
        }
        g.upload(p, &zt)?;
        let null = 0;
        g.k(
            "vae_dw",
            (t.div_ceil(512) as u32, lat as u32 + 1, 1),
            512,
            0,
            &[A::P(p), A::P(h), A::P(self.dec_dw_w), A::P(self.dec_dw_b), A::P(null), A::P(null), A::I(lat as i32), A::I(t as i32), A::I(1), A::I(1)],
        )?;
        g.sgemm(t, self.cfg.decoder_dim, lat + 1, h, t, self.dec_pw, lat + 1, x, t, 0.0)?;
        let mut l = t;
        for b in &self.dec {
            g.k1("vae_tconv_in", b.cin * l, &[A::P(x), A::P(h), A::P(b.scale), A::P(b.shift), A::P(b.alpha), A::I(b.cin as i32), A::I(l as i32)])?;
            g.sgemm(l, b.s * b.cout, 2 * b.cin, h, l, b.tw, 2 * b.cin, p, l, 0.0)?;
            g.k1("vae_unshuffle", b.cout * l * b.s, &[A::P(p), A::P(x), A::P(b.tb), A::I(b.cout as i32), A::I(l as i32), A::I(b.s as i32)])?;
            l *= b.s;
            for u in &b.ru {
                Self::ru(g, u, x, h, b.cout, l)?;
            }
        }
        g.k(
            "vae_out",
            (l.div_ceil(256) as u32, 1, 1),
            256,
            0,
            &[A::P(x), A::P(p), A::P(self.out_w), A::P(self.out_b), A::P(self.out_alpha), A::I(self.out_c as i32), A::I(l as i32)],
        )?;
        g.download::<f32>(p, l)
    }

    /// Audio at sample_rate, length a multiple of hop() -> latents [frames][latent_dim] (time-major), the encoder's mu.
    pub fn encode(&mut self, g: &mut Gpu, wav: &[f32]) -> R<Vec<f32>> {
        let hop = self.cfg.hop();
        let l0 = wav.len();
        if l0 == 0 || !l0.is_multiple_of(hop) {
            return Err(format!("audiovae: encoder input length {l0} is not a positive multiple of {hop}").into());
        }
        let e0 = self.cfg.encoder_dim;
        let (mut act, mut inp, mut l) = (e0 * l0, 8 * l0, l0);
        for b in &self.enc {
            inp = inp.max((b.cin + 1) * l);
            l /= b.s;
            inp = inp.max((b.cin * 2 * b.s + 1) * l);
            act = act.max(b.cout * l);
        }
        inp = inp.max((self.enc_out * 3 + 1) * l);
        let x = self.buf(g, 0, act)?;
        let p = self.buf(g, 1, act.max(l0))?;
        let h = self.buf(g, 2, inp)?;
        let null = 0;
        g.upload(p, wav)?;
        let im2col = |g: &Gpu, x: Ptr, alpha: Ptr, c: usize, l: usize, k: usize, s: usize, pad: usize, lout: usize| {
            g.k1(
                "vae_im2col",
                (c * k + 1) * lout,
                &[A::P(x), A::P(h), A::P(alpha), A::I(c as i32), A::I(l as i32), A::I(k as i32), A::I(s as i32), A::I(pad as i32), A::I(lout as i32), A::I(1)],
            )
        };
        im2col(g, p, null, 1, l0, 7, 1, 6, l0)?;
        g.sgemm(l0, e0, 8, h, l0, self.enc_in, 8, x, l0, 0.0)?;
        let (mut x, mut p) = (x, p);
        let mut l = l0;
        for b in &self.enc {
            for u in &b.ru {
                Self::ru(g, u, x, h, b.cin, l)?;
            }
            let (k, lout) = (2 * b.s, l / b.s);
            let pad = 2 * b.s.div_ceil(2) - b.s % 2;
            im2col(g, x, b.alpha, b.cin, l, k, b.s, pad, lout)?;
            g.sgemm(lout, b.cout, b.cin * k + 1, h, lout, b.w, b.cin * k + 1, p, lout, 0.0)?;
            std::mem::swap(&mut x, &mut p);
            l = lout;
        }
        let lat = self.cfg.latent_dim;
        im2col(g, x, null, self.enc_out, l, 3, 1, 2, l)?;
        g.sgemm(l, lat, self.enc_out * 3 + 1, h, l, self.mu, self.enc_out * 3 + 1, p, l, 0.0)?;
        let mu = g.download::<f32>(p, lat * l)?;
        let mut out = vec![0f32; lat * l];
        for c in 0..lat {
            for i in 0..l {
                out[i * lat + c] = mu[c * l + i];
            }
        }
        Ok(out)
    }
}
