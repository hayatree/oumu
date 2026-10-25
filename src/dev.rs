//! Development tools: each stage against references from scripts/hf_*.py, and decoding benchmarks.
use crate::gpu::{self, R, rb};
use crate::tts::{DIM, PATCH, encode_wav};
use crate::{Args, audio, json, model, prompt, st, tok, vae};
use std::time::Instant;

/// Relative L2 error and max abs error of a against reference r (of the same length).
pub fn cmp(a: &[f32], r: &[f32]) -> R<(f64, f64)> {
    if a.len() != r.len() {
        return Err(format!("comparing {} values with a reference of {}", a.len(), r.len()).into());
    }
    let (mut d2, mut r2, mut mx) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(r) {
        let d = (*x - *y) as f64;
        d2 += d * d;
        r2 += (*y as f64) * (*y as f64);
        mx = mx.max(d.abs());
    }
    Ok(((d2 / r2.max(1e-30)).sqrt(), mx))
}

/// [n, 64, 4] (channel-major, as the reference stores patches) -> [n, 4, 64]
fn to_time_major(v: &[f32]) -> Vec<f32> {
    let n = v.len() / 256;
    let mut o = vec![0f32; v.len()];
    for p in 0..n {
        for c in 0..64 {
            for t in 0..4 {
                o[p * 256 + t * 64 + c] = v[p * 256 + c * 4 + t];
            }
        }
    }
    o
}

/// Time every GEMM shape the decoder uses, custom kernels vs cuBLAS, interleaved so GPU sharing hits both alike.
/// Weights rotate through 1 GiB so nothing is served from L2.
fn gemm_bench(g: &mut gpu::Gpu) -> R<()> {
    let pool = 1usize << 29; // bf16 elements
    let w = g.alloc_bf16(pool)?;
    let x = g.alloc_bf16(128 * 12288)?;
    let y = g.alloc_bf16(128 * 12288)?;
    let shapes = [(2560, 2048), (12288, 2048), (2048, 6144), (2048, 2048), (2560, 1024), (8192, 1024), (1024, 4096), (1024, 2048)];
    for m in [1usize, 4, 8, 20, 44, 88] {
        for &(n, k) in &shapes {
            let mut res = [0f64; 2];
            for round in 0..3 {
                for (i, cublas) in [false, true].into_iter().enumerate() {
                    g.cublas_only = cublas;
                    let iters = 40;
                    let mut off = 0usize;
                    g.sync()?;
                    let t = Instant::now();
                    for _ in 0..iters {
                        if off + n * k > pool {
                            off = 0;
                        }
                        g.gemm(x, k, w + 2 * off as u64, y, n, m, n, k, 0.0)?;
                        off += n * k;
                    }
                    g.sync()?;
                    let dt = t.elapsed().as_secs_f64() / iters as f64;
                    if round > 0 {
                        res[i] += dt / 2.0;
                    }
                }
            }
            let gbs = |t: f64| (n * k * 2) as f64 / t / 1e9;
            println!(
                "M {m:3} N {n:5} K {k:5}  custom {:7.1} us {:5.0} GB/s | cublas {:7.1} us {:5.0} GB/s",
                res[0] * 1e6,
                gbs(res[0]),
                res[1] * 1e6,
                gbs(res[1])
            );
        }
    }
    g.cublas_only = false;
    Ok(())
}

pub fn check_vae(a: &Args, model_dir: &str, p: &str) -> R<()> {
    let r = st::File::open(p)?;
    let mut g = gpu::Gpu::new()?;
    let t = Instant::now();
    let mut v = vae::Vae::load(&mut g, model_dir)?;
    g.sync()?;
    println!("AudioVAE loaded in {:.2}s", t.elapsed().as_secs_f64());
    let (wav, _) = r.f32s("wav16")?;
    if let Some(src) = a.get("--wav") {
        // this engine's resampler against librosa (soxr), and what it changes in the latents
        let (y, sr) = audio::read_wav(src)?;
        let mut y = audio::resample(&y, sr, 16000)?;
        y.resize(wav.len(), 0.0);
        let (rel, mx) = cmp(&y, &wav)?;
        println!("resampled vs librosa      rel {rel:.2e} max {mx:.3e}");
        let (mine, theirs) = (v.encode(&mut g, &y)?, v.encode(&mut g, &wav)?);
        let (rel, mx) = cmp(&mine, &theirs)?;
        println!("latents of the two inputs rel {rel:.2e} max {mx:.3e}");
        let (rd, _) = r.f32s("mu_default")?;
        let (rel, mx) = cmp(&theirs, &rd)?;
        println!("fp32 vs PyTorch default   rel {rel:.2e} max {mx:.3e}");
    }
    for rep in 0..2 {
        let t = Instant::now();
        let mu = v.encode(&mut g, &wav)?;
        let te = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (z, _) = r.f32s("mu_fp32")?;
        let w = v.decode(&mut g, &z)?;
        let td = t.elapsed().as_secs_f64();
        println!("rep{rep}: encode {:.1}s audio in {te:.3}s, decode {} frames in {td:.3}s", wav.len() as f64 / 16000.0, z.len() / 64);
        if rep == 0 {
            for k in ["mu_fp32", "mu_default"] {
                let (rv, _) = r.f32s(k)?;
                let (rel, mx) = cmp(&mu, &rv)?;
                println!("  encode vs {k:16} rel {rel:.2e} max {mx:.3e}");
            }
            for k in ["wav48_fp32", "wav48_default"] {
                let (rv, _) = r.f32s(k)?;
                let (rel, mx) = cmp(&w, &rv)?;
                println!("  decode vs {k:16} rel {rel:.2e} max {mx:.3e}");
            }
        }
    }
    Ok(())
}

/// `p`: [{"text", "ids"}, ...]
pub fn check_tokens(model_dir: &str, p: &str) -> R<()> {
    let tk = tok::Tok::load(model_dir)?;
    let json::Json::Arr(items) = json::parse_json(&std::fs::read(p)?)? else { return Err("expected a list".into()) };
    let mut same = 0;
    for it in &items {
        let text = it.get("text").and_then(|t| t.str()).unwrap_or("");
        let want: Vec<i32> = it.get("ids").and_then(|i| i.uints()).unwrap_or_default().into_iter().map(|v| v as i32).collect();
        let got = tk.encode(text);
        if got == want {
            same += 1;
        } else {
            println!("DIFF {text:?}\n  engine    {got:?}\n  reference {want:?}");
        }
    }
    println!("tokens identical for {same} of {} texts", items.len());
    if same < items.len() {
        return Err("the tokenizer differs from the reference".into());
    }
    Ok(())
}

pub fn check_case(a: &Args, model_dir: &str, case: &str) -> R<()> {
    let text = a.get("--text").ok_or("--check-case needs --text (the text of the case)")?;
    let tok = tok::Tok::load(model_dir)?;
    let mut g = gpu::Gpu::new()?;
    let mut vae = vae::Vae::load(&mut g, model_dir)?;
    let ref_feat = a.get("--ref").map(|p| encode_wav(&mut g, &mut vae, p, false)).transpose()?;
    let prompt_feat = a.get("--prompt").map(|p| encode_wav(&mut g, &mut vae, p, true)).transpose()?;
    let p = prompt::build(&tok, text, a.get("--prompt-text").zip(prompt_feat.as_deref()), ref_feat.as_deref(), PATCH * DIM, 4096);
    let c = st::File::open(case)?;
    let mut differ = false;
    for (name, mine) in [("text", &p.tokens), ("text_mask", &p.text_mask), ("feat_mask", &p.audio_mask)] {
        let (r, _) = c.i32s(name)?;
        differ |= &r != mine;
        println!("{name:10} {}", if &r == mine { "identical".to_string() } else { format!("DIFFER ({} vs {} values)", mine.len(), r.len()) });
    }
    let (rf, _) = c.f32s("feat")?;
    let mine: Vec<f32> = p.feat.iter().map(|&v| rb(v)).collect();
    let (rel, mx) = cmp(&mine, &rf)?;
    differ |= rel.is_nan() || rel >= 1e-2; // the AudioVAE in fp32 against the reference's, both rounded to bf16
    let same = mine.iter().zip(&rf).filter(|(x, y)| x == y).count();
    println!("feat       rel {rel:.2e} max {mx:.3e}, {same} of {} bf16 values identical", rf.len());
    let want = c.meta.get("max_len").map(String::as_str).unwrap_or("?");
    differ |= p.max_len.to_string() != want;
    println!("max_len    engine {} reference {want}", p.max_len);
    if differ {
        return Err("the prompt differs from the reference".into());
    }
    Ok(())
}

/// Times the decoding loop; with the reference noise (--noise table) slot 0 is also checked against the trace.
pub fn run_case(a: &Args, model_dir: &str, case_paths: &str) -> R<()> {
    let b: usize = a.num("--batch", 1)?;
    // one seed per slot: the given list, repeated or cut to the batch size
    let seeds: Vec<u32> = a.seeds("7,11,23,42")?.into_iter().cycle().take(b).collect();
    let table = a.get("--noise") != Some("random");
    let graph = !a.flag("--no-graph");
    let reps: usize = a.num("--reps", 1)?;
    let maxlen: usize = a.num("--kv", 2048)?;
    let tmax: usize = a.num("--max-prompt", 1024)?;
    let cases: Vec<&str> = case_paths.split(',').collect();
    let timesteps = st::File::open(cases[0])?.meta.get("inference_timesteps").and_then(|v| v.parse().ok()).unwrap_or(10);

    let mut e = model::Engine::new(model_dir, b, maxlen, tmax, timesteps)?;
    e.g.cublas_only = a.flag("--cublas");
    if a.flag("--bw") {
        // device-to-device copy bandwidth available right now (the GPU may be shared)
        let bytes = 1usize << 30;
        let (src, dst) = (e.g.alloc(bytes)?, e.g.alloc(bytes)?);
        e.g.copy(dst, src, bytes)?;
        e.g.sync()?;
        let t = Instant::now();
        for _ in 0..10 {
            e.g.copy(dst, src, bytes)?;
        }
        e.g.sync()?;
        println!("copy bandwidth {:.0} GB/s (read + write)", 20.0 * bytes as f64 / t.elapsed().as_secs_f64() / 1e9);
    }
    if a.flag("--gemm-bench") {
        return gemm_bench(&mut e.g);
    }
    let dbg = e.g.alloc_bf16(2 * b * 256)?;
    for path in cases {
        let case = st::File::open(path)?;
        let meta = |k: &str| case.meta.get(k).cloned().unwrap_or_default();
        let (tokens, _) = case.i32s("text")?;
        let (tm, _) = case.i32s("text_mask")?;
        let (am, _) = case.i32s("feat_mask")?;
        let (feat, _) = case.f32s("feat")?;
        let noise = if table { Some(case.f32s("noise")?.0) } else { None };
        let t = tokens.len();
        e.cfg = meta("cfg_value").parse().unwrap_or(2.0);
        e.min_len = meta("min_len").parse().unwrap_or(2);
        let mut max_steps = meta("max_len").parse().unwrap_or(2000).min(e.maxlen.saturating_sub(t));
        if let Some(n) = &noise {
            max_steps = max_steps.min(n.len() / (PATCH * DIM)); // the reference saved noise a few steps past its stop
        }
        for rep in 0..reps {
            let t0 = Instant::now();
            e.prefill(&tokens, &tm, &am, &feat, noise.as_deref(), &seeds, max_steps)?;
            let t_pf = t0.elapsed().as_secs_f64();
            let verify = rep == 0 && table && case.has("patches");
            if verify {
                // the first step outside the graph, so that its first DiT output can be compared
                e.step(Some(dbg))?;
                let (base, res) = e.prefill_last(t)?;
                for (name, got, r) in [("base LM prefill (last pos)", base, "base_prefill"), ("residual LM prefill (last pos)", res, "residual_prefill")] {
                    let (rel, mx) = cmp(&got, &case.f32s(r)?.0)?;
                    println!("verify {name:32} rel {rel:.2e} max {mx:.3e}");
                }
                let d = e.g.download_bf16(dbg, 2 * b * 256)?;
                let r = to_time_major(&case.f32s("dit0")?.0);
                if r.len() != 512 {
                    return Err("dit0: expected one cond and one uncond patch".into());
                }
                let (rc, mc) = cmp(&d[..256], &r[..256])?;
                let (ru, mu) = cmp(&d[b * 256..b * 256 + 256], &r[256..])?;
                println!("verify {:32} rel {rc:.2e} max {mc:.3e} | uncond rel {ru:.2e} max {mu:.3e}", "first DiT output (cond)");
            }
            let t1 = Instant::now();
            let lens = e.decode(graph, verify as usize)?;
            let wall = t1.elapsed().as_secs_f64();
            let n = *lens.iter().max().unwrap_or(&1);
            let audio: f64 = lens.iter().map(|&l| l as f64 * 0.16).sum();
            println!(
                "rep{rep} B={b} prompt {t} prefill {t_pf:.3}s | {n} steps in {wall:.2}s = {:.1} ms/step | patches per slot {lens:?} | RTF {:.3} per take",
                1000.0 * wall / n as f64,
                (wall + t_pf) / (audio / b as f64),
            );
            if verify {
                let r = to_time_major(&case.f32s("patches")?.0);
                let got = &e.latents(&lens)?[0];
                let k = (r.len() / 256).min(got.len() / 256);
                for upto in [1, 4, 16, k] {
                    if upto <= k {
                        let (rel, mx) = cmp(&got[..upto * 256], &r[..upto * 256])?;
                        println!("verify patches[0..{upto:3}]{:18} rel {rel:.2e} max {mx:.3e}", "");
                    }
                }
                println!("verify stop: engine {} patches, reference {}", lens[0], r.len() / 256);
            }
        }
    }
    Ok(())
}
