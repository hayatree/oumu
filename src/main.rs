//! oumu: VoxCPM2 text-to-speech on a CUDA GPU, several seeds of a text decoded at once.
//! See README.md for the options; `--help` prints them.
mod audio;
mod dev;
mod gpu;
mod json;
mod model;
mod prompt;
mod pth;
mod st;
mod tok;
mod tts;
mod vae;

use gpu::R;
use json::{Json, parse_json};
use std::time::Instant;

const USAGE: &str = "\
usage: oumu --model DIR [voice] --text TEXT --out FILE.wav [options]
       oumu --model DIR [voice] --texts FILE.json --out-dir DIR [options]

Speaks text with VoxCPM2 (Hugging Face snapshot openbmb/VoxCPM2: model.safetensors, audiovae.pth, config.json,
tokenizer.json, tokenizer_config.json). Every seed is decoded in its own slot of one batch, so a few takes of a
text cost about as much as one. Output: 48 kHz mono WAV, one file per text and seed.

voice (optional; without either, the model picks a voice):
  --ref FILE.wav        reference recording: the voice to clone, kept apart from the text
  --prompt FILE.wav     prompt recording to continue from, with
  --prompt-text TEXT    its exact transcript; usually the same file as --ref
text:
  --text TEXT           one text, written to --out (with several seeds: FILE-s<seed>.wav)
  --texts FILE.json     {\"id\": \"text\", ...} or [\"text\", ...]; written to --out-dir as <prefix><id>-s<seed>.wav
  --prefix STR          file name prefix for --texts
options:
  --seeds LIST          comma-separated seeds, one take each (default 1)
  --cfg X               classifier-free guidance (default 2.0)
  --timesteps N         diffusion steps per patch (default 10)
  --min-len N           patches before the stop head may end a take (default 2)
  --max-len N           upper limit in patches (0.16 s each); also capped at 6 per text token + 10 (default 4096)
  --trim                cut leading and trailing silence (below 2% of the peak), keeping 50 ms
  --float               write 32-bit float WAV instead of 16-bit PCM
  --kv N                KV cache length per slot: prompt plus generated patches (default 4096)
  --max-prompt N        longest prompt in positions (default 1024)
  --quiet               no progress or timing on standard error
development:
  --check-tokens FILE   tokenizer against scripts/hf_tokens.py output
  --check-vae FILE      AudioVAE against scripts/hf_vae.py output (--wav FILE.wav: also the resampler)
  --check-case FILE     prompt assembly against scripts/hf_case.py output (with the voice options and --text)
  --case FILE[,FILE]    decode cases from scripts/hf_case.py; with --noise table (default) slot 0 is checked
                        against the PyTorch trace; also --batch N, --noise random, --reps N, --cublas,
                        --gemm-bench, --bw
  --no-graph            launch every decoding step's kernels instead of replaying a CUDA graph
";

pub struct Args {
    opts: Vec<(String, String)>,
    flags: Vec<String>,
}

impl Args {
    fn parse() -> R<Self> {
        const WITH_VALUE: &[&str] = &[
            "--model",
            "--ref",
            "--prompt",
            "--prompt-text",
            "--text",
            "--out",
            "--texts",
            "--out-dir",
            "--prefix",
            "--seeds",
            "--cfg",
            "--timesteps",
            "--min-len",
            "--max-len",
            "--kv",
            "--max-prompt",
            "--check-tokens",
            "--check-vae",
            "--wav",
            "--check-case",
            "--case",
            "--batch",
            "--noise",
            "--reps",
        ];
        const FLAGS: &[&str] = &["--trim", "--float", "--quiet", "--no-graph", "--cublas", "--gemm-bench", "--bw", "--help", "-h"];
        let mut a = Args { opts: vec![], flags: vec![] };
        let mut it = std::env::args().skip(1);
        while let Some(s) = it.next() {
            if WITH_VALUE.contains(&s.as_str()) {
                let v = it.next().ok_or(format!("{s} needs a value"))?;
                a.opts.push((s, v));
            } else if FLAGS.contains(&s.as_str()) {
                a.flags.push(s);
            } else {
                return Err(format!("unknown argument {s} (see --help)").into());
            }
        }
        Ok(a)
    }
    pub fn get(&self, k: &str) -> Option<&str> {
        self.opts.iter().rev().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
    }
    pub fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k)
    }
    pub fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> R<T> {
        match self.get(k) {
            Some(v) => v.parse().map_err(|_| format!("{k}: bad value {v:?}").into()),
            None => Ok(default),
        }
    }
    pub fn seeds(&self, default: &str) -> R<Vec<u32>> {
        let s: Vec<u32> = self
            .get("--seeds")
            .unwrap_or(default)
            .split(',')
            .map(|s| s.trim().parse())
            .collect::<Result<_, _>>()
            .map_err(|_| "--seeds: comma-separated integers")?;
        if s.is_empty() {
            return Err("--seeds: at least one seed".into());
        }
        Ok(s)
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> R<()> {
    let a = Args::parse()?;
    if a.flag("--help") || a.flag("-h") || std::env::args().len() == 1 {
        print!("{USAGE}");
        return Ok(());
    }
    let model_dir = a.get("--model").ok_or("--model is required (see --help)")?;
    if let Some(p) = a.get("--check-tokens") {
        return dev::check_tokens(model_dir, p);
    }
    if let Some(p) = a.get("--check-vae") {
        return dev::check_vae(&a, model_dir, p);
    }
    if let Some(p) = a.get("--check-case") {
        return dev::check_case(&a, model_dir, p);
    }
    if let Some(p) = a.get("--case") {
        return dev::run_case(&a, model_dir, p);
    }
    speak(&a, model_dir)
}

/// The texts to speak: (id, text, output path without the seed suffix).
fn jobs(a: &Args) -> R<Vec<(String, String, String)>> {
    if let Some(t) = a.get("--text") {
        let out = a.get("--out").ok_or("--text needs --out FILE.wav")?;
        return Ok(vec![(String::new(), t.to_string(), out.to_string())]);
    }
    let path = a.get("--texts").ok_or("give --text TEXT --out FILE.wav, or --texts FILE.json --out-dir DIR (see --help)")?;
    let dir = a.get("--out-dir").ok_or("--texts needs --out-dir DIR")?;
    let prefix = a.get("--prefix").unwrap_or_default();
    if prefix.contains(['/', '\\']) || prefix.starts_with('.') {
        return Err("--prefix cannot contain a path separator or start with a dot".into());
    }
    let items: Vec<(String, Json)> = match parse_json(&std::fs::read(path).map_err(|e| format!("{path}: {e}"))?)? {
        Json::Obj(v) => v,
        Json::Arr(v) => v.into_iter().enumerate().map(|(i, t)| (i.to_string(), t)).collect(),
        _ => return Err(format!("{path}: expected an object {{id: text}} or a list of texts").into()),
    };
    let mut out = vec![];
    for (id, t) in items {
        let t = t.str().ok_or(format!("{path}: {id} is not a string"))?;
        if id.is_empty() || id.contains(['/', '\\']) || id.starts_with('.') {
            return Err(format!("{path}: id {id:?} cannot be used in a file name").into());
        }
        out.push((id.clone(), t.to_string(), format!("{dir}/{prefix}{id}.wav")));
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
    Ok(out)
}

fn speak(a: &Args, model_dir: &str) -> R<()> {
    let jobs = jobs(a)?;
    if let Some((id, _, _)) = jobs.iter().find(|(_, t, _)| t.trim().is_empty()) {
        return Err(format!("text {id:?} is empty").into());
    }
    let seeds = a.seeds("1")?;
    let prompt = match (a.get("--prompt"), a.get("--prompt-text")) {
        (Some(p), Some(t)) => Some((p, t)),
        (None, None) => None,
        _ => return Err("--prompt and --prompt-text go together".into()),
    };
    let opts = tts::Options {
        cfg: a.num("--cfg", 2.0)?,
        timesteps: a.num("--timesteps", 10)?,
        min_len: a.num("--min-len", 2)?,
        max_len: a.num("--max-len", 4096)?,
        kv: a.num("--kv", 4096)?,
        max_prompt: a.num("--max-prompt", 1024)?,
        graph: !a.flag("--no-graph"),
    };
    let quiet = a.flag("--quiet");
    let t0 = Instant::now();
    let mut tts = tts::Tts::load(model_dir, seeds.len(), opts)?;
    let voice = tts.voice(a.get("--ref"), prompt)?;
    let sr = tts.output_rate();
    if !quiet {
        eprintln!("ready in {:.1}s (weights, AudioVAE, voice recordings)", t0.elapsed().as_secs_f64());
    }
    let mut total = 0.0;
    for (id, text, out) in &jobs {
        let takes = tts.speak(text, &voice, &seeds)?;
        for (w, seed) in takes.audio.iter().zip(&seeds) {
            let w = if a.flag("--trim") { audio::trim(w, sr, 0.05) } else { w.clone() };
            let path = if seeds.len() == 1 { out.clone() } else { format!("{}-s{seed}.wav", out.trim_end_matches(".wav")) };
            audio::write_wav(&path, &w, sr, a.flag("--float"))?;
            total += w.len() as f64 / sr as f64;
        }
        if !quiet {
            let [pf, dec, vae] = takes.secs;
            let name = if id.is_empty() { out } else { id };
            eprintln!("{name}: prompt {}, patches {:?}; prefill {pf:.2}s, decode {dec:.2}s, AudioVAE {vae:.2}s", takes.prompt_len, takes.patches);
        }
    }
    if !quiet {
        eprintln!("{} text(s) x {} seed(s): {total:.1}s of audio in {:.1}s including loading", jobs.len(), seeds.len(), t0.elapsed().as_secs_f64());
    }
    Ok(())
}
