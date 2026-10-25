//! Text to speech: voice recordings through the AudioVAE encoder, prompt assembly, decoding of every seed at
//! once, and the AudioVAE decoder back to audio.
use crate::gpu::{Gpu, R, rb};
use crate::model::Engine;
use crate::tok::Tok;
use crate::vae::Vae;
use crate::{audio, prompt};
use std::time::Instant;

/// Latent frames per patch and latent channels of VoxCPM2.
pub const PATCH: usize = 4;
pub const DIM: usize = 64;

pub struct Options {
    pub cfg: f32,
    pub timesteps: usize,
    pub min_len: i32,
    pub max_len: usize,
    pub kv: usize,
    pub max_prompt: usize,
    pub graph: bool,
}

/// Encoded voice recordings: reference patches, and the prompt transcript with its patches.
#[derive(Default)]
pub struct Voice {
    reference: Option<Vec<f32>>,
    prompt: Option<(String, Vec<f32>)>,
}

/// The takes of one text, one waveform per seed, and where the time went.
pub struct Takes {
    pub audio: Vec<Vec<f32>>,
    pub prompt_len: usize,
    pub patches: Vec<usize>,
    pub secs: [f64; 3], // prefill, decoding, AudioVAE
}

/// Reads a WAV, resamples it to the VAE's rate, pads it to whole patches (on the left for a continuation prompt,
/// on the right for a reference) and encodes it: patches [n][PATCH][DIM].
pub fn encode_wav(g: &mut Gpu, vae: &mut Vae, path: &str, left: bool) -> R<Vec<f32>> {
    let (y, sr) = audio::read_wav(path)?;
    let mut y = audio::resample(&y, sr, vae.cfg.sample_rate)?;
    let n = PATCH * vae.cfg.hop();
    let pad = (n - y.len() % n) % n;
    if left {
        y.splice(0..0, std::iter::repeat_n(0.0, pad));
    } else {
        y.extend(std::iter::repeat_n(0.0, pad));
    }
    vae.encode(g, &y)
}

pub struct Tts {
    tok: Tok,
    engine: Engine,
    vae: Vae,
    opts: Options,
}

impl Tts {
    /// Loads the model for up to `slots` takes per text.
    pub fn load(model_dir: &str, slots: usize, opts: Options) -> R<Self> {
        if !opts.cfg.is_finite() {
            return Err("--cfg must be a finite number".into());
        }
        let tok = Tok::load(model_dir)?;
        let mut engine = Engine::new(model_dir, slots, opts.kv, opts.max_prompt, opts.timesteps)?;
        let vae = Vae::load(&mut engine.g, model_dir)?;
        engine.cfg = opts.cfg;
        engine.min_len = opts.min_len;
        Ok(Tts { tok, engine, vae, opts })
    }

    pub fn output_rate(&self) -> usize {
        self.vae.cfg.out_sample_rate
    }

    /// `reference`: a recording of the voice; `prompt`: a recording and its exact transcript to continue from.
    pub fn voice(&mut self, reference: Option<&str>, prompt: Option<(&str, &str)>) -> R<Voice> {
        let g = &mut self.engine.g;
        Ok(Voice {
            reference: reference.map(|p| encode_wav(g, &mut self.vae, p, false)).transpose()?,
            prompt: prompt.map(|(p, text)| encode_wav(g, &mut self.vae, p, true).map(|f| (text.to_string(), f))).transpose()?,
        })
    }

    /// Speaks `text` once per seed (at most as many seeds as slots).
    pub fn speak(&mut self, text: &str, voice: &Voice, seeds: &[u32]) -> R<Takes> {
        if seeds.is_empty() || seeds.len() > self.engine.b {
            return Err(format!("between 1 and {} seeds per text", self.engine.b).into());
        }
        let t0 = Instant::now();
        let prompt = voice.prompt.as_ref().map(|(t, f)| (t.as_str(), f.as_slice()));
        let p = prompt::build(&self.tok, text, prompt, voice.reference.as_deref(), PATCH * DIM, self.opts.max_len);
        let t = p.tokens.len();
        let kv = self.engine.maxlen;
        if t >= kv {
            return Err(format!("the prompt has {t} positions; the KV cache holds {kv}").into());
        }
        let max_steps = p.max_len.min(kv - t);
        self.engine.prefill(&p.tokens, &p.text_mask, &p.audio_mask, &p.feat, None, seeds, max_steps)?;
        let t1 = Instant::now();
        let mut lens = self.engine.decode(self.opts.graph, 0)?;
        lens.truncate(seeds.len());
        let t2 = Instant::now();
        // as in the reference, the last prompt patches are decoded in front for continuity, then cut off
        let ctx: Vec<f32> = p.feat[p.feat.len() - p.context * PATCH * DIM..].iter().map(|&v| rb(v)).collect();
        let skip = p.context * PATCH * self.vae.cfg.decode_hop();
        let mut audio = vec![];
        for lat in self.engine.latents(&lens)? {
            let z = [ctx.as_slice(), &lat].concat();
            let w = self.vae.decode(&mut self.engine.g, &z)?;
            audio.push(w[skip.min(w.len())..].to_vec());
        }
        let secs = [(t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), t2.elapsed().as_secs_f64()];
        Ok(Takes { audio, prompt_len: t, patches: lens, secs })
    }
}
