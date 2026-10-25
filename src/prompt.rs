//! Prompt assembly as in VoxCPM2's `_generate_with_prompt_cache`: the token and latent sequences the model is
//! prefilled with, for the four modes (zero-shot, reference voice, continuation of a prompt, both).
//!
//!   reference part:    [ref_start] [0 x R] [ref_end]       latents  [0] [ref patches] [0]
//!   text part:         tokens(prompt_text + text) [audio_start]   latents 0
//!   continuation part: [0 x P]                               latents  [prompt patches]
//!
//! The text mask is 1 on token positions and the audio mask 1 on latent positions. Reference audio is padded
//! on the right to whole patches, prompt audio on the left (so the prompt ends exactly where generation starts).
use crate::tok::Tok;

pub const REF_START: i32 = 103;
pub const REF_END: i32 = 104;
pub const AUDIO_START: i32 = 101;

pub struct Prompt {
    pub tokens: Vec<i32>,
    pub text_mask: Vec<i32>,
    pub audio_mask: Vec<i32>,
    /// [positions][patch_size][latent_dim]
    pub feat: Vec<f32>,
    /// generation limit in patches: min(6 * target tokens + 10, max_len)
    pub max_len: usize,
    /// trailing prompt patches decoded in front of the output for continuity (VoxCPM2 uses up to 3)
    pub context: usize,
}

/// The target text as VoxCPM cleans it before tokenizing (`re.sub(r"\s+", " ", text)`): every run of whitespace
/// becomes one space; leading and trailing spaces stay.
pub fn clean_text(t: &str) -> String {
    let mut out = String::with_capacity(t.len());
    let mut ws = false;
    for c in t.chars() {
        if c.is_whitespace() {
            if !ws {
                out.push(' ');
            }
            ws = true;
        } else {
            out.push(c);
            ws = false;
        }
    }
    out
}

/// `ref_feat` / `prompt_feat`: latent patches [n][patch][dim] of the encoded audio; `patch` floats per position
/// is patch_size * latent_dim.
pub fn build(tok: &Tok, text: &str, prompt: Option<(&str, &[f32])>, ref_feat: Option<&[f32]>, patch: usize, max_len: usize) -> Prompt {
    let target = clean_text(text);
    let full = match prompt {
        Some((pt, _)) => format!("{pt}{target}"),
        None => target.clone(),
    };
    let mut text_tokens = tok.encode(&full);
    text_tokens.push(AUDIO_START);
    let target_len = tok.encode(&target).len();
    let (mut tokens, mut tm, mut am, mut feat) = (vec![], vec![], vec![], vec![]);
    if let Some(r) = ref_feat {
        let n = r.len() / patch;
        tokens.push(REF_START);
        tokens.extend(std::iter::repeat_n(0, n));
        tokens.push(REF_END);
        tm.push(1);
        tm.extend(std::iter::repeat_n(0, n));
        tm.push(1);
        am.push(0);
        am.extend(std::iter::repeat_n(1, n));
        am.push(0);
        feat.extend(std::iter::repeat_n(0f32, patch));
        feat.extend_from_slice(r);
        feat.extend(std::iter::repeat_n(0f32, patch));
    }
    let nt = text_tokens.len();
    tokens.extend(&text_tokens);
    tm.extend(std::iter::repeat_n(1, nt));
    am.extend(std::iter::repeat_n(0, nt));
    feat.extend(std::iter::repeat_n(0f32, nt * patch));
    let mut context = 0;
    if let Some((_, p)) = prompt {
        let n = p.len() / patch;
        tokens.extend(std::iter::repeat_n(0, n));
        tm.extend(std::iter::repeat_n(0, n));
        am.extend(std::iter::repeat_n(1, n));
        feat.extend_from_slice(p);
        context = n.min(3);
    }
    Prompt { tokens, text_mask: tm, audio_mask: am, feat, max_len: ((target_len as f64 * 6.0 + 10.0) as usize).min(max_len), context }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes() {
        let t = crate::tok::tests::tiny();
        let feat = |n: usize, v: f32| vec![v; n * 2];
        // zero-shot: text tokens + audio_start, no audio
        let p = build(&t, "ab", None, None, 2, 100);
        assert_eq!(p.tokens, vec![5, AUDIO_START]);
        assert_eq!((p.text_mask.clone(), p.audio_mask.clone(), p.feat.len(), p.context), (vec![1, 1], vec![0, 0], 4, 0));
        assert_eq!(p.max_len, 16); // 6 * 1 target token + 10
        // reference + continuation: [ref_start, 0 x 3, ref_end] [prompt text + text, audio_start] [0 x 2]
        let (r, q) = (feat(3, 1.0), feat(2, 2.0));
        let p = build(&t, "ab", Some(("ab ", &q)), Some(&r), 2, 100); // "ab ab" -> ▁ab ▁ab
        assert_eq!(p.tokens, vec![REF_START, 0, 0, 0, REF_END, 5, 5, AUDIO_START, 0, 0]);
        assert_eq!(p.text_mask, vec![1, 0, 0, 0, 1, 1, 1, 1, 0, 0]);
        assert_eq!(p.audio_mask, vec![0, 1, 1, 1, 0, 0, 0, 0, 1, 1]);
        assert_eq!(p.feat.len(), 10 * 2);
        assert_eq!(&p.feat[2..8], &r[..]);
        assert_eq!(&p.feat[16..], &q[..]);
        assert_eq!(p.context, 2);
        assert_eq!(build(&t, "ab", None, None, 2, 12).max_len, 12);
    }

    #[test]
    fn whitespace_is_collapsed() {
        assert_eq!(clean_text(" a\nb \t c  "), " a b c ");
        assert_eq!(clean_text("あ\u{3000}\u{3000}い"), "あ い");
    }
}
