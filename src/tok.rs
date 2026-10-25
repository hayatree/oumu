//! VoxCPM2's tokenizer as `transformers` loads it (byte-fallback BPE, Metaspace, added tokens first), followed by
//! VoxCPM's `mask_multichar_chinese_tokens`, which splits tokens of several CJK ideographs into single characters.
//! No BOS/EOS is added, as in the reference `tokenize()`.
use crate::gpu::R;
use crate::json::{Json, parse_json};
use std::collections::HashMap;

pub struct Tok {
    vocab: HashMap<String, u32>,
    names: Vec<String>,
    merges: HashMap<(u32, u32), (u32, u32)>, // pair -> (rank, merged id)
    added: Vec<(String, u32)>,               // longest first
    unk: u32,
}

/// Token ids are bounded so that a malformed file cannot make the id -> name table huge.
const MAX_ID: u32 = 1 << 24;

fn token_id(j: &Json) -> Option<u32> {
    j.uint().filter(|&n| n < MAX_ID as usize).map(|n| n as u32)
}

fn cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

impl Tok {
    pub fn load(model_dir: &str) -> R<Self> {
        let path = format!("{model_dir}/tokenizer.json");
        let j = parse_json(&std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?)?;
        let model = j.get("model").ok_or("tokenizer.json: no model")?;
        if model.get("type").and_then(|t| t.str()) != Some("BPE") {
            return Err("tokenizer.json: only BPE models are supported".into());
        }
        let Some(Json::Obj(v)) = model.get("vocab") else { return Err("tokenizer.json: no vocab".into()) };
        let mut vocab = HashMap::with_capacity(v.len());
        for (s, id) in v {
            let id = token_id(id).ok_or_else(|| format!("tokenizer.json: bad id for {s:?}"))?;
            vocab.insert(s.clone(), id);
        }
        let mut added = vec![];
        if let Some(Json::Arr(a)) = j.get("added_tokens") {
            for t in a {
                let s = t.get("content").and_then(|c| c.str()).filter(|s| !s.is_empty());
                let (Some(s), Some(id)) = (s, t.get("id").and_then(token_id)) else { return Err("tokenizer.json: bad added token".into()) };
                if *vocab.entry(s.to_string()).or_insert(id) != id {
                    return Err(format!("tokenizer.json: added token {s:?} has a different id in the vocabulary").into());
                }
                added.push((s.to_string(), id));
            }
        }
        // transformers takes the added tokens of tokenizer_config.json over those of tokenizer.json
        if let Ok(b) = std::fs::read(format!("{model_dir}/tokenizer_config.json"))
            && let Some(Json::Obj(dec)) = parse_json(&b)?.get("added_tokens_decoder")
        {
            for (id, t) in dec {
                let id = id.parse::<u32>().ok().filter(|&n| n < MAX_ID);
                let s = t.get("content").and_then(|c| c.str()).filter(|s| !s.is_empty());
                let (Some(id), Some(s)) = (id, s) else { return Err("tokenizer_config.json: bad added token".into()) };
                added.retain(|(a, _)| a != s);
                added.push((s.to_string(), id));
                vocab.insert(s.to_string(), id);
            }
        }
        added.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
        let n = vocab.values().copied().max().unwrap_or(0) as usize + 1;
        let mut names = vec![String::new(); n];
        for (s, &id) in &vocab {
            names[id as usize] = s.clone();
        }
        let mut merges = HashMap::new();
        if let Some(Json::Arr(m)) = model.get("merges") {
            for (rank, e) in m.iter().enumerate() {
                let (a, b) = match e {
                    Json::Str(s) => match s.split_once(' ') {
                        Some((a, b)) => (a.to_string(), b.to_string()),
                        None => continue,
                    },
                    Json::Arr(p) if p.len() == 2 => match (p[0].str(), p[1].str()) {
                        (Some(a), Some(b)) => (a.to_string(), b.to_string()),
                        _ => continue,
                    },
                    _ => continue,
                };
                if let (Some(&ia), Some(&ib), Some(&im)) = (vocab.get(&a), vocab.get(&b), vocab.get(&format!("{a}{b}"))) {
                    merges.entry((ia, ib)).or_insert((rank as u32, im));
                }
            }
        }
        let unk_name = model.get("unk_token").and_then(|u| u.str()).unwrap_or("<unk>");
        let unk = *vocab.get(unk_name).unwrap_or(&0);
        Ok(Tok { vocab, names, merges, added, unk })
    }
}

/// Metaspace: spaces become "▁" and a "▁" is prepended unless there already is one.
fn metaspace(seg: &str) -> String {
    let s = seg.replace(' ', "\u{2581}");
    if s.starts_with('\u{2581}') { s } else { format!("\u{2581}{s}") }
}

impl Tok {
    /// BPE over one word; appends token ids.
    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        let mut sym: Vec<u32> = vec![];
        for c in word.chars() {
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            if let Some(&id) = self.vocab.get(&*s) {
                sym.push(id);
                continue;
            }
            // byte fallback: <0xXX> for every UTF-8 byte, unknown if any of them is missing
            let bytes: Option<Vec<u32>> = s.bytes().map(|b| self.vocab.get(&format!("<0x{b:02X}>")).copied()).collect();
            match bytes {
                Some(b) => sym.extend(b),
                None => {
                    if sym.last() != Some(&self.unk) {
                        sym.push(self.unk); // fuse_unk
                    }
                }
            }
        }
        // repeatedly merge the lowest-ranked adjacent pair (the leftmost on ties)
        loop {
            let best = sym.windows(2).enumerate().filter_map(|(i, w)| self.merges.get(&(w[0], w[1])).map(|&(r, id)| (r, i, id))).min();
            let Some((_, i, id)) = best else { break };
            sym[i] = id;
            sym.remove(i + 1);
        }
        out.extend(sym);
    }

    /// Ids before the multi-character CJK split (the reference tokenizer's `tokenize`).
    fn base_ids(&self, text: &str) -> Vec<u32> {
        let mut out = vec![];
        let mut rest = text;
        while !rest.is_empty() {
            // the earliest added token, the longest one at that position
            let hit = self.added.iter().filter_map(|(s, id)| rest.find(s.as_str()).map(|p| (p, std::cmp::Reverse(s.len()), *id))).min();
            let (seg, after) = match hit {
                Some((p, std::cmp::Reverse(len), id)) => {
                    let seg = &rest[..p];
                    if !seg.is_empty() {
                        self.bpe(&metaspace(seg), &mut out);
                    }
                    out.push(id);
                    (None, &rest[p + len..])
                }
                None => (Some(rest), ""),
            };
            if let Some(seg) = seg {
                self.bpe(&metaspace(seg), &mut out);
            }
            rest = after;
        }
        out
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        let mut out = vec![];
        for id in self.base_ids(text) {
            let clean: String = self.names[id as usize].chars().filter(|&c| c != '\u{2581}').collect();
            if clean.chars().count() >= 2 && clean.chars().all(cjk) && self.vocab.contains_key(&clean) {
                out.extend(clean.chars().map(|c| *self.vocab.get(&c.to_string()).unwrap_or(&self.unk) as i32));
            } else {
                out.push(id as i32);
            }
        }
        out
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A tiny tokenizer: a few merges, byte fallback for "z", one added token, a two-ideograph token.
    pub fn tiny() -> Tok {
        let dir = std::env::temp_dir().join(format!("ve-tok-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tj = r#"{"model": {"type": "BPE", "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "▁": 1, "a": 2, "b": 3, "▁a": 4, "▁ab": 5, "中": 6, "国": 7, "中国": 8, "<0x7A>": 9},
            "merges": ["▁ a", "▁a b", "中 国"]},
            "added_tokens": [{"id": 10, "content": "<s>"}]}"#;
        std::fs::write(dir.join("tokenizer.json"), tj).unwrap();
        let t = Tok::load(dir.to_str().unwrap()).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        t
    }

    #[test]
    fn bpe_metaspace_fallback_added_and_cjk() {
        let t = tiny();
        assert_eq!(t.encode("ab"), vec![5]); // "▁ab": ▁ a b -> ▁a b -> ▁ab
        assert_eq!(t.encode("a b"), vec![4, 1, 3]); // the space becomes ▁, which does not merge with b here
        assert_eq!(t.encode("z"), vec![1, 9]); // byte fallback
        assert_eq!(t.encode("q"), vec![1, 0]); // no byte token either: unknown
        assert_eq!(t.encode("a<s>ab"), vec![4, 10, 5]); // added token split out, each side gets its own ▁
        assert_eq!(t.encode("中国"), vec![1, 6, 7]); // two-ideograph token split into characters
        assert_eq!(t.encode(""), Vec::<i32>::new());
    }

    #[test]
    fn conflicting_added_token_is_an_error() {
        let dir = std::env::temp_dir().join(format!("ve-tok-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tj = r#"{"model": {"type": "BPE", "vocab": {"<s>": 0}, "merges": []}, "added_tokens": [{"id": 10, "content": "<s>"}]}"#;
        std::fs::write(dir.join("tokenizer.json"), tj).unwrap();
        assert!(Tok::load(dir.to_str().unwrap()).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
