# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "voxcpm @ git+https://github.com/OpenBMB/VoxCPM@cce58a5b59303c9bd63d12f23afe6a49a4a80c59",
#     "torch==2.10.0",
#     "torchaudio==2.10.0",
#     "transformers==5.3.0",
#     "librosa==0.11.0",
# ]
# ///
"""Reference token ids for `oumu --check-tokens`, from VoxCPM's own tokenizer path
(LlamaTokenizerFast + mask_multichar_chinese_tokens, as VoxCPM2Model uses it).

    uv run scripts/hf_tokens.py --model DIR texts.json --out tokens.json

texts.json is a JSON list of strings; the output is a list of {"text", "ids"}.
"""
import argparse
import json

from transformers import LlamaTokenizerFast
from voxcpm.model.utils import mask_multichar_chinese_tokens

ap = argparse.ArgumentParser()
ap.add_argument("texts")
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

tok = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(a.model))
texts = json.load(open(a.texts, encoding="utf-8"))
out = [{"text": t, "ids": [int(i) for i in tok(t)]} for t in texts]
json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
print(f"{len(out)} texts -> {a.out}")
