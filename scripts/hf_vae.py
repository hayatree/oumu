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
"""Reference AudioVAE outputs from the voxcpm package, for `oumu --check-vae`.

Encodes a WAV as VoxCPM2 does for a reference recording (librosa at 16 kHz, padded on the right to whole patches) and decodes the resulting
latents back to 48 kHz audio, once in strict fp32 and once with PyTorch's defaults (cuDNN may use TF32), and saves
everything to one safetensors file.

    uv run scripts/hf_vae.py --model DIR --wav prompt.wav --out vae_ref.safetensors
"""
import argparse
import json
import math

import librosa
import torch
from safetensors.torch import save_file
from voxcpm.modules.audiovae.audio_vae_v2 import AudioVAE, AudioVAEConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--wav", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--patch", type=int, default=4)
a = ap.parse_args()

cfg = json.load(open(f"{a.model}/config.json"))["audio_vae_config"]
vae = AudioVAE(AudioVAEConfig(**cfg))
sd = torch.load(f"{a.model}/audiovae.pth", map_location="cpu", weights_only=True)["state_dict"]
vae.load_state_dict(sd, strict=True)
vae = vae.cuda().eval()

y, _ = librosa.load(a.wav, sr=vae.sample_rate, mono=True)
y = torch.from_numpy(y)
n = a.patch * vae.chunk_size
y = torch.nn.functional.pad(y, (0, math.ceil(len(y) / n) * n - len(y)))  # right padding, as for a reference wav

out = {"wav16": y.contiguous()}
for name, tf32 in [("fp32", False), ("default", True)]:
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        mu = vae.encode(y[None].cuda(), vae.sample_rate)  # [1, 64, T]
        z = out.get("mu_fp32", mu[0].T.cpu()).T[None].cuda().float()  # decode the strict-fp32 latents both times
        w = vae.decode(z)[0, 0]
    out[f"mu_{name}"] = mu[0].T.contiguous().cpu()  # [T, 64]
    out[f"wav48_{name}"] = w.contiguous().cpu()
save_file(out, a.out, metadata={"wav": a.wav})
print(f"saved {a.out}: {len(y)} samples -> {out['mu_fp32'].shape[0]} frames -> {out['wav48_fp32'].shape[0]} samples")
