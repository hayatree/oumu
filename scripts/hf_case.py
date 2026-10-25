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
"""Reference trace of one VoxCPM2 generation from the voxcpm package, for `oumu --case` / `--check-case`.

Saves (safetensors): the exact prefill inputs of VoxCPM2Model._inference (tokens, masks, prompt latents) and the
decode limits; unless --prep-only, also the diffusion noise it consumed (drawn from numpy so the engine can replay
it) and reference values: hidden states after prefill, the first DiT output, every generated patch and the
waveform.

    uv run scripts/hf_case.py --model DIR --text "..." [--ref R.wav] [--prompt P.wav --prompt-text "..."] \
        --seed 7 --out case.safetensors
"""
import argparse
import types

import numpy as np
import torch
from safetensors.torch import save_file
from voxcpm import VoxCPM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--text", required=True)
ap.add_argument("--ref")
ap.add_argument("--prompt")
ap.add_argument("--prompt-text")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--cfg", type=float, default=2.0)
ap.add_argument("--timesteps", type=int, default=10)
ap.add_argument("--prep-only", action="store_true", help="save only the prefill inputs (no generation)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

model = VoxCPM.from_pretrained(a.model, load_denoiser=False, optimize=False)
tts = model.tts_model
cfm = tts.feat_decoder

rng = np.random.default_rng(a.seed)
noise = torch.from_numpy(rng.standard_normal((4096, 64, 4), dtype=np.float32))
trace = {"patches": [], "dit0": None}
step = [0]


def cfm_forward(self, mu, n_timesteps, patch_size, cond, temperature=1.0, cfg_value=1.0,
                sway_sampling_coef=1.0, use_cfg_zero_star=True):
    # same as UnifiedCFM.forward, but the noise comes from the replayable table
    z = noise[step[0]].to(mu.device, mu.dtype).unsqueeze(0) * temperature
    t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
    t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
    out = self.solve_euler(x=z, t_span=t_span, mu=mu, cond=cond, cfg_value=cfg_value,
                           use_cfg_zero_star=use_cfg_zero_star)
    trace["patches"].append(out.float().cpu())
    step[0] += 1
    return out


cfm.forward = types.MethodType(torch.inference_mode()(cfm_forward), cfm)

orig_est = cfm.estimator.forward


def est_forward(*args, **kw):
    out = orig_est(*args, **kw)
    if trace["dit0"] is None:
        trace["dit0"] = out.float().cpu()
    return out


cfm.estimator.forward = est_forward


def keep_last(module, key):
    orig = module.forward

    def fwd(*args, **kw):
        out = orig(*args, **kw)
        if key not in trace:
            trace[key] = out[0][:, -1].float().cpu()
        return out

    module.forward = fwd


keep_last(tts.base_lm, "base_prefill")
keep_last(tts.residual_lm, "residual_prefill")


class Prepared(Exception):
    pass


orig_inference = tts._inference


def inference(text, text_mask, feat, feat_mask, **kw):
    trace["inputs"] = dict(text=text.cpu().to(torch.int32), text_mask=text_mask.cpu().to(torch.int32),
                           feat=feat.float().cpu(), feat_mask=feat_mask.cpu().to(torch.int32))
    trace["kw"] = {k: v for k, v in kw.items() if isinstance(v, (int, float, bool))}
    if a.prep_only:
        raise Prepared
    yield from orig_inference(text, text_mask, feat, feat_mask, **kw)


tts._inference = inference

args = dict(text=a.text, reference_wav_path=a.ref, prompt_wav_path=a.prompt, prompt_text=a.prompt_text,
            cfg_value=a.cfg, inference_timesteps=a.timesteps, retry_badcase=False)
try:
    w = model.generate(**args, seed=a.seed)
except Prepared:
    w = None
meta = {k: str(v) for k, v in trace["kw"].items()}
tensors = dict(trace["inputs"])
if w is not None:
    n = len(trace["patches"])
    tensors.update({"noise": noise[:n + 8], "patches": torch.cat(trace["patches"]),  # [n, 64, 4]
                    "base_prefill": trace["base_prefill"], "residual_prefill": trace["residual_prefill"],
                    "dit0": trace["dit0"], "wav": torch.from_numpy(np.asarray(w, dtype=np.float32))})
    meta.update(steps=str(n), seed=str(a.seed))
save_file({k: v.contiguous() for k, v in tensors.items()}, a.out, metadata=meta)
print(f"saved {a.out}: prompt {tensors['text'].shape[1]} positions, meta {meta}")
