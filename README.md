<h1 align="center">oumu</h1>

<p align="center"><a href="https://github.com/OpenBMB/VoxCPM">VoxCPM2</a> text-to-speech in Rust and CUDA.</p>

<p align="center"><a href="#quick-start"><b>Quick start</b></a> · <a href="#usage">Usage</a> · <a href="#how-it-works">How it works</a> · <a href="#verification">Verification</a> · <a href="#limitations">Limitations</a></p>

<p align="center">
  <a href="https://github.com/hayatree/oumu/actions/workflows/ci.yml"><img src="https://github.com/hayatree/oumu/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue" alt="License: MIT OR Apache-2.0"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.ja.md">日本語</a></sub></p>

Give it text and, optionally, a recording of a voice; it writes 48 kHz WAV files.

- **Batched takes**: several takes of each text (one per seed) are decoded together, so you can generate candidates
  and keep the best.
- **Faithful to the reference**: the same prompt tokens as `voxcpm`, AudioVAE within 1e-5 of PyTorch, and similar
  scores on the samples evaluated (see [Verification](#verification)).
- **One binary, one dependency** ([cudarc](https://github.com/coreylowman/cudarc)): no Python at run time, and the
  checkpoint's pickle is read without executing it.

Status: 0.1. Read [Responsible use](#responsible-use) before cloning a voice.

## Quick start

You need Rust, a CUDA toolkit and an NVIDIA GPU (see [Build](#build)), plus [uv](https://docs.astral.sh/uv/) to download the model:

    git clone https://github.com/hayatree/oumu && cd oumu
    cargo build --release --locked

    uvx --from huggingface_hub hf download openbmb/VoxCPM2 \
        model.safetensors audiovae.pth config.json tokenizer.json tokenizer_config.json \
        --local-dir models/VoxCPM2

    ./target/release/oumu --model models/VoxCPM2 --text "Hello from oumu." --out hello.wav

Without a recording the model picks a voice; [Usage](#usage) shows how to clone one.

## Build

Requirements: Linux, an NVIDIA GPU of the Ampere generation or newer (sm_80+) with about 7 GB free, a CUDA toolkit
supported by cudarc 0.19 (`nvcc`; cuBLAS is loaded at run time), and Rust 1.88+.

    cargo build --release --locked      # the binary is target/release/oumu

| variable | default | |
|---|---|---|
| `NVCC`, `CUDA_HOME` | `nvcc` on `PATH`, then `/usr/local/cuda` | CUDA compiler for the kernels; cudarc still runs `nvcc --version` from `PATH` unless `CUDARC_CUDA_VERSION` is set |
| `CUDA_ARCH` | `compute_80` | PTX architecture; the driver JIT-compiles it for the GPU |
| `CUDA_SASS` | none | also embed machine code for these GPUs, e.g. `86,89,121`, to skip the JIT (useful in containers) |
| `CUDARC_CUDA_VERSION` | from `nvcc --version` | CUDA API version of the bindings, e.g. `12080` |

Developed and tested on an NVIDIA DGX Spark (GB10, sm_121, CUDA 13.0, aarch64). CI builds and runs the CPU tests on
x86-64 with CUDA 12.8, without a GPU. Reports from other GPUs are welcome.

## Usage

The model directory is the Hugging Face snapshot [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
(`model.safetensors`, `audiovae.pth`, `config.json`, `tokenizer.json`, `tokenizer_config.json`).

    # one take in a cloned voice
    oumu --model ~/models/VoxCPM2 --ref me.wav --prompt me.wav --prompt-text "What I say in me.wav." \
        --text "Hello from the engine." --out hello.wav

    # several texts, four takes each: takes/cand-<id>-s<seed>.wav
    oumu --model ~/models/VoxCPM2 --ref me.wav --prompt me.wav --prompt-text "..." \
        --texts texts.json --out-dir takes --prefix cand- --seeds 7,11,23,42 --trim

| voice options | effect |
|---|---|
| none | the model picks a voice |
| `--ref R.wav` | clones the voice of R |
| `--prompt P.wav --prompt-text T` | continues the recording P, whose exact transcript is T: the closest delivery |
| all three, same file twice | the closest clone (recommended by the VoxCPM2 authors) |

Recordings: WAV, 16/24/32-bit PCM or 32-bit float, any number of channels (mixed down), 1–768 kHz. Style hints such
as `(slightly faster, cheerful)` at the start of a text work as in `voxcpm`.

| option | |
|---|---|
| `--text TEXT --out FILE.wav` | one text; with several seeds the files are `FILE-s<seed>.wav` |
| `--texts FILE.json --out-dir DIR [--prefix STR]` | `{"id": "text", ...}` or `["text", ...]` → `DIR/<prefix><id>[-s<seed>].wav` |
| `--seeds LIST` | comma-separated seeds, one take each (default `1`, at most 256) |
| `--cfg X` | classifier-free guidance (default 2.0) |
| `--timesteps N` | diffusion steps per patch, 2–100 (default 10) |
| `--min-len N` | a take has at least N + 2 patches of 0.16 s (default 2, as in `voxcpm`) |
| `--max-len N` | at most N patches per take (default 4096); also at most 6 per text token + 10, as in `voxcpm`, and no more than the KV cache has room for |
| `--trim` | cut leading and trailing silence (below 2% of the peak), keeping 50 ms |
| `--float` | 32-bit float WAV instead of 16-bit PCM |
| `--kv N` | KV cache length per take: prompt plus generated patches (default 4096, rounded up to a multiple of 64) |
| `--max-prompt N` | longest prompt in positions (default 1024; a 13 s recording used twice plus a sentence is about 330) |
| `--quiet` | no progress or timing on standard error |

## How it works

| step | files | |
|---|---|---|
| text | `tok.rs`, `prompt.rs` | the tokenizer as `transformers` loads it, VoxCPM's CJK split, prompt assembly as `_generate_with_prompt_cache` |
| voice | `audio.rs`, `vae.rs` | WAV → 16 kHz → AudioVAE encoder in fp32 (cuBLAS GEMMs, fused depthwise convolution + Snake kernel) |
| decoding | `model.rs`, `kernels/voxcpm.cu` | prefill once, copy the state into each take's slot, then run one CUDA graph per step; bf16 storage, fp32 arithmetic, rounding where PyTorch rounds |
| waveform | `vae.rs`, `kernels/vae.cu` | AudioVAE decoder in fp32; the last prompt patches are decoded in front for continuity and cut off, as in the reference |
| loading | `st.rs`, `pth.rs` | safetensors memory-mapped and streamed to the GPU; `audiovae.pth` read by a pickle interpreter that knows only tensors and dicts |

`tts.rs` ties these together (`Tts::load`, `voice`, `speak`), `main.rs` is the command line, and `dev.rs` holds the
checks below.

A decoding step does little arithmetic for each weight it reads, so the loop is built around reading less memory and
launching fewer kernels:

- A whole step (the diffusion passes with guidance, the local encoder, the stop head and both language models) is
  captured once as a CUDA graph and replayed with one launch per step.
- Attention reads only the filled part of the KV cache.
- Small matrix products read each weight once for all takes in the batch, not once per take.

## Verification

Each stage was compared with the Python implementation (`voxcpm`, PyTorch 2.10), using a 13 s recording as both
reference and prompt and Japanese narration texts:

| stage | compared with | result |
|---|---|---|
| tokenizer | `transformers`, 56 texts (CJK, Korean, accents, emoji, digits, whitespace, special tokens) | 55 identical; the other contains the literal text `<\|im_end\|>`, which `transformers` maps inconsistently with its own config |
| prompt | `voxcpm` | identical tokens, masks and length limit |
| resampling to 16 kHz | librosa (soxr HQ) | relative L2 error 9e-5 |
| AudioVAE encoder / decoder | PyTorch, strict fp32 | relative L2 error 3.5e-6 / 1.3e-6 (PyTorch's own TF32 default is 4e-3 / 7e-4 away) |
| decoding, with the reference noise replayed | PyTorch bf16 trace | relative L2 error 1e-2 after prefill and the first diffusion pass (bf16 rounding); stops at the same step |

In later patches bf16 rounding differences can compound, as between any two bf16 implementations, so the output was
compared instead: 32 takes (8 texts × 4 seeds) from each implementation, scored the same way.

| | `voxcpm` | oumu |
|---|---|---|
| character error rate (Whisper large-v3) | 0.084 | 0.086 |
| pitch distance from the voice | 0.9 semitones | 1.3 semitones |
| pace | 7.34 chars/s | 7.42 chars/s |
| spectral centroid | 911 Hz | 911 Hz |

With the same inputs, options, GPU and build, the same seed list gave bit-identical audio on every run. Seeds do not reproduce PyTorch takes (the noise generator
is the engine's own), and a seed can come out slightly different in a batch of a different size.

To repeat the checks, run the scripts with [uv](https://docs.astral.sh/uv/), which installs the versions used above
(pinned in each script). Each script writes a reference that the engine compares itself with;
`--check-tokens` and `--check-case` exit with an error when something that should be identical is not.

    uv run scripts/hf_tokens.py --model DIR texts.json --out tokens.json   # a list of strings
    oumu --model DIR --check-tokens tokens.json

    uv run scripts/hf_vae.py --model DIR --wav voice.wav --out vae.safetensors
    oumu --model DIR --check-vae vae.safetensors --wav voice.wav

    uv run scripts/hf_case.py --model DIR --ref voice.wav --prompt voice.wav \
        --prompt-text "..." --text "..." --seed 7 --out case.safetensors
    oumu --model DIR --check-case case.safetensors --ref voice.wav --prompt voice.wav --prompt-text "..." --text "..."
    oumu --model DIR --case case.safetensors     # decoding loop against the trace

For development, `--case FILE` also takes `--batch N --noise random` to time prefill and decoding of N takes (no WAV
is written), and `--gemm-bench` / `--bw` to time the matrix kernels and memory bandwidth on your GPU.

## Limitations

- VoxCPM2 only; other checkpoints are refused. No LoRA, streaming, prompt denoiser, text normalisation or automatic
  retry of overlong takes: generate several seeds and pick.
- WAV input only (`ffmpeg -i in.mp3 out.wav`).
- A command-line tool, not a library: GPU buffers are freed when the process exits.
- Linux and NVIDIA GPUs only.

## Responsible use

Clone only your own voice or a voice whose owner agreed to it, for the use they agreed to. Do not use synthetic
speech to impersonate people, deceive listeners or get around voice-based security, and say that speech is synthetic
where it could be taken for a real recording. Laws on likeness, consent and disclosure differ between countries. The
engine adds no watermark.

## Security

The engine makes no network access: it reads the model directory, the recordings and the texts, and writes WAV
files. Model files, WAV and JSON go through validating readers (offsets, tensor shapes, token ids and nesting depth
are checked), and the pickle in `audiovae.pth` is interpreted with a whitelist and bounded in what it can build, so
no code from the checkpoint is run. A hostile checkpoint can still make the output wrong: use models from sources you
trust.

## License

[Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option. Model weights are not included (VoxCPM2:
Apache-2.0, by OpenBMB).

oumu is an independent implementation of OpenBMB's VoxCPM2 ([VoxCPM](https://github.com/OpenBMB/VoxCPM),
Apache-2.0) and follows it step by step. It also builds on Hugging Face `transformers` (tokenizer behaviour) and
cudarc; [NOTICE](NOTICE) lists the parts derived from VoxCPM and `transformers`.
