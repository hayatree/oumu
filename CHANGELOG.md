# Changelog

## 0.1.0 (unreleased)

- First version: VoxCPM2 text-to-speech from text to 48 kHz WAV in one binary (tokenizer, AudioVAE encoder and
  decoder, decoding loop), with zero-shot, reference, continuation and combined voice modes, several seeds decoded
  as one batch, and checks against the reference implementation (`scripts/`, `--check-*`, `--case`).
