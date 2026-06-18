# Privacy Policy — audio-analysis-mcp

audio-analysis-mcp is a local MCP server that runs on your own machine. It
imports, separates, analyzes, transcribes, and compares audio so an AI agent
can help recreate a sound. The audio you give it, and every file it produces,
stay on your machine — they are never uploaded anywhere.

It performs **no analytics, telemetry, tracking, or usage reporting**, and it
does not collect, store, or share personal data. (Verified by source scan: the
shipped server makes no outbound calls of its own.)

The only network access in normal use is **one-time downloads of open ML model
weights**, described below. Once a model is cached locally, it is never fetched
again.

## First-run model downloads

- **Demucs stem-separation model — `stem_separate`.** The first time you run
  `stem_separate`, the Demucs library downloads its pretrained model weights
  from Meta's public file host (`https://dl.fbaipublicfiles.com/demucs/`) and
  caches them in your local Torch hub cache
  (`~/.cache/torch/hub/checkpoints/`). Subsequent runs use the cache and make
  no network calls. No audio or personal data is sent — this is an outbound
  download only.

- **Note-transcription model — `note_transcribe`.** No download. The Basic
  Pitch model ships **bundled inside the installed package**, so
  `note_transcribe` runs entirely offline.

## Optional component (not installed by default)

- **Structure analysis — `structure_analyze`.** The SongFormer model this tool
  needs is **not part of the published package** and is not installed by
  `uvx audio-analysis-mcp`. Out of the box, `structure_analyze` returns a
  graceful "not installed" error and makes **no network calls**. If you
  explicitly opt in by installing the optional `songformer` dependency, then on
  first use it downloads open model weights from the Hugging Face Hub
  (`OpenMuQ/MuQ-large-msd-iter`, a `facebook/wav2vec2-conformer-rope-large-960h-ft`
  config, and the `minzwon/MusicFM` / `ASLP-lab/SongFormer` weights fetched by
  its `songformer-download` setup step), cached under your Hugging Face cache
  (`~/.cache/huggingface/`). As with Demucs, these are downloads only — no audio
  or personal data is uploaded.

## Optional system dependencies (local only)

The audio-capture tools `audio_render` and `audio_list_devices` use
[PortAudio](https://www.portaudio.com/) to talk to your machine's audio
devices, and capturing system audio (rather than a microphone) additionally
needs a virtual device such as [BlackHole](https://existential.audio/blackhole/)
on macOS. These touch local audio hardware only — no network access. None of
the other tools need them.

No data is sent to the author or any third party in any mode.

_Last updated: 2026-06-18_
