# Privacy Policy — audio-analysis-mcp

audio-analysis-mcp is a local MCP server that runs on your own machine. It
imports, separates, analyzes, transcribes, and compares audio so an AI agent
can help recreate a sound. The audio you give it, and every file it produces,
stay on your machine — they are never uploaded anywhere.

It performs **no analytics, telemetry, tracking, or usage reporting**, and it
does not transmit or share your data with the author or any third party.
(Verified by source scan: the shipped server makes no outbound calls of its
own.) The audio you import and the files it generates *are* written to a local
workspace on your machine — that is the only place your data is stored.

The only network access in normal use is downloading **ML model weights**,
described below. These are cached locally and reused, so in normal operation a
model is not downloaded again — though the Hugging Face client used by the
optional SongFormer component may still make lightweight version/metadata
requests unless you run it offline.

## First-run model downloads

- **Demucs stem-separation model — `stem_separate`.** The first time you run
  `stem_separate`, the Demucs library downloads its pretrained model weights
  from Meta's public file host (`https://dl.fbaipublicfiles.com/demucs/`) and
  caches them in your local Torch hub cache (by default
  `~/.cache/torch/hub/checkpoints/`, configurable via `TORCH_HOME` and
  platform-dependent). Once cached, later runs load the weights from disk
  without re-downloading; a download recurs only if the cache is missing or
  cleared. No audio or personal data is sent — this is an outbound download
  only.

- **Note-transcription model — `note_transcribe`.** No download. The Basic
  Pitch model ships **bundled inside the installed package**, so
  `note_transcribe` runs entirely offline.

## Optional component (not installed by default)

- **Structure analysis — `structure_analyze`.** The SongFormer model this tool
  needs is **not part of the published package** and is not installed by
  `uvx audio-analysis-mcp`. Out of the box, `structure_analyze` returns a
  graceful "not installed" error and makes **no network calls**. If you
  explicitly opt in by installing the optional `songformer` dependency, then on
  first use it downloads model weights from the Hugging Face Hub
  (`OpenMuQ/MuQ-large-msd-iter`, a `facebook/wav2vec2-conformer-rope-large-960h-ft`
  config, and the `minzwon/MusicFM` / `ASLP-lab/SongFormer` weights fetched by
  its `songformer-download` setup step), cached under your Hugging Face cache
  (by default `~/.cache/huggingface/`, configurable via `HF_HOME` /
  `HUGGINGFACE_HUB_CACHE` and platform-dependent). Note that the MuQ weights
  (`OpenMuQ/MuQ-large-msd-iter`) are licensed CC-BY-NC-4.0 (non-commercial).
  These transfers send no audio or personal data — they fetch model weights and,
  via the Hugging Face client, may also make version/metadata requests.

## Optional system dependencies (local only)

The audio-capture tools `audio_render` and `audio_list_devices` use
[PortAudio](https://www.portaudio.com/) to talk to your machine's audio
devices, and capturing system audio (rather than a microphone) additionally
needs a virtual device such as [BlackHole](https://existential.audio/blackhole/)
on macOS. These touch local audio hardware only — no network access. None of
the other tools need them.

No data is sent to the author or any third party in any mode.

_Last updated: 2026-06-18_
