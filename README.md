# audio-analysis-mcp

A Python MCP server providing audio analysis tools for AI-driven sound recreation. It imports, separates, analyzes, and compares audio so an AI agent can configure hardware synthesizers to match a target sound.

## Tools

| Tool | Description |
|------|-------------|
| `import_audio` | Import a local audio file, normalize to 44.1kHz 16-bit mono WAV |
| `stem_separate` | Separate audio into stems (vocals, drums, bass, other, guitar, piano) using Demucs |
| `audio_list_devices` | List available audio input devices |
| `audio_render` | Capture audio from a system device (BlackHole, USB audio) |
| `spectrum_analyze` | Extract mel spectrogram, spectral features, ADSR, and modulation |
| `audio_compare` | Compare target vs. synthesized audio (mel spectrogram distance, per-band energy) |
| `note_transcribe` | Polyphonic transcription via Basic Pitch — outputs MIDI + note events JSON |
| `note_triage` | Analyze transcription, select best candidate notes for isolation |
| `note_isolate` | Isolate a note from audio within a time-frequency box via STFT masking |
| `amplitude_analyze` | Per-cluster ADSR analysis with cross-candidate consistency check (**not ready for production** — see `docs/TODO.md`) |

## Install & run

Requires [uv](https://docs.astral.sh/uv/) (which provisions Python 3.11 for you — the pin matters: Basic Pitch needs CoreML/TensorFlow, which break on 3.12+).

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Add this to your MCP client config:

   ```json
   {
     "mcpServers": {
       "audio-analysis-mcp": { "command": "uvx", "args": ["audio-analysis-mcp"] }
     }
   }
   ```

3. Restart the client. The 9 core tools work out of the box; two tools need optional deps (below).

### Optional: `audio_render` (system-audio capture)

`audio_render` / `audio_list_devices` need [PortAudio](https://www.portaudio.com/) (`brew install portaudio` on macOS), plus [BlackHole](https://existential.audio/blackhole/) for system audio. Without them the server still runs; only those two tools error.

### Optional: `structure_analyze` (song-structure detection)

Needs SongFormer. Enable it by adding it to the run:

```json
{
  "mcpServers": {
    "audio-analysis-mcp": {
      "command": "uvx",
      "args": ["--with", "songformer @ git+https://github.com/uribrecher/SongFormer.git@v0.2.0", "audio-analysis-mcp"]
    }
  }
}
```

> **License note:** this pulls MuQ model weights licensed **CC-BY-NC-4.0 (non-commercial use only)**. SongFormer's own code/weights are CC-BY-4.0 (ASLP-lab/NPU).

### FastAPI service mode

The HTTP `/jobs/*` service is an optional extra: `uvx --from 'audio-analysis-mcp[service]' python -m audio_analysis_mcp.service`.

## Development

```bash
uv sync --dev --group research --extra service   # dev tools + signalflow (tone_generation tests)
uv run pytest -m "not slow"       # fast suite (CI default)
uv run mypy src/                  # type check
uv run python -m audio_analysis_mcp  # run the stdio server from source
```

## Scratch tools

The `scratch/` directory holds ad-hoc Python scripts used during research and
debugging — not part of the MCP server. They explore algorithms (clustering,
ADSR fitting, envelope shapes, overlap detection), inspect intermediate
outputs from real songs (e.g. Van Halen "Jump" triage clusters), and generate
the plots referenced from `docs/TODO.md` and `docs/research/`.

These scripts are intentionally untested and may bit-rot as the underlying
analysis modules evolve. Treat them as a notebook: useful starting points for
reproducing past experiments or scaffolding new ones, not as a stable API.
Run them directly with `uv run python scratch/<script>.py`.

## License

Licensed under the GNU General Public License v3.0 (GPL-3.0-or-later). See [LICENSE](LICENSE).

This project depends on GPL-licensed components (notably `vmo` and `cvxopt`, both
GPLv3), so the combined work is distributed under the GPL.
