# audio-analysis-mcp

A Python MCP server providing audio analysis tools for AI-driven sound recreation. It imports, separates, analyzes, and compares audio so an AI agent can configure hardware synthesizers to match a target sound.

## Tools

| Tool | Description |
|------|-------------|
| `import_audio` | Import a local audio file, normalize to 44.1kHz 16-bit mono WAV |
| `stem_separate` | Separate audio into stems (vocals, drums, bass, other) using Demucs |
| `audio_list_devices` | List available audio input devices |
| `audio_render` | Capture audio from a system device (BlackHole, USB audio) |
| `spectrum_analyze` | Extract mel spectrogram, spectral features, ADSR, and modulation |
| `audio_compare` | Compare target vs. synthesized audio (mel spectrogram distance, per-band energy) |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
```

`audio_render` requires [PortAudio](https://www.portaudio.com/) on the system. On macOS:

```bash
brew install portaudio
```

For system audio capture (not just microphone), install [BlackHole](https://existential.audio/blackhole/).

## Usage

The server communicates over stdio and is intended to be spawned by an MCP client:

```bash
uv run python -m audio_analysis_mcp
```

MCP client configuration:

```json
{
  "mcpServers": {
    "audio-analysis-mcp": {
      "command": "/path/to/audio-analysis-mcp/.venv/bin/python",
      "args": ["-m", "audio_analysis_mcp"]
    }
  }
}
```

## Development

```bash
uv run pytest -v       # Run tests
uv run mypy src/       # Type check
```

## License

See repository root for license information.
