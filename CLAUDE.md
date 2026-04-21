# audio-analysis-mcp

Python MCP server providing audio analysis tools for sound recreation.

## Quick Reference

```bash
uv sync --dev              # Install all dependencies
uv run pytest -v           # Run fast tests
uv run pytest -m slow      # Run slow tests (need ML models)
uv run mypy src/           # Type check
uv run python -m audio_analysis_mcp  # Run MCP server (stdio)
```

## Architecture

FastMCP server (stdio transport). Tools registered via `@mcp.tool()` in `src/audio_analysis_mcp/tools/`.
Each tool has pure logic (in `analysis/` or `audio/`) + thin MCP wrapper (in `tools/`).
Unit tests target logic modules directly. E2E tests call MCP tool functions.

## Testing

- `pytest` with synthetic audio fixtures (sine/square waves via numpy)
- `@pytest.mark.slow` for tests needing ML models (Demucs, CLAP)
- CI runs `pytest -m "not slow"`
