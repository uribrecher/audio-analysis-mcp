# audio-analysis-mcp

Python MCP server providing audio analysis tools for sound recreation.

## Quick Reference

```bash
uv sync --dev --group research --extra service   # Install dev + research (signalflow) deps
uv run pytest -v           # Run all tests
uv run mypy src/           # Type check
uv run python -m audio_analysis_mcp  # Run MCP server (stdio)
```

## Architecture

FastMCP server (stdio transport). Tools registered via `@mcp.tool()` in `src/audio_analysis_mcp/tools/`.
Each tool has pure logic (in `analysis/` or `audio/`) + thin MCP wrapper (in `tools/`).
Unit tests target logic modules directly. E2E tests call MCP tool functions.

## Testing

- `pytest` with synthetic audio fixtures (sine/square waves via numpy)
- Tests that require ML model downloads or hardware (sounddevice) are mocked in unit tests
- CI runs `pytest -m "not slow"` (use `@pytest.mark.slow` for future tests needing real ML models)

## Releases

Versions + `CHANGELOG.md` are commit-derived via Commitizen — see README "Releases". **Every PR
title must be a valid Conventional Commit** (gated by the `pr-title` check); the squash-merged title
is what `cz bump` reads. Releases are cut by merging the auto-generated `chore(release):` PR, then
approving the `pypi` environment. Never hand-edit `[project].version` or `CHANGELOG.md`.
