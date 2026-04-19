# audio-analysis-mcp Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new Python repo with a working MCP server skeleton (stdio transport), stub tools, mypy, pytest, and CI. Tool implementations are out of scope — covered by existing plan 7.

**Architecture:** Python MCP server using the `mcp` package with `stdio_server` transport. Each tool is a stub that returns a "not implemented" response. The server starts, registers tools, and can be connected to from the agent.

**Tech Stack:** Python 3.12+, `mcp` (Python MCP SDK), `uv` (package manager), mypy, pytest

**Spec:** `../keyboards-mcp/docs/superpowers/specs/2026-04-19-multi-repo-split-design.md`

**Prerequisite:** Plan 8a (keyboards-mcp cleanup) must be complete — the parent folder `~/test/sounds-and-recreation/` must exist.

---

### Task 1: Initialize repo with uv, mypy, and CI

**Files:**
- Create: `~/test/sounds-and-recreation/audio-analysis-mcp/pyproject.toml`
- Create: `~/test/sounds-and-recreation/audio-analysis-mcp/.python-version`
- Create: `~/test/sounds-and-recreation/audio-analysis-mcp/.gitignore`
- Create: `~/test/sounds-and-recreation/audio-analysis-mcp/.github/workflows/ci.yml`

- [ ] **Step 1: Initialize git repo**

```bash
cd ~/test/sounds-and-recreation/audio-analysis-mcp
git init
```

- [ ] **Step 2: Create .python-version**

```
3.12
```

- [ ] **Step 3: Create .gitignore**

```gitignore
__pycache__/
*.pyc
.venv/
dist/
*.egg-info/
.mypy_cache/
.pytest_cache/
models/
data/
.env
```

- [ ] **Step 4: Create pyproject.toml**

```toml
[project]
name = "audio-analysis-mcp"
version = "0.1.0"
description = "MCP server for audio analysis, stem separation, and inverse synthesis"
requires-python = ">=3.12"
dependencies = [
    "mcp>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "mypy>=1.10",
    "pytest>=8.0",
]

[tool.mypy]
strict = true
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 5: Create CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --extra dev
      - run: uv run mypy src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --extra dev
      - run: uv run pytest
```

- [ ] **Step 6: Initialize uv and install dependencies**

```bash
uv sync --extra dev
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: initialize repo with uv, mypy, and CI"
```

---

### Task 2: MCP server entry point

**Files:**
- Create: `src/audio_analysis_mcp/__init__.py`
- Create: `src/audio_analysis_mcp/__main__.py`
- Create: `src/audio_analysis_mcp/server.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_server.py`:

```python
"""Test that the MCP server can be imported and has expected tools."""

from audio_analysis_mcp.server import create_server


def test_server_creates_successfully() -> None:
    server = create_server()
    assert server is not None


def test_server_has_expected_tools() -> None:
    server = create_server()
    # The server should have registered our tool stubs
    # We check by inspecting the server's tool registry
    tool_names = [t.name for t in server.list_tools()]
    expected = [
        "fetch_audio",
        "stem_separate",
        "spectrum_analyze",
        "audio_compare",
        "audio_render",
        "inverse_synth",
        "train_model",
        "list_models",
    ]
    for name in expected:
        assert name in tool_names, f"Missing tool: {name}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_server.py -v
```

Expected: FAIL — `audio_analysis_mcp` module not found.

- [ ] **Step 3: Create package structure**

Create `src/audio_analysis_mcp/__init__.py`:

```python
"""Audio Analysis MCP Server."""
```

Create `src/audio_analysis_mcp/__main__.py`:

```python
"""Entry point: python -m audio_analysis_mcp"""

import asyncio

from audio_analysis_mcp.server import run_server


def main() -> None:
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement server.py with stub tools**

Create `src/audio_analysis_mcp/server.py`:

```python
"""MCP server with stdio transport and stub tool implementations."""

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


def create_server() -> Server:
    server = Server("audio-analysis-mcp")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="fetch_audio",
                description="Download and normalize audio from YouTube URL or file path",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "YouTube URL or local file path"},
                    },
                    "required": ["source"],
                },
            ),
            Tool(
                name="stem_separate",
                description="Separate audio into stems (vocals, drums, bass, other) using Demucs",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_path": {"type": "string", "description": "Path to audio file"},
                    },
                    "required": ["audio_path"],
                },
            ),
            Tool(
                name="spectrum_analyze",
                description="Analyze spectral features: harmonic profile, ADSR, synthesis type hints",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_path": {"type": "string", "description": "Path to audio file or stem"},
                    },
                    "required": ["audio_path"],
                },
            ),
            Tool(
                name="audio_compare",
                description="Compare two audio signals and return spectral difference analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "reference_path": {"type": "string", "description": "Path to reference audio"},
                        "rendered_path": {"type": "string", "description": "Path to rendered audio"},
                    },
                    "required": ["reference_path", "rendered_path"],
                },
            ),
            Tool(
                name="audio_render",
                description="Capture audio from system audio device for a specified duration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "duration_seconds": {"type": "number", "description": "Recording duration"},
                    },
                    "required": ["duration_seconds"],
                },
            ),
            Tool(
                name="inverse_synth",
                description="Predict synthesizer parameters from audio using a trained ML model",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_path": {"type": "string", "description": "Path to audio file"},
                        "model_name": {"type": "string", "description": "Trained model to use"},
                    },
                    "required": ["audio_path", "model_name"],
                },
            ),
            Tool(
                name="train_model",
                description="Train an inverse synthesis model for a given synthesis type",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "synthesis_type": {"type": "string", "description": "e.g. subtractive, fm, organ"},
                    },
                    "required": ["synthesis_type"],
                },
            ),
            Tool(
                name="list_models",
                description="List available trained inverse synthesis models with metadata",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:  # type: ignore[type-arg]
        return [
            TextContent(
                type="text",
                text=f"Tool '{name}' is not yet implemented. Arguments: {arguments}",
            )
        ]

    return server


async def run_server() -> None:
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_server.py -v
```

Expected: PASS (2 tests).

**Note:** The test imports may need adjustment depending on the exact `mcp` Python SDK API. If `server.list_tools()` is async or returns a different structure, adapt the test accordingly. The key is that the server creates successfully and registers 8 tools.

- [ ] **Step 6: Run mypy**

```bash
uv run mypy src/
```

Expected: PASS (or fix any type errors).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: add MCP server skeleton with 8 stub tools"
```

---

### Task 3: Verify stdio transport works

- [ ] **Step 1: Test that the server starts and responds to MCP initialize**

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}}}' | uv run python -m audio_analysis_mcp
```

Expected: JSON response with server capabilities (should not hang or crash).

If it hangs waiting for more input, that's expected — stdio servers read continuously. Send the message and check that a response line appears before sending EOF.

- [ ] **Step 2: Commit any fixes**

If any fixes were needed, commit them:

```bash
git add -A
git commit -m "fix: stdio transport initialization"
```

---

### Task 4: CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Create CLAUDE.md**

```markdown
# CLAUDE.md

## Build & Run

```bash
uv sync                          # Install dependencies
uv run python -m audio_analysis_mcp   # Start MCP server (stdio transport)
```

## Linting

```bash
uv run mypy src/                 # Type checking (strict mode)
```

## Testing

```bash
uv run pytest                    # Run all tests
uv run pytest -v                 # Verbose output
```

## Architecture

Python MCP server using stdio transport. Currently has 8 stub tools — implementations are tracked in plan 7 (in keyboards-mcp repo).

### Tools

| Tool | Purpose | Status |
|------|---------|--------|
| fetch_audio | Download + normalize audio | Stub |
| stem_separate | Demucs 4-way stem separation | Stub |
| spectrum_analyze | Spectral features + synth hints | Stub |
| audio_compare | A/B spectral diff | Stub |
| audio_render | System audio capture | Stub |
| inverse_synth | ML parameter prediction | Stub |
| train_model | Train inverse synthesis model | Stub |
| list_models | Trained model inventory | Stub |

### Workspace

Part of `~/test/sounds-and-recreation/`:
- `../keyboards-mcp/` — Keyboard MCP server
- `../sound-recreation-agent/` — AI agent (TypeScript)
- `../macos-packager/` — macOS app packaging
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md"
```