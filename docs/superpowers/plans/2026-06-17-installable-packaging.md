# Installable Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `audio-analysis-mcp` installable and runnable from any MCP client in ≤3 copy-paste steps via `uvx audio-analysis-mcp`, with no manual venv/server setup.

**Architecture:** Publish to PyPI; users run `uvx audio-analysis-mcp` (auto-provisions Python 3.11). Split today's single dependency list into a lean core (the 11 stdio tools), a `[service]` extra (FastAPI mode), a `research` group (`signalflow`, training-only), and a `dev` group (keeps `songformer` for typecheck/tests). Optional features (`audio_render` → PortAudio, `structure_analyze` → SongFormer) degrade gracefully so the core server always starts.

**Tech Stack:** Python 3.11, `uv`/`uvx`, hatchling build backend, FastMCP (`mcp` SDK), pytest + pytest-asyncio, GitHub Actions + PyPI Trusted Publishing (OIDC).

## Global Constraints

These apply to every task. Exact values copied from the spec (`docs/superpowers/specs/2026-06-17-installable-packaging-design.md`):

- **Python pin:** `requires-python = ">=3.11,<3.12"` — never widen it (Basic Pitch / CoreML break on 3.12+).
- **Published `Requires-Dist` MUST NOT contain:** `songformer` (or any `@ git+` direct reference), `signalflow`, `fastapi`, `uvicorn`, `sse-starlette`. The published wheel must have **zero direct-reference dependencies**.
- **Distribution name == console-script name ==** `audio-analysis-mcp`.
- **Canonical MCP client config** (must appear verbatim in the README): `{"mcpServers":{"audio-analysis-mcp":{"command":"uvx","args":["audio-analysis-mcp"]}}}`.
- **Dev/CI sync command becomes:** `uv sync --dev --group research --extra service` (the `research` group holds `signalflow` for the not-yet-moved `tone_generation` tests — see #47; `--extra service` installs the FastAPI deps that the `service/` tests + `mypy src/` require — established during Task 2 implementation).
- **CI keeps** `UV_NO_SOURCES: "1"` and `TONE_GEN_SCHEMA_DIR` env exactly as today.
- **`structure_analyze` graceful-error text** must name SongFormer, give the `uvx --with 'songformer @ git+https://github.com/uribrecher/SongFormer.git@v0.2.0' audio-analysis-mcp` opt-in, and disclose **MuQ weights are CC-BY-NC-4.0 (non-commercial)**.
- **Git:** signed commits only; work stays on the worktree branch (`worktree-audio-mcp-packaging-spec`); never push to `main`. (Out of scope: publishing SongFormer to PyPI; the `service/` distribution; the training-pipeline move #47.)

---

### Task 1: Console-script entry point (`main()`)

Make `uvx audio-analysis-mcp` launch the stdio server. Today `__main__.py` runs `mcp.run()` at import time with no callable entry point.

**Files:**
- Modify: `src/audio_analysis_mcp/__main__.py` (whole file)
- Modify: `pyproject.toml` (add `[project.scripts]`)
- Test: `tests/test_entry_point.py` (create)

**Interfaces:**
- Produces: `audio_analysis_mcp.__main__.main() -> None` (registers all tools, runs stdio server); `audio_analysis_mcp.__main__._register_tools() -> None` (imports every tool module to register it on the shared `mcp` singleton).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_entry_point.py
from audio_analysis_mcp.__main__ import main, _register_tools
from audio_analysis_mcp.server import mcp

EXPECTED_TOOLS = {
    "import_audio", "stem_separate", "audio_list_devices", "audio_render",
    "spectrum_analyze", "audio_compare", "note_transcribe", "note_triage",
    "note_isolate", "amplitude_analyze", "structure_analyze",
}


async def test_register_tools_registers_all() -> None:
    _register_tools()
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS <= names


def test_main_is_callable() -> None:
    assert callable(main)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_entry_point.py -v`
Expected: FAIL — `ImportError: cannot import name 'main'` (and `_register_tools`).

- [ ] **Step 3: Rewrite `__main__.py`**

```python
from audio_analysis_mcp.server import mcp


def _register_tools() -> None:
    """Import every tool module so its ``@mcp.tool()`` decorator registers it."""
    import audio_analysis_mcp.tools.import_audio  # noqa: F401
    import audio_analysis_mcp.tools.stem_separate  # noqa: F401
    import audio_analysis_mcp.tools.audio_render  # noqa: F401
    import audio_analysis_mcp.tools.spectrum_analyze  # noqa: F401
    import audio_analysis_mcp.tools.audio_compare  # noqa: F401
    import audio_analysis_mcp.tools.note_transcribe  # noqa: F401
    import audio_analysis_mcp.tools.note_triage  # noqa: F401
    import audio_analysis_mcp.tools.note_isolate  # noqa: F401
    import audio_analysis_mcp.tools.amplitude_analyze  # noqa: F401
    import audio_analysis_mcp.tools.structure_analyze  # noqa: F401


def main() -> None:
    """Console-script + ``python -m`` entry point: register tools, run stdio server."""
    _register_tools()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the console script to `pyproject.toml`**

Add this block (after the `[project]` table, before `[tool.uv.sources]`):

```toml
[project.scripts]
audio-analysis-mcp = "audio_analysis_mcp.__main__:main"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_entry_point.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git add src/audio_analysis_mcp/__main__.py pyproject.toml tests/test_entry_point.py
git commit -S -m "feat(packaging): add main() console-script entry point"
```

---

### Task 2: Dependency partition in `pyproject.toml`

Make the published wheel lean and direct-reference-free. Move `fastapi`/`uvicorn`/`sse-starlette` → `[service]` extra, `signalflow` → `research` group, `songformer` → `dev` group (out of published metadata, but kept for dev/typecheck), and drop the now-unneeded `allow-direct-references`.

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml` (sync the `research` group)
- Test: `tests/test_packaging_metadata.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: a buildable wheel whose `METADATA` satisfies the Global Constraints.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_packaging_metadata.py
import subprocess
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = ("songformer", "signalflow", "fastapi", "uvicorn", "sse-starlette", "sse_starlette")


@pytest.fixture(scope="module")
def wheel_metadata(tmp_path_factory: pytest.TempPathFactory) -> str:
    out = tmp_path_factory.mktemp("wheelbuild")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out)],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    )
    wheel = next(out.glob("*.whl"))
    with zipfile.ZipFile(wheel) as zf:
        meta_name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        return zf.read(meta_name).decode()


def test_core_requires_have_no_forbidden_or_direct_refs(wheel_metadata: str) -> None:
    msg = Parser().parsestr(wheel_metadata)
    requires = msg.get_all("Requires-Dist") or []
    core = [r for r in requires if "extra ==" not in r]  # core deps only
    for r in core:
        assert "@" not in r, f"direct reference leaked into core deps: {r}"
        assert not any(f in r.lower() for f in FORBIDDEN), f"forbidden core dep: {r}"


def test_service_extra_present(wheel_metadata: str) -> None:
    msg = Parser().parsestr(wheel_metadata)
    assert "service" in (msg.get_all("Provides-Extra") or [])
    requires = msg.get_all("Requires-Dist") or []
    service = [r for r in requires if 'extra == "service"' in r]
    assert any("fastapi" in r for r in service)
    assert any("uvicorn" in r for r in service)


def test_core_keeps_essential_deps(wheel_metadata: str) -> None:
    msg = Parser().parsestr(wheel_metadata)
    core = " ".join(r for r in (msg.get_all("Requires-Dist") or []) if "extra ==" not in r)
    for dep in ("mcp", "torch", "demucs", "librosa", "basic-pitch", "sounddevice"):
        assert dep in core, f"core dep missing: {dep}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_packaging_metadata.py -v`
Expected: FAIL — current core deps still contain `songformer @ git+...`, `signalflow`, `fastapi`, etc., and there is no `service` extra.

- [ ] **Step 3: Edit `pyproject.toml` — core `[project.dependencies]`**

Replace the `dependencies = [...]` list (lines 5–26) so it no longer contains `songformer`, `signalflow`, `fastapi`, `uvicorn[standard]`, or `sse-starlette`:

```toml
dependencies = [
  "mcp>=1.0.0",
  "demucs>=4.0.0",
  "librosa>=0.10.0",
  "torch>=2.0",
  "torchaudio>=2.0",
  "torchcodec>=0.1",
  "numpy>=1.24",
  "scipy>=1.10",
  "soundfile>=0.12",
  "sounddevice>=0.4",
  "pydantic>=2.0",
  "basic-pitch>=0.4.0",
  "jsonschema>=4.21",
  "referencing>=0.30",
]

[project.optional-dependencies]
# FastAPI service mode (service/ subpackage). Not needed for the stdio MCP server.
service = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "sse-starlette>=2.1",
]
```

- [ ] **Step 4: Edit `pyproject.toml` — dependency groups**

Update `[dependency-groups]` so `dev` keeps `songformer` (git, dev-only — not published) and a new `research` group holds `signalflow`:

```toml
[dependency-groups]
dev = [
  "matplotlib>=3.10.9",
  "mypy>=1.10",
  "pytest>=8.0",
  "pytest-asyncio>=0.24",
  "httpx>=0.27",
  # Dev-only: structure_analyze typecheck + e2e. Published wheel never includes it
  # (dependency-groups are not part of distribution metadata). End users opt in via
  # `uvx --with`. [tool.uv.sources] redirects this to ../SongFormer for local editable
  # dev; CI sets UV_NO_SOURCES=1 so the pinned git URL wins.
  "songformer @ git+https://github.com/uribrecher/SongFormer.git@v0.2.0",
]
# Training-only (tone_generation renderer). Leaves the published package entirely.
# Temporary home until #47 moves the training pipeline to reverse-synth-research.
research = [
  "signalflow>=0.4.0,<0.5.4",
]
```

- [ ] **Step 5: Edit `pyproject.toml` — drop `allow-direct-references`**

Delete the now-unneeded block (the published metadata has no direct references):

```toml
# DELETE these lines:
# [tool.hatch.metadata]
# allow-direct-references = true
```

Keep `[tool.uv.sources]` (the `songformer` editable path override still applies to the dev group for local dev).

- [ ] **Step 6: Update CI to sync the `research` group**

In `.github/workflows/ci.yml`, change the sync step so the still-present `tone_generation` tests get `signalflow`:

```yaml
      - run: uv sync --dev --group research
```

(Replaces the existing `- run: uv sync --dev`. Leave the `env:` block, `UV_NO_SOURCES`, and `TONE_GEN_SCHEMA_DIR` unchanged.)

- [ ] **Step 7: Run the metadata test + full fast suite + mypy**

Run: `uv sync --dev --group research`
Run: `uv run pytest tests/test_packaging_metadata.py -v`
Expected: PASS (3 tests).
Run: `uv run pytest -m "not slow" -q && uv run mypy src/`
Expected: PASS — nothing else broke (structure_analyze e2e mocks the pipeline; tone_generation tests still have signalflow).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml tests/test_packaging_metadata.py
git commit -S -m "feat(packaging): partition deps (core / [service] extra / research+dev groups)"
```

---

### Task 3: Graceful degradation for optional deps

Guarantee the server starts without PortAudio or SongFormer, and that `audio_render`/`structure_analyze` return clear, actionable errors when invoked without their optional dependency.

**Files:**
- Modify: `src/audio_analysis_mcp/audio/capture.py` (the `_get_sd` helper)
- Modify: `src/audio_analysis_mcp/server.py` (the `get_structure_pipeline` function)
- Test: `tests/test_graceful_degradation.py` (create)

**Interfaces:**
- Consumes: `audio_analysis_mcp.audio.capture._get_sd`, `audio_analysis_mcp.server.get_structure_pipeline`.
- Produces: both raise `RuntimeError` with the friendly text below when their dep is missing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graceful_degradation.py
import sys

import pytest

import audio_analysis_mcp.server as srv
from audio_analysis_mcp.audio.capture import _get_sd


def test_get_sd_without_portaudio_raises_friendly(monkeypatch: pytest.MonkeyPatch) -> None:
    # Setting the module to None makes `import sounddevice` raise ImportError.
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(RuntimeError, match="PortAudio"):
        _get_sd()


def test_get_structure_pipeline_without_songformer_raises_friendly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "songformer", None)
    monkeypatch.setattr(srv, "_structure_pipeline", None)
    with pytest.raises(RuntimeError) as exc:
        srv.get_structure_pipeline()
    msg = str(exc.value)
    assert "SongFormer" in msg
    assert "uvx --with" in msg
    assert "CC-BY-NC" in msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graceful_degradation.py -v`
Expected: FAIL — current code raises bare `ImportError`/`ModuleNotFoundError`, not a `RuntimeError` with the matched text.

- [ ] **Step 3: Harden `_get_sd` in `audio/capture.py`**

Replace the existing `_get_sd` (lines 8–11):

```python
def _get_sd() -> Any:
    """Lazy-import sounddevice (requires the PortAudio system library)."""
    try:
        import sounddevice as sd
    except (OSError, ImportError) as exc:
        raise RuntimeError(
            "audio_render / audio_list_devices require the PortAudio system "
            "library. Install it (e.g. `brew install portaudio` on macOS) and, "
            "for system-audio capture, BlackHole "
            "(https://existential.audio/blackhole/). The other tools work "
            "without it."
        ) from exc
    return sd
```

- [ ] **Step 4: Harden `get_structure_pipeline` in `server.py`**

Replace the function body (lines 24–30) so the lazy import is guarded:

```python
def get_structure_pipeline() -> "SongFormerPipeline":
    global _structure_pipeline
    if _structure_pipeline is None:
        try:
            from songformer import SongFormerPipeline
        except ImportError as exc:
            raise RuntimeError(
                "structure_analyze requires SongFormer, which is not installed. "
                "Enable it with: uvx --with "
                "'songformer @ git+https://github.com/uribrecher/SongFormer.git@v0.2.0' "
                "audio-analysis-mcp . Note: this pulls MuQ model weights licensed "
                "CC-BY-NC-4.0 (non-commercial use only)."
            ) from exc

        _structure_pipeline = SongFormerPipeline.from_pretrained()
    return _structure_pipeline
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_graceful_degradation.py -v`
Expected: PASS (2 tests).
Run: `uv run pytest -m "not slow" -q && uv run mypy src/`
Expected: PASS (existing `test_audio_render`/`test_structure_analyze_e2e` still pass — they don't exercise the missing-dep path).

- [ ] **Step 6: Commit**

```bash
git add src/audio_analysis_mcp/audio/capture.py src/audio_analysis_mcp/server.py tests/test_graceful_degradation.py
git commit -S -m "feat(packaging): graceful errors when PortAudio/SongFormer are absent"
```

---

### Task 4: Clean-env packaging smoke test + CI job

Prove the end-to-end turnkey path: build the wheel, run it in an isolated env via `uvx`, and drive a real MCP handshake — with no hand-made venv, no PortAudio, no SongFormer.

**Files:**
- Create: `scripts/smoke_packaging.py`
- Modify: `.github/workflows/ci.yml` (add a separate `packaging-smoke` job)

**Interfaces:**
- Consumes: the console-script entry point (Task 1), the lean wheel (Task 2), graceful errors (Task 3).
- Produces: `scripts/smoke_packaging.py` — exits `0` on success, non-zero on any failed assertion.

- [ ] **Step 1: Write the smoke script**

```python
# scripts/smoke_packaging.py
"""End-to-end packaging smoke test.

Builds nothing itself — expects a pre-built wheel in dist/. Spawns the
installed server in an isolated uvx env and drives an MCP stdio session:
the server must start (proving the turnkey install + Python 3.11 + no
import-time failure from missing optional deps), list all tools, run a
light tool, and degrade gracefully for SongFormer (and PortAudio, on a
headless runner via --expect-no-portaudio).

Usage:
    python scripts/smoke_packaging.py [--expect-no-portaudio]
"""
from __future__ import annotations

import asyncio
import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {
    "import_audio", "stem_separate", "audio_list_devices", "audio_render",
    "spectrum_analyze", "audio_compare", "note_transcribe", "note_triage",
    "note_isolate", "amplitude_analyze", "structure_analyze",
}


def _make_sine_wav(path: Path) -> None:
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(str(path), 0.5 * np.sin(2 * np.pi * 440 * t), sr, subtype="PCM_16")


def _text(result) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


async def run(wheel: str, expect_no_portaudio: bool) -> None:
    params = StdioServerParameters(command="uvx", args=["--from", wheel, "audio-analysis-mcp"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert EXPECTED_TOOLS <= names, f"missing tools: {EXPECTED_TOOLS - names}"

            with tempfile.TemporaryDirectory() as d:
                wav = Path(d) / "sine.wav"
                _make_sine_wav(wav)

                imported = await session.call_tool("import_audio", {"file_path": str(wav)})
                src = json.loads(_text(imported))["audio_path"]

                spec = await session.call_tool(
                    "spectrum_analyze", {"audio_path": src, "duration": 1.0}
                )
                f0 = json.loads(_text(spec))["spectral_features"]["fundamental_hz"]
                assert f0 is not None and abs(f0 - 440) < 10, f"bad fundamental: {f0}"

                structure = await session.call_tool("structure_analyze", {"audio_path": src})
                assert structure.isError and "SongFormer" in _text(structure), _text(structure)

                if expect_no_portaudio:
                    render = await session.call_tool("audio_render", {"duration": 0.1})
                    assert render.isError and "PortAudio" in _text(render), _text(render)

    print("packaging smoke: OK")


def main() -> int:
    expect_no_portaudio = "--expect-no-portaudio" in sys.argv[1:]
    matches = sorted(glob.glob("dist/*.whl"))
    if not matches:
        print("no wheel in dist/ — run `uv build --wheel` first", file=sys.stderr)
        return 1
    asyncio.run(run(matches[-1], expect_no_portaudio))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Build the wheel and run the smoke script locally**

Run: `uv build --wheel`
Run: `uv run python scripts/smoke_packaging.py`
Expected: prints `packaging smoke: OK` and exits 0. (First run downloads the isolated env incl. torch — expect several minutes. Omit `--expect-no-portaudio` locally on macOS where PortAudio is installed.)

- [ ] **Step 3: Add the `packaging-smoke` CI job**

Append to `.github/workflows/ci.yml` (a second job, alongside `test-and-lint`):

```yaml
  packaging-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          python-version: "3.11"
      - run: uv build --wheel
      # Outer env needs numpy/soundfile + the mcp client to drive the session.
      - run: uv run --with numpy --with soundfile --with mcp python scripts/smoke_packaging.py --expect-no-portaudio
```

(The ubuntu runner has no PortAudio and the isolated wheel env has no SongFormer, so `--expect-no-portaudio` exercises both graceful-degradation paths.)

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_packaging.py .github/workflows/ci.yml
git commit -S -m "test(packaging): clean-env uvx smoke test + CI job"
```

---

### Task 5: README rewrite + dev-docs

Replace the from-source setup with the 3-step `uvx` quickstart and the canonical config; gate the optional features; fix the dev sync command.

**Files:**
- Modify: `README.md` (Setup + Usage + Development sections)
- Modify: `CLAUDE.md` (Quick Reference sync command)

**Interfaces:** none (documentation).

- [ ] **Step 1: Rewrite the README `## Setup` and `## Usage` sections**

Replace the current Setup + Usage (lines 20–55) with:

````markdown
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
````

- [ ] **Step 2: Update the README `## Development` section**

Replace the Development block (lines 57–62) with:

````markdown
## Development

```bash
uv sync --dev --group research   # dev tools + signalflow (tone_generation tests)
uv run pytest -m "not slow"       # fast suite (CI default)
uv run mypy src/                  # type check
uv run python -m audio_analysis_mcp  # run the stdio server from source
```
````

- [ ] **Step 3: Update `CLAUDE.md` Quick Reference**

In `CLAUDE.md`, change the install line:

```
uv sync --dev --group research   # Install dev + research (signalflow) deps
```

(Replaces `uv sync --dev              # Install all dependencies`.)

- [ ] **Step 4: Verify the package long-description renders for PyPI**

Run: `uv build && uvx twine check dist/*`
Expected: `Checking dist/...: PASSED` for both sdist and wheel.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -S -m "docs(packaging): uvx quickstart, gated optional tools, research-group dev sync"
```

---

### Task 6: PyPI publish workflow (Trusted Publishing)

Add a tag-triggered release workflow using OIDC Trusted Publishing — no stored token. The user cuts the tag; the workflow builds and publishes.

**Files:**
- Create: `.github/workflows/publish.yml`

**Interfaces:** none (CI/CD).

- [ ] **Step 1: Create the publish workflow**

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI
on:
  push:
    tags: ["v*"]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          python-version: "3.11"
      - run: uv build
      - run: uvx twine check dist/*
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write   # OIDC for Trusted Publishing
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 2: Validate the workflow YAML parses**

Run: `uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/publish.yml')); print('yaml ok')"`
Expected: prints `yaml ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/publish.yml
git commit -S -m "ci(packaging): tag-triggered PyPI publish via Trusted Publishing"
```

- [ ] **Step 4: Manual prerequisites (note for the user — not an automated step)**

Before the first tag-push release, the user must: (1) create the PyPI project `audio-analysis-mcp` (or pre-register the name) and configure a **Trusted Publisher** pointing at this repo + `publish.yml` + the `pypi` environment; (2) create the `pypi` GitHub environment. Then a `git tag vX.Y.Z && git push --tags` triggers the release. These require the user's PyPI/GitHub identity and are intentionally outside this plan.

---

## Self-Review

**Spec coverage:**
- §1 Distribution & invocation → Task 1 (entry point) + Task 5 (canonical config in README). ✓
- §2 Dependency partition → Task 2 (+ metadata test). ✓
- §3 Entry point → Task 1. ✓
- §4 Graceful degradation → Task 3. ✓
- §5 Publish flow → Task 6 (+ `twine check` in Task 5/6). ✓
- §6 README rewrite → Task 5. ✓
- §7 Verification → Task 4 (smoke) + Task 2 (metadata) + per-task `pytest`/`mypy`. ✓
- §8 Scope boundaries → Global Constraints + Task 6 Step 4 (SongFormer publish, service, #47 all kept out). ✓

**Placeholder scan:** No TBD/TODO; every code/edit step shows concrete code or exact diffs. ✓

**Type consistency:** `_register_tools`/`main` (Task 1) match the smoke `EXPECTED_TOOLS` set (Task 4) and the test set (Task 1) — all 11 tool names identical. `get_structure_pipeline` (Task 3) keeps its existing signature; `_get_sd` keeps `-> Any`. The `service` extra name (Task 2) matches the README (`[service]`, Task 5). The `research` group name (Task 2) matches CI + README + CLAUDE (Tasks 2/5). ✓
