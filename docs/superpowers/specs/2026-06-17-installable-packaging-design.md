# Installable Packaging — Design Spec

**Date:** 2026-06-17
**Status:** design direction agreed; spec under review
**Issue:** [#46](https://github.com/uribrecher/audio-analysis-mcp/issues/46) — "Usability & installability: simple install + run without manual env/server setup"
**Adoption context:** unblocks the MCP-registry publish + Claude-plugin submission (umbrella `sounds-and-recreation#9`) and the Hugging Face Space (`sounds-and-recreation#36`). `keyboards-mcp` is already packaged; this is the audio side.

## Goal

A new user installs `audio-analysis-mcp` and drives it from an MCP client in ≤ 3 copy-paste steps — no hand-assembled Python/uv/system-deps setup, no manually-launched server, no absolute `.venv` path in the config. The pinned Python 3.11 environment is provisioned for them.

## Success criteria (from #46 "done when")

- Zero-to-working MCP connection in ≤ ~3 documented, copy-paste steps — no manual venv/build.
- Python 3.11 is provisioned automatically, so the 3.12 footgun (Basic Pitch / CoreML) cannot happen.
- The analysis server is spawned by the client per the documented config — no separate manual server step.
- A canonical MCP-client config snippet lives in the README and works verbatim.
- Optional system deps (PortAudio/BlackHole) are clearly gated to the features that need them, so core tools work without them.

## Decisions (locked during brainstorming)

1. **Distribution mechanism: PyPI + `uvx`.** Idiomatic for MCP clients, auto-provisions Python 3.11, and is what the MCP registry expects. (Considered and rejected: Docker/OCI — heavier UX for an stdio server, awkward audio-device capture; standalone binary — torch + native ML deps make PyInstaller fragile and huge.)
2. **SongFormer / `structure_analyze`: decouple.** Ship the core package with **no** SongFormer dependency (it is the only git/direct-reference, which PyPI forbids). `structure_analyze` becomes an opt-in. Publishing `songformer` to PyPI is a **separate follow-up** (see Out of scope).
3. **Project framing: GPL-3.0 OSS, non-commercial.** The MuQ CC-BY-NC weight restriction (reached transitively by `structure_analyze`) is not a problem for the project's own use; it is **disclosed** for downstream users of the opt-in tool.

## Design

### 1. Distribution & invocation

Publish `audio-analysis-mcp` to PyPI. Users install/run via `uvx`, which creates an isolated ephemeral environment and auto-provisions a compatible Python (3.11, honoring `requires-python = ">=3.11,<3.12"`). The canonical MCP-client config:

```json
{
  "mcpServers": {
    "audio-analysis-mcp": { "command": "uvx", "args": ["audio-analysis-mcp"] }
  }
}
```

The only prerequisite is `uv` on the user's PATH. The three steps: (1) install uv, (2) paste the config block, (3) restart the MCP client.

### 2. Dependency partition

Today's single dependency list is split so the published package installs only what the stdio MCP server's tools need:

- **Core runtime:** `mcp`, `torch`, `torchaudio`, `torchcodec`, `demucs`, `librosa`, `basic-pitch`, `numpy`, `scipy`, `soundfile`, `sounddevice`, `pydantic`, `jsonschema`, `referencing`.
- **Removed from runtime → dedicated `research` dependency-group:** `signalflow`. It is imported only by `research/tone_generation/renderer.py` (the subtractive-tone-generation **training** MVP) and reached only by the `scripts/{train,eval,generate}_*` training scripts — never by any shipped `tools/`, `analysis/`, `audio/` module or the stdio server. It belongs with the training pipeline, not MCP dev tooling, so it goes in its own `research` group (`uv sync --group research`), not the `dev` group. Removing it from runtime deps **eliminates the native macOS-wheel ceiling** (`signalflow<0.5.4`) from the published package. (This `research` group is a **temporary home**: [#47](https://github.com/uribrecher/audio-analysis-mcp/issues/47) tracks moving the whole tone-generation training pipeline — including `signalflow` — into the `reverse-synth-research` repo, after which the group disappears entirely.)
- **`[service]` extra:** `fastapi`, `uvicorn[standard]`, `sse-starlette`. Used only by the separate `service/` subpackage (FastAPI `/jobs/*` SSE mode), which the stdio MCP entry point never imports. Service users install `audio-analysis-mcp[service]`.
- **`songformer`: removed from core deps entirely.** It is the only direct/git reference (`git+https://github.com/uribrecher/SongFormer.git@v0.2.0`) and PyPI rejects direct references in uploaded packages. The `[tool.uv.sources]` editable/path override and `allow-direct-references` become unnecessary for the core package.

### 3. Entry point

`__main__.py` currently calls `mcp.run(transport="stdio")` at module top level (works for `python -m audio_analysis_mcp`, but there is no console-script callable). Refactor the run logic into a `main()` function and register `[project.scripts]`:

```toml
[project.scripts]
audio-analysis-mcp = "audio_analysis_mcp.__main__:main"
```

So `uvx audio-analysis-mcp` launches the stdio server. `python -m audio_analysis_mcp` continues to work (the `if __name__` / module path calls `main()`).

### 4. Graceful degradation (core always starts)

The lazy-import architecture already supports this; the design formalizes and tests it:

- **`sounddevice` / PortAudio** — already lazy-imported inside `audio/capture.py`. Server starts without PortAudio; `audio_render` returns a clear "install PortAudio (and BlackHole for system capture)" error only when invoked.
- **`songformer`** — already imported only under `TYPE_CHECKING` and lazily inside `server.get_structure_pipeline()`; `analysis/structure_analysis.py` takes the pipeline as a `Protocol` argument and never imports it. Server starts without songformer; `structure_analyze` returns a clear actionable error when invoked:
  > "structure_analyze requires SongFormer. Install with: `uvx --with 'songformer @ git+https://github.com/uribrecher/SongFormer.git@v0.2.0' audio-analysis-mcp`. Note: this pulls MuQ model weights licensed CC-BY-NC-4.0 (non-commercial)."

No tool import at server startup may eagerly import an optional/native dependency.

### 5. Publish flow

- Build with `hatchling` (already configured).
- GitHub Actions workflow using **PyPI Trusted Publishing (OIDC)** — no long-lived token — triggered on a version tag (`v*`).
- Pre-publish dry run: `uv build` + `twine check dist/*` (and the clean-env smoke test below).
- The **user** authorizes/cuts the release (tags the version); Claude prepares the workflow and the dry run. Per project rules: branch + PR, signed commits, never push directly to `main`.

### 6. README rewrite

- 3-step quickstart + the canonical `uvx` config snippet (replacing the absolute `.venv/bin/python` snippet).
- Clearly-gated optional sections: (a) `audio_render` system deps (PortAudio/BlackHole); (b) the `structure_analyze` `--with` opt-in, **with the MuQ CC-BY-NC disclosure**.
- Keep the existing GPL-3.0 license section; note the `[service]` extra for the FastAPI mode.

### 7. Verification

1. `uv build` → wheel in `dist/`.
2. Install into a clean ephemeral env and run via `uvx --from ./dist/audio_analysis_mcp-*.whl audio-analysis-mcp` **with no hand-made venv, no PortAudio, no songformer**.
3. Drive a real MCP handshake over stdio (`initialize` + `tools/list`) and call one light tool (`spectrum_analyze` on a synthetic sine/square fixture).
4. Assert: server starts; `tools/list` returns all registered tools (including `structure_analyze`); the light tool returns; `audio_render` and `structure_analyze` each return their graceful error (not a crash) when invoked without their optional dependency.
5. Add this as a clean-runner smoke test in CI (separate from the existing `pytest -m "not slow"` job, since it exercises packaging, not logic).

## Scope

**In:** PyPI packaging metadata, the `uvx` turnkey path, the dependency partition (core / `[service]` extra / research-dev), the `main()` entry point, graceful-degradation guarantees + tests, the publish workflow (Trusted Publishing), the README rewrite, and the clean-env smoke test.

## Out of scope (separate follow-ups)

- **Publishing `songformer` to PyPI** — its own issue: correct the mislabeled license (`pyproject` says Apache-2.0; upstream is **CC-BY-4.0**), add a NOTICE crediting ASLP-lab/NPU + the arXiv citation + "changes made", mark it an unofficial community wrapper, and send ASLP-lab a courtesy heads-up. Once published, `structure_analyze` can become a clean `[structure]` PyPI extra instead of the `--with` opt-in.
- **`macos-packager` .app/.dmg** — the bundled-product distribution is a different track.
- **Distributing the FastAPI `service/` mode** — this spec only moves its deps behind `[service]`; turnkey packaging of the service is not addressed.
- **New analysis tools or ML-accuracy work** (e.g. productionizing `amplitude_analyze`).

## Deliverables

1. `pyproject.toml` — dependency partition (core / `[service]` extra / dedicated `research` group for `signalflow`), `[project.scripts]` entry point, removal of the songformer direct reference and now-unneeded `[tool.uv.sources]`/`allow-direct-references`.
2. `__main__.py` — `main()` function; module path still runnable.
3. Graceful-degradation error messages for `audio_render` and `structure_analyze`.
4. GitHub Actions publish workflow (Trusted Publishing, tag-triggered) + dry-run docs.
5. README rewrite (3-step quickstart, canonical config, gated optional sections, MuQ disclosure).
6. Clean-env packaging smoke test + CI job.
