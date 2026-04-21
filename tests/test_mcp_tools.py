"""E2E tests: call MCP tool functions directly and verify JSON output."""
import json
from pathlib import Path

import audio_analysis_mcp.server as srv
from audio_analysis_mcp.workspace import Workspace

# Import tools to register them on the mcp instance
import audio_analysis_mcp.tools.import_audio  # noqa: F401
import audio_analysis_mcp.tools.stem_separate  # noqa: F401
import audio_analysis_mcp.tools.audio_render  # noqa: F401
import audio_analysis_mcp.tools.spectrum_analyze  # noqa: F401
import audio_analysis_mcp.tools.audio_compare  # noqa: F401

import pytest


@pytest.fixture(autouse=True)
def _use_tmp_workspace(tmp_path: Path):
    old = srv._workspace
    srv._workspace = Workspace(tmp_path / "workspace")
    yield
    srv._workspace = old


def test_import_audio_e2e(sine_440_wav: Path):
    from audio_analysis_mcp.tools.import_audio import import_audio

    result = json.loads(import_audio(file_path=str(sine_440_wav)))
    assert result["sample_rate"] == 44100
    assert result["channels"] == 1
    assert Path(result["audio_path"]).exists()


def test_spectrum_analyze_e2e(sine_440_wav: Path):
    from audio_analysis_mcp.tools.spectrum_analyze import spectrum_analyze

    result = json.loads(spectrum_analyze(audio_path=str(sine_440_wav), duration=1.0))
    assert result["spectral_features"]["fundamental_hz"] is not None
    assert abs(result["spectral_features"]["fundamental_hz"] - 440) < 10
    assert Path(result["mel_spectrogram"]["array_path"]).exists()


def test_audio_compare_e2e(sine_440_wav: Path, square_440_wav: Path):
    from audio_analysis_mcp.tools.audio_compare import audio_compare

    result = json.loads(
        audio_compare(
            target_path=str(sine_440_wav),
            rendered_path=str(square_440_wav),
        )
    )
    assert result["mel_spectrogram_distance"] > 0
    assert len(result["band_diffs"]) >= 3
