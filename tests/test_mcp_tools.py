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
import audio_analysis_mcp.tools.note_transcribe  # noqa: F401
import audio_analysis_mcp.tools.note_triage  # noqa: F401
import audio_analysis_mcp.tools.note_isolate  # noqa: F401

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
    assert result["job_name"] == "sine-440"
    assert Path(result["audio_path"]).exists()
    assert "jobs/sine-440/source.wav" in result["audio_path"]


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


from unittest.mock import patch, MagicMock
import numpy as np
import pretty_midi


def _mock_predict_result(notes: list[tuple[float, float, int, float]]):
    model_output = {"note": np.zeros((1, 1)), "onset": np.zeros((1, 1))}
    midi_data = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for start, end, pitch, vel in notes:
        note = pretty_midi.Note(
            velocity=int(vel * 127), pitch=pitch, start=start, end=end,
        )
        inst.notes.append(note)
    midi_data.instruments.append(inst)
    note_events = [(s, e, p, v, None) for s, e, p, v in notes]
    return model_output, midi_data, note_events


@patch("audio_analysis_mcp.analysis.transcription.predict")
def test_note_transcribe_e2e(mock_predict: MagicMock, sine_440_wav: Path):
    from audio_analysis_mcp.tools.note_transcribe import note_transcribe

    mock_predict.return_value = _mock_predict_result([
        (0.05, 0.95, 69, 0.8),
    ])
    result = json.loads(note_transcribe(audio_path=str(sine_440_wav)))
    assert Path(result["midi_path"]).exists()
    assert Path(result["notes_path"]).exists()
    assert result["note_count"] == 1


def test_note_isolate_e2e(sine_440_wav: Path):
    from audio_analysis_mcp.tools.note_isolate import note_isolate

    result = json.loads(
        note_isolate(
            audio_path=str(sine_440_wav),
            start_time=0.0,
            end_time=0.5,
            start_freq=400.0,
            end_freq=500.0,
        )
    )
    assert Path(result["audio_path"]).exists()
    assert result["duration_seconds"] > 0
