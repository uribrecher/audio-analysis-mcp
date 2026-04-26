"""E2E tests: call MCP tool functions directly and verify JSON output."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pretty_midi
import pytest

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
import audio_analysis_mcp.tools.amplitude_analyze  # noqa: F401


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

    ws = srv.get_workspace()
    stem_file = ws.job_stem_file("test-song", "fast", "bass")
    shutil.copy(sine_440_wav, stem_file)

    mock_predict.return_value = _mock_predict_result([
        (0.05, 0.95, 69, 0.8),
    ])
    result = json.loads(note_transcribe(audio_path=str(stem_file)))
    assert Path(result["midi_path"]).exists()
    assert Path(result["notes_path"]).exists()
    assert result["note_count"] == 1
    assert "test-song/transcriptions/bass_fast" in result["midi_path"]


def test_note_triage_returns_clusters(sine_440_wav: Path):
    from audio_analysis_mcp.tools.note_triage import note_triage
    from audio_analysis_mcp.schemas import NoteEvent

    ws = srv.get_workspace()
    stem_file = ws.job_stem_file("test-song", "fast", "bass")
    shutil.copy(sine_440_wav, stem_file)

    notes = [
        NoteEvent(start_time=0.0, end_time=0.3, pitch_midi=60, amplitude=0.5, pitch_bends=None),
        NoteEvent(start_time=2.0, end_time=4.0, pitch_midi=64, amplitude=0.9, pitch_bends=None),
        NoteEvent(start_time=6.0, end_time=7.0, pitch_midi=67, amplitude=0.6, pitch_bends=None),
        NoteEvent(start_time=6.0, end_time=7.0, pitch_midi=71, amplitude=0.6, pitch_bends=None),
    ]
    notes_path = ws.job_notes_file("test-song")
    notes_path.write_text(json.dumps([n.model_dump() for n in notes]))

    result = json.loads(note_triage(audio_path=str(stem_file), notes_path=str(notes_path)))
    assert result["candidate_count"] >= 1

    top = result["top_candidates"][0]
    # Cluster shape:
    assert top["kind"] in ("single", "chord")
    assert "score" in top
    assert "members" in top and len(top["members"]) >= 1
    # The long high-velocity single (pitch 64) should rank highly:
    pitches = [m["note"]["pitch_midi"] for c in result["top_candidates"] for m in c["members"]]
    assert 64 in pitches


def test_note_triage_respects_time_window(sine_440_wav: Path):
    from audio_analysis_mcp.tools.note_triage import note_triage
    from audio_analysis_mcp.schemas import NoteEvent

    ws = srv.get_workspace()
    stem_file = ws.job_stem_file("test-song", "fast", "bass")
    shutil.copy(sine_440_wav, stem_file)

    notes = [
        NoteEvent(start_time=0.0, end_time=1.0, pitch_midi=60, amplitude=0.8, pitch_bends=None),
        NoteEvent(start_time=5.0, end_time=6.0, pitch_midi=64, amplitude=0.8, pitch_bends=None),
        NoteEvent(start_time=10.0, end_time=11.0, pitch_midi=67, amplitude=0.8, pitch_bends=None),
    ]
    notes_path = ws.job_notes_file("test-song")
    notes_path.write_text(json.dumps([n.model_dump() for n in notes]))

    result = json.loads(note_triage(
        audio_path=str(stem_file), notes_path=str(notes_path),
        start_time=4.0, end_time=8.0,
    ))
    assert result["candidate_count"] == 1
    assert result["top_candidates"][0]["members"][0]["note"]["pitch_midi"] == 64


def test_note_isolate_e2e(sine_440_wav: Path):
    from audio_analysis_mcp.tools.note_isolate import note_isolate

    ws = srv.get_workspace()
    stem_file = ws.job_stem_file("test-song", "fast", "bass")
    shutil.copy(sine_440_wav, stem_file)

    result = json.loads(
        note_isolate(
            audio_path=str(stem_file),
            start_time=0.0,
            end_time=0.5,
            start_freq=400.0,
            end_freq=500.0,
            pitch_midi=69,
        )
    )
    assert Path(result["audio_path"]).exists()
    assert result["duration_seconds"] > 0
    assert "note_001_A4_0.0s.wav" in result["audio_path"]
    assert "test-song/isolated_notes/bass_fast" in result["audio_path"]


def test_amplitude_analyze_e2e(sine_440_wav: Path):
    """End-to-end: triage → amplitude_analyze produces per-cluster outputs."""
    from audio_analysis_mcp.tools.amplitude_analyze import amplitude_analyze
    from audio_analysis_mcp.tools.note_triage import note_triage
    from audio_analysis_mcp.schemas import NoteEvent

    ws = srv.get_workspace()
    stem_file = ws.job_stem_file("test-song", "fast", "bass")
    shutil.copy(sine_440_wav, stem_file)

    notes = [
        NoteEvent(start_time=0.05, end_time=0.95, pitch_midi=69, amplitude=0.8, pitch_bends=None),
    ]
    notes_path = ws.job_notes_file("test-song")
    notes_path.write_text(json.dumps([n.model_dump() for n in notes]))

    triage_json = json.loads(note_triage(audio_path=str(stem_file), notes_path=str(notes_path), min_duration=0.0))
    triage_path = triage_json["triage_path"]

    result_json = amplitude_analyze(audio_path=str(stem_file), triage_path=triage_path)
    payload = json.loads(result_json)
    # Single 1-second sine note → triage produces ≥1 cluster → orchestrator analyzes it.
    # The sine wave has no real ADSR shape, so sustain may or may not be present; either way
    # the tool must return a valid result structure.
    assert "candidates" in payload
    assert "is_consistent" in payload
    assert "divergence_score" in payload
