"""Tests for analysis.transcription — Basic Pitch is mocked."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pretty_midi
import pytest

from audio_analysis_mcp.analysis.transcription import transcribe_audio
from audio_analysis_mcp.schemas import NoteEvent


def _mock_predict_result(notes: list[tuple[float, float, int, float]]):
    """Build a mock return value matching basic_pitch.inference.predict signature.

    predict returns: (model_output_dict, pretty_midi.PrettyMIDI, note_events_list)
    Each note event: (start_s, end_s, pitch_midi, velocity, pitch_bends)
    """
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
def test_single_note(mock_predict: MagicMock, sine_440_wav: Path, tmp_path: Path):
    mock_predict.return_value = _mock_predict_result([
        (0.05, 0.95, 69, 0.8),  # A4 = MIDI 69
    ])
    midi_path, notes_path, notes = transcribe_audio(str(sine_440_wav), output_dir=str(tmp_path))
    assert Path(midi_path).exists()
    assert Path(midi_path).suffix == ".mid"
    assert Path(notes_path).exists()
    assert Path(notes_path).suffix == ".json"
    assert len(notes) == 1
    assert notes[0].pitch_midi == 69
    assert notes[0].start_time == pytest.approx(0.05, abs=0.01)
    assert notes[0].end_time == pytest.approx(0.95, abs=0.01)
    assert 0.0 <= notes[0].amplitude <= 1.0


@patch("audio_analysis_mcp.analysis.transcription.predict")
def test_two_simultaneous_notes(mock_predict: MagicMock, two_note_wav: Path, tmp_path: Path):
    mock_predict.return_value = _mock_predict_result([
        (0.05, 0.95, 60, 0.7),  # C4
        (0.05, 0.95, 64, 0.7),  # E4
    ])
    midi_path, notes_path, notes = transcribe_audio(str(two_note_wav), output_dir=str(tmp_path))
    assert len(notes) == 2
    pitches = {n.pitch_midi for n in notes}
    assert pitches == {60, 64}


@patch("audio_analysis_mcp.analysis.transcription.predict")
def test_note_event_fields_populated(mock_predict: MagicMock, sine_440_wav: Path, tmp_path: Path):
    mock_predict.return_value = _mock_predict_result([
        (0.1, 0.9, 69, 0.6),
    ])
    _, _, notes = transcribe_audio(str(sine_440_wav), output_dir=str(tmp_path))
    note = notes[0]
    assert isinstance(note.start_time, float)
    assert isinstance(note.end_time, float)
    assert isinstance(note.pitch_midi, int)
    assert isinstance(note.amplitude, float)
    assert note.pitch_bends is None  # mock returns None for pitch_bends


@patch("audio_analysis_mcp.analysis.transcription.predict")
def test_empty_transcription(mock_predict: MagicMock, silence_wav: Path, tmp_path: Path):
    mock_predict.return_value = _mock_predict_result([])
    midi_path, notes_path, notes = transcribe_audio(str(silence_wav), output_dir=str(tmp_path))
    assert Path(midi_path).exists()
    assert notes == []
