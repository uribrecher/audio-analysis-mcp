from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import pytest
from audio_analysis_mcp.tools.stem_separate import stem_separate_impl

SOURCES = ["drums", "bass", "other", "vocals"]


def _make_mock_model():
    model = MagicMock()
    model.sources = SOURCES
    model.samplerate = 44100
    model.audio_channels = 2
    return model


def _make_mock_audio_file():
    af = MagicMock()
    af.read.return_value = torch.zeros(2, 44100)
    return af


def _fake_save(tensor, path, samplerate):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("fake")


def test_stem_separate_returns_all_stems(sine_440_wav: Path, tmp_path: Path):
    model = _make_mock_model()
    sources_tensor = torch.zeros(1, 4, 2, 44100)

    with patch("audio_analysis_mcp.tools.stem_separate.get_model", return_value=model), \
         patch("audio_analysis_mcp.tools.stem_separate.apply_model", return_value=sources_tensor), \
         patch("audio_analysis_mcp.tools.stem_separate.AudioFile", return_value=_make_mock_audio_file()), \
         patch("audio_analysis_mcp.tools.stem_separate.save_audio", side_effect=_fake_save):
        result = stem_separate_impl(str(sine_440_wav), tmp_path, model_name="htdemucs")

    assert len(result.stems) == 4
    assert {s.stem for s in result.stems} == set(SOURCES)
    assert result.cached is False


def test_stem_separate_uses_cache(sine_440_wav: Path, tmp_path: Path):
    model = _make_mock_model()
    sources_tensor = torch.zeros(1, 4, 2, 44100)

    with patch("audio_analysis_mcp.tools.stem_separate.get_model", return_value=model) as mock_get, \
         patch("audio_analysis_mcp.tools.stem_separate.apply_model", return_value=sources_tensor) as mock_apply, \
         patch("audio_analysis_mcp.tools.stem_separate.AudioFile", return_value=_make_mock_audio_file()), \
         patch("audio_analysis_mcp.tools.stem_separate.save_audio", side_effect=_fake_save):
        stem_separate_impl(str(sine_440_wav), tmp_path, model_name="htdemucs")
        result2 = stem_separate_impl(str(sine_440_wav), tmp_path, model_name="htdemucs")

    assert result2.cached is True
    # Model loaded only on first call; cache hit avoids get_model on second call
    assert mock_get.call_count == 1
    assert mock_apply.call_count == 1


def test_stem_separate_failure_raises(sine_440_wav: Path, tmp_path: Path):
    model = _make_mock_model()

    with patch("audio_analysis_mcp.tools.stem_separate.get_model", return_value=model), \
         patch("audio_analysis_mcp.tools.stem_separate.apply_model", side_effect=RuntimeError("OOM")), \
         patch("audio_analysis_mcp.tools.stem_separate.AudioFile", return_value=_make_mock_audio_file()):
        with pytest.raises(RuntimeError, match="OOM"):
            stem_separate_impl(str(sine_440_wav), tmp_path, model_name="htdemucs")


def test_stem_separate_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not found"):
        stem_separate_impl("/nonexistent/file.wav", tmp_path, model_name="htdemucs")
