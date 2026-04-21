from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import pytest
from audio_analysis_mcp.tools.stem_separate import stem_separate_impl

STEMS = ["vocals", "drums", "bass", "other"]


def _make_mock_model():
    model = MagicMock()
    model.sources = ["drums", "bass", "other", "vocals"]
    model.samplerate = 44100
    model.audio_channels = 2
    return model


def _make_mock_audio_file():
    af = MagicMock()
    af.read.return_value = torch.zeros(2, 44100)
    return af


def test_stem_separate_returns_all_stems(sine_440_wav: Path, tmp_path: Path):
    model = _make_mock_model()
    # apply_model returns [batch, sources, channels, samples] — we index [0]
    sources_tensor = torch.zeros(1, 4, 2, 44100)

    with patch("audio_analysis_mcp.tools.stem_separate.get_model", return_value=model), \
         patch("audio_analysis_mcp.tools.stem_separate.apply_model", return_value=sources_tensor), \
         patch("audio_analysis_mcp.tools.stem_separate.AudioFile", return_value=_make_mock_audio_file()), \
         patch("audio_analysis_mcp.tools.stem_separate.save_audio") as mock_save:
        result = stem_separate_impl(str(sine_440_wav), tmp_path, model_name="htdemucs")

    assert len(result.stems) == 4
    assert {s.stem for s in result.stems} == set(STEMS)
    assert result.cached is False
    assert mock_save.call_count == 4


def test_stem_separate_uses_cache(sine_440_wav: Path, tmp_path: Path):
    model = _make_mock_model()
    sources_tensor = torch.zeros(1, 4, 2, 44100)

    # Track save calls to create actual files for caching
    def fake_save(tensor, path, samplerate):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake")

    with patch("audio_analysis_mcp.tools.stem_separate.get_model", return_value=model) as mock_get, \
         patch("audio_analysis_mcp.tools.stem_separate.apply_model", return_value=sources_tensor), \
         patch("audio_analysis_mcp.tools.stem_separate.AudioFile", return_value=_make_mock_audio_file()), \
         patch("audio_analysis_mcp.tools.stem_separate.save_audio", side_effect=fake_save):
        stem_separate_impl(str(sine_440_wav), tmp_path, model_name="htdemucs")
        result2 = stem_separate_impl(str(sine_440_wav), tmp_path, model_name="htdemucs")

    assert result2.cached is True
    # get_model called both times (to read source_names), but apply_model only once
    assert mock_get.call_count == 2


def test_stem_separate_failure_raises(sine_440_wav: Path, tmp_path: Path):
    model = _make_mock_model()

    with patch("audio_analysis_mcp.tools.stem_separate.get_model", return_value=model), \
         patch("audio_analysis_mcp.tools.stem_separate.apply_model", side_effect=RuntimeError("OOM")), \
         patch("audio_analysis_mcp.tools.stem_separate.AudioFile", return_value=_make_mock_audio_file()):
        with pytest.raises(RuntimeError, match="OOM"):
            stem_separate_impl(str(sine_440_wav), tmp_path, model_name="htdemucs")
