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
