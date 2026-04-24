import pytest
from pathlib import Path

from audio_analysis_mcp.workspace import Workspace, sanitize_job_name, resolve_job_context


def test_workspace_creates_subdirs(tmp_path):
    ws = Workspace(root=tmp_path / "ws")
    assert ws.imported.exists()
    assert ws.stems.exists()
    assert ws.spectrograms.exists()
    assert ws.rendered.exists()


def test_workspace_is_idempotent(tmp_path):
    root = tmp_path / "ws"
    ws1 = Workspace(root=root)
    _ = ws1.imported  # trigger creation
    ws2 = Workspace(root=root)
    assert ws2.imported.exists()


def test_sanitize_basic():
    assert sanitize_job_name("My Song.wav") == "my-song"


def test_sanitize_parentheses():
    assert sanitize_job_name("Smooth Criminal (Radio Edit).wav") == "smooth-criminal-radio-edit"


def test_sanitize_special_chars():
    assert sanitize_job_name("song #1 [feat. Artist] & More!.mp3") == "song-1-feat-artist-more"


def test_sanitize_multiple_spaces():
    assert sanitize_job_name("  too   many   spaces  .wav") == "too-many-spaces"


def test_sanitize_unicode():
    assert sanitize_job_name("café naïve.wav") == "caf-na-ve"


def test_sanitize_dots_and_underscores():
    assert sanitize_job_name("my_song.v2.final.wav") == "my-song-v2-final"


def test_resolve_source(tmp_path: Path):
    ws = Workspace(tmp_path)
    source = tmp_path / "jobs" / "my-song" / "source.wav"
    source.parent.mkdir(parents=True)
    source.touch()
    ctx = resolve_job_context(str(source), ws)
    assert ctx.job_name == "my-song"
    assert ctx.stem is None
    assert ctx.preset is None


def test_resolve_stem(tmp_path: Path):
    ws = Workspace(tmp_path)
    stem = tmp_path / "jobs" / "my-song" / "stems" / "medium" / "bass.wav"
    stem.parent.mkdir(parents=True)
    stem.touch()
    ctx = resolve_job_context(str(stem), ws)
    assert ctx.job_name == "my-song"
    assert ctx.stem == "bass"
    assert ctx.preset == "medium"


def test_resolve_outside_workspace(tmp_path: Path):
    ws = Workspace(tmp_path)
    with pytest.raises(ValueError, match="not inside"):
        resolve_job_context("/some/other/path.wav", ws)
