from audio_analysis_mcp.workspace import Workspace


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
