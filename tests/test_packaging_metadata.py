# tests/test_packaging_metadata.py
import re
import subprocess
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = ("songformer", "signalflow", "fastapi", "uvicorn", "sse-starlette", "sse_starlette")


@pytest.fixture(scope="module")
def wheel_metadata(tmp_path_factory: pytest.TempPathFactory) -> str:
    out = tmp_path_factory.mktemp("wheelbuild")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out)],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    )
    wheel = next(out.glob("*.whl"))
    with zipfile.ZipFile(wheel) as zf:
        meta_name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        return zf.read(meta_name).decode()


def test_core_requires_have_no_forbidden_or_direct_refs(wheel_metadata: str) -> None:
    msg = Parser().parsestr(wheel_metadata)
    requires = msg.get_all("Requires-Dist") or []
    core = [r for r in requires if "extra ==" not in r]  # core deps only
    for r in core:
        assert "@" not in r, f"direct reference leaked into core deps: {r}"
        assert not any(f in r.lower() for f in FORBIDDEN), f"forbidden core dep: {r}"


def test_service_extra_present(wheel_metadata: str) -> None:
    msg = Parser().parsestr(wheel_metadata)
    assert "service" in (msg.get_all("Provides-Extra") or [])
    requires = msg.get_all("Requires-Dist") or []
    service = [r for r in requires if "extra ==" in r and "service" in r]
    assert any("fastapi" in r for r in service)
    assert any("uvicorn" in r for r in service)


def test_core_keeps_essential_deps(wheel_metadata: str) -> None:
    msg = Parser().parsestr(wheel_metadata)
    # Parse the package NAME (leading token) from each core Requires-Dist line and
    # assert exact membership — a substring check would let "torch" match
    # "torchaudio"/"torchcodec" even if torch itself were dropped.
    names = set()
    for r in msg.get_all("Requires-Dist") or []:
        if "extra ==" in r:
            continue
        m = re.match(r"[A-Za-z0-9._-]+", r)
        if m:
            names.add(m.group(0).lower())
    for dep in ("mcp", "torch", "demucs", "librosa", "basic-pitch", "sounddevice"):
        assert dep in names, f"core dep missing: {dep}"
