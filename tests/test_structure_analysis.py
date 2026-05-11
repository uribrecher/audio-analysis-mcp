import json
from dataclasses import dataclass
from pathlib import Path

from audio_analysis_mcp.analysis.structure_analysis import analyze_structure


@dataclass
class _FakeSeg:
    start: float
    end: float
    label: str


@dataclass
class _FakeResult:
    segments: list[_FakeSeg]
    duration: float


class _FakePipeline:
    def __init__(self, result: _FakeResult) -> None:
        self.result = result
        self.calls = 0

    def analyze(self, audio_path: str) -> _FakeResult:
        self.calls += 1
        return self.result


def _result() -> _FakeResult:
    return _FakeResult(
        segments=[
            _FakeSeg(0.0, 12.0, "intro"),
            _FakeSeg(12.0, 60.0, "verse"),
            _FakeSeg(60.0, 90.0, "chorus"),
        ],
        duration=180.0,
    )


def test_first_call_writes_structure_json(tmp_path: Path) -> None:
    pipeline = _FakePipeline(_result())
    out = tmp_path / "structure"

    res = analyze_structure(
        audio_path="dummy.wav", output_dir=str(out), pipeline=pipeline
    )

    assert pipeline.calls == 1
    assert res.cached is False
    assert res.duration == 180.0
    assert len(res.segments) == 3
    assert res.segments[1].label == "verse"

    path = Path(res.structure_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["duration"] == 180.0
    assert data["segments"][0]["label"] == "intro"


def test_second_call_hits_cache(tmp_path: Path) -> None:
    pipeline = _FakePipeline(_result())
    out = tmp_path / "structure"
    analyze_structure("dummy.wav", str(out), pipeline)

    pipeline.result = _FakeResult(segments=[_FakeSeg(0.0, 1.0, "different")], duration=99.0)
    res = analyze_structure("dummy.wav", str(out), pipeline)

    assert pipeline.calls == 1, "pipeline.analyze should not be called on cache hit"
    assert res.cached is True
    assert res.duration == 180.0
    assert res.segments[1].label == "verse"
