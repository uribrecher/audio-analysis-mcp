from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from audio_analysis_mcp.schemas import StructureAnalyzeResult, StructureSegment


class _SegmentLike(Protocol):
    start: float
    end: float
    label: str


class _AnalysisResultLike(Protocol):
    segments: list[_SegmentLike]
    duration: float


class _Pipeline(Protocol):
    def analyze(self, audio_path: str) -> _AnalysisResultLike: ...


def analyze_structure(
    audio_path: str,
    output_dir: str,
    pipeline: _Pipeline,
) -> StructureAnalyzeResult:
    """Run SongFormer on ``audio_path`` and cache results to ``<output_dir>/structure.json``.

    Cache-hits return the parsed JSON without invoking the model.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    structure_path = out / "structure.json"

    if structure_path.exists():
        data = json.loads(structure_path.read_text())
        return StructureAnalyzeResult(
            structure_path=str(structure_path),
            segments=[StructureSegment(**s) for s in data["segments"]],
            duration=float(data["duration"]),
            cached=True,
        )

    result = pipeline.analyze(audio_path)
    segments = [
        StructureSegment(
            start=float(seg.start),
            end=float(seg.end),
            label=str(seg.label),
        )
        for seg in result.segments
    ]
    duration = float(result.duration)

    structure_path.write_text(
        json.dumps(
            {
                "segments": [s.model_dump() for s in segments],
                "duration": duration,
            },
            indent=2,
        )
    )

    return StructureAnalyzeResult(
        structure_path=str(structure_path),
        segments=segments,
        duration=duration,
        cached=False,
    )
