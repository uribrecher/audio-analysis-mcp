from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol, TypeAlias

from audio_analysis_mcp.schemas import StructureAnalyzeResult, StructureSegment


class _SegmentLike(Protocol):
    start: float
    end: float
    label: str


class _AnalysisResultLike(Protocol):
    segments: list[_SegmentLike]
    duration: float


# `pipeline.analyze` accepts an optional progress callback (SongFormer >=
# 0.2.0). Declaring it here so mypy validates the call.
class _Pipeline(Protocol):
    def analyze(
        self,
        audio_path: str,
        progress: "Callable[[str, float], None] | None" = None,
    ) -> _AnalysisResultLike: ...


ProgressFn: TypeAlias = Callable[[str, float, "str | None"], None]
"""Caller-side progress callback. Stages: ``cache_hit | load_audio | encode |
infer | postprocess | done``. ``detail`` is currently always ``None`` here —
the parameter exists so the type matches the stems-side ``ProgressFn``."""


def analyze_structure(
    audio_path: str,
    output_dir: str,
    pipeline: _Pipeline,
    progress: ProgressFn | None = None,
) -> StructureAnalyzeResult:
    """Run SongFormer on ``audio_path`` and cache to ``<output_dir>/structure.json``.

    Cache-hits return the parsed JSON without invoking the model.
    Optional ``progress(stage, fraction, detail)`` is forwarded into
    ``pipeline.analyze`` so callers see real-time fractions during the 30-75s run.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    structure_path = out / "structure.json"

    if structure_path.exists():
        if progress is not None:
            try:
                progress("cache_hit", 1.0, None)
            except Exception:
                pass
        data = json.loads(structure_path.read_text())
        return StructureAnalyzeResult(
            structure_path=str(structure_path),
            segments=[StructureSegment(**s) for s in data["segments"]],
            duration=float(data["duration"]),
            cached=True,
        )

    # Adapt our (stage, fraction, detail) signature to SongFormer's
    # (stage, fraction) callback. SongFormer's pipeline already clamps and
    # swallows callback exceptions, so we don't need to wrap it again.
    if progress is None:
        result = pipeline.analyze(audio_path)
    else:
        result = pipeline.analyze(
            audio_path,
            progress=lambda stage, fraction: progress(stage, fraction, None),
        )

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

    if progress is not None:
        try:
            progress("done", 1.0, None)
        except Exception:
            pass

    return StructureAnalyzeResult(
        structure_path=str(structure_path),
        segments=segments,
        duration=duration,
        cached=False,
    )
