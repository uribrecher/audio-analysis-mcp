"""Canonical-schema I/O for subtractive tone-generation MVP.

The schema source of truth is `subtractive.schema.json` in the sibling
`reverse-synth-research/parameter-mapping/` directory. We resolve the path
once at import; an env var override (TONE_GEN_SCHEMA_DIR) handles non-monorepo
layouts.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema
from referencing import Registry
from referencing.jsonschema import DRAFT202012

# ---- Schema-locked constants ------------------------------------------------

SHAPE_LABELS: list[str] = ["sine", "saw", "square", "triangle"]
"""Free shapes the MVP predicts. `pulse` deliberately excluded — see backlog."""

CUTOFF_HZ_MIN = 50.0
CUTOFF_HZ_MAX = 10_000.0
"""Free-cutoff sampling bounds. Schema's hard bounds are 20 / 20_000."""

BASELINE_AMP_ADSR: dict[str, float] = {
    "attack_ms": 10.0,
    "decay_ms": 200.0,
    "sustain": 0.7,
    "release_ms": 200.0,
}
"""Frozen amp ADSR for MVP. Pad-style preset; identical across all renders."""

# ---- Schema loading ---------------------------------------------------------

_DEFAULT_SCHEMA_DIR = (
    Path(__file__).resolve().parents[5]
    / "reverse-synth-research"
    / "parameter-mapping"
)


def _schema_dir() -> Path:
    override = os.environ.get("TONE_GEN_SCHEMA_DIR")
    if override:
        return Path(override)
    return _DEFAULT_SCHEMA_DIR


@lru_cache(maxsize=1)
def _load_validator() -> Any:
    """Load and cache a jsonschema validator for subtractive.schema.json.

    Decision: use the modern `referencing.Registry` API. The legacy
    `RefResolver` is deprecated since jsonschema 4.18 and (verified in
    scratch/explore_jsonschema_resolver.py) actually mis-resolves
    `#/$defs/...` pointers in jsonschema 4.26 once an external `$ref` to
    `synth-base.schema.json` has shifted resolution scope.
    """
    schema_dir = _schema_dir()
    subtractive_path = schema_dir / "subtractive.schema.json"
    base_path = schema_dir / "synth-base.schema.json"
    if not subtractive_path.exists():
        raise FileNotFoundError(
            f"subtractive.schema.json not found at {subtractive_path} — "
            "set TONE_GEN_SCHEMA_DIR if your monorepo layout differs."
        )
    if not base_path.exists():
        raise FileNotFoundError(
            f"synth-base.schema.json not found at {base_path} — "
            "set TONE_GEN_SCHEMA_DIR if your monorepo layout differs."
        )
    with subtractive_path.open() as f:
        subtractive_schema = json.load(f)
    with base_path.open() as f:
        base_schema = json.load(f)
    base_resource = DRAFT202012.create_resource(base_schema)
    sub_resource = DRAFT202012.create_resource(subtractive_schema)
    # Register both under their $id and under the bare filename used by the
    # subtractive schema's `$ref: "synth-base.schema.json"`.
    registry = Registry().with_resources(
        [
            (base_schema["$id"], base_resource),
            (subtractive_schema["$id"], sub_resource),
            ("synth-base.schema.json", base_resource),
        ]
    )
    cls = jsonschema.validators.validator_for(subtractive_schema)
    cls.check_schema(subtractive_schema)
    return cls(subtractive_schema, registry=registry)


def validate_canonical(instance: dict[str, Any]) -> None:
    """Validate instance against subtractive.schema.json. Raises on invalid."""
    _load_validator().validate(instance)


# ---- Canonical instance construction ---------------------------------------


def build_canonical_instance(
    *, shape: str, cutoff_hz: float, resonance: float
) -> dict[str, Any]:
    """Build a schema-conformant subtractive instance from the 3 free MVP params."""
    if shape not in SHAPE_LABELS:
        raise ValueError(f"shape must be one of {SHAPE_LABELS}, got {shape!r}")
    return {
        "schema_version": "0.1",
        "engine": "subtractive",
        "params": {
            "osc": {
                "1": {
                    "shape": shape,
                    "level": 1.0,
                    "octave": 0,
                    "detune_cents": 0,
                }
            },
            "filter": {
                "lp": {
                    "cutoff_hz": float(cutoff_hz),
                    "resonance": float(resonance),
                    "envelope_amount": 0.0,
                    "key_tracking": 0.0,
                    "drive": 0.0,
                }
            },
            "envelope": {"amp": dict(BASELINE_AMP_ADSR)},
            "voice": {"mode": "poly"},
            # Schema makes lfo required for subtractive engines, so we MUST
            # supply one. MVP renders without modulation; lfo.depth = 0 makes
            # it acoustically inert. Modulation expert will own this slot for
            # real predictions later.
            "lfo": _inert_lfo(),
        },
    }


def _inert_lfo() -> dict[str, Any]:
    """LFO with depth=0 — schema-required slot, acoustically silent."""
    return {
        "1": {
            "rate_hz": 1.0,
            "shape": "sine",
            "depth": 0.0,
            "target": "filter.cutoff",
        }
    }


# ---- Normalize / denormalize ------------------------------------------------


def _cutoff_to_norm(cutoff_hz: float) -> float:
    return (math.log(cutoff_hz) - math.log(CUTOFF_HZ_MIN)) / (
        math.log(CUTOFF_HZ_MAX) - math.log(CUTOFF_HZ_MIN)
    )


def _norm_to_cutoff(cutoff_norm: float) -> float:
    log_lo = math.log(CUTOFF_HZ_MIN)
    log_hi = math.log(CUTOFF_HZ_MAX)
    return math.exp(log_lo + cutoff_norm * (log_hi - log_lo))


def normalize_params(
    *, shape: str, cutoff_hz: float, resonance: float
) -> dict[str, float | int]:
    """Canonical params → training-time normalized scalars."""
    return {
        "shape_label": SHAPE_LABELS.index(shape),
        "cutoff_norm": _cutoff_to_norm(cutoff_hz),
        "resonance": float(resonance),
    }


def denormalize_predictions(
    *,
    shape_label: int,
    cutoff_norm: float,
    resonance: float,
    midi_pitches: list[int],
) -> dict[str, Any]:
    """Predictions → schema-conformant canonical instance.

    `midi_pitches` is accepted for forward-compatibility with the renderer
    contract but does not affect the canonical instance — pitch lives in the
    rendering driver, not in the canonical params.
    """
    cutoff_norm = max(0.0, min(1.0, cutoff_norm))
    resonance = max(0.0, min(1.0, resonance))
    shape = SHAPE_LABELS[shape_label]
    cutoff_hz = _norm_to_cutoff(cutoff_norm)
    return build_canonical_instance(
        shape=shape, cutoff_hz=cutoff_hz, resonance=resonance
    )
