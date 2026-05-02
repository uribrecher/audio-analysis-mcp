"""Probe jsonschema 4.26 RefResolver/Registry behavior with our two-file schema.

Run: uv run python scratch/explore_jsonschema_resolver.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT.parent / "reverse-synth-research" / "parameter-mapping"

with (SCHEMA_DIR / "subtractive.schema.json").open() as f:
    subtractive_schema = json.load(f)
with (SCHEMA_DIR / "synth-base.schema.json").open() as f:
    base_schema = json.load(f)

instance = {
    "schema_version": "0.1",
    "engine": "subtractive",
    "params": {
        "osc": {"1": {"shape": "saw", "level": 1.0, "octave": 0, "detune_cents": 0}},
        "filter": {
            "lp": {
                "cutoff_hz": 2000.0,
                "resonance": 0.5,
                "envelope_amount": 0.0,
                "key_tracking": 0.0,
                "drive": 0.0,
            }
        },
        "envelope": {
            "amp": {
                "attack_ms": 10.0,
                "decay_ms": 200.0,
                "sustain": 0.7,
                "release_ms": 200.0,
            }
        },
        "voice": {"mode": "poly"},
        "lfo": {
            "1": {
                "rate_hz": 1.0,
                "shape": "sine",
                "depth": 0.0,
                "target": "filter.cutoff",
            }
        },
    },
}

print("=== Approach 1: legacy RefResolver ===")
store = {
    subtractive_schema["$id"]: subtractive_schema,
    base_schema["$id"]: base_schema,
    "synth-base.schema.json": base_schema,
}
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    resolver = jsonschema.RefResolver.from_schema(subtractive_schema, store=store)
    cls = jsonschema.validators.validator_for(subtractive_schema)
    cls.check_schema(subtractive_schema)
    validator = cls(subtractive_schema, resolver=resolver)
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        validator.validate(instance)
    print("RefResolver OK")
except Exception as e:
    print(f"RefResolver FAILED: {type(e).__name__}: {e}")


print("\n=== Approach 2: referencing.Registry ===")
try:
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    base_resource = Resource(contents=base_schema, specification=DRAFT202012)
    sub_resource = Resource(contents=subtractive_schema, specification=DRAFT202012)
    registry = Registry().with_resources(
        [
            (base_schema["$id"], base_resource),
            (subtractive_schema["$id"], sub_resource),
            ("synth-base.schema.json", base_resource),
        ]
    )
    cls = jsonschema.validators.validator_for(subtractive_schema)
    cls.check_schema(subtractive_schema)
    validator = cls(subtractive_schema, registry=registry)
    validator.validate(instance)
    print("Registry OK")
except Exception as e:
    print(f"Registry FAILED: {type(e).__name__}: {e}")


print("\n=== Approach 3: minimal failing case for #/$defs ===")
# Try to validate using just subtractive_schema with a manual resolver against the in-memory dict
inline_store = {
    "synth-base.schema.json": base_schema,
}
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    resolver = jsonschema.RefResolver(
        base_uri="",
        referrer=subtractive_schema,
        store=inline_store,
    )
    cls = jsonschema.validators.validator_for(subtractive_schema)
    validator = cls(subtractive_schema, resolver=resolver)
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        validator.validate(instance)
    print("base_uri='' OK")
except Exception as e:
    print(f"base_uri='' FAILED: {type(e).__name__}: {e}")
