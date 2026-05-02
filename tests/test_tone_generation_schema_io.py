import math

import pytest

from audio_analysis_mcp.research.tone_generation.schema_io import (
    BASELINE_AMP_ADSR,
    SHAPE_LABELS,
    build_canonical_instance,
    denormalize_predictions,
    normalize_params,
    validate_canonical,
)


def test_baseline_amp_adsr_is_valid_adsr_dict():
    keys = {"attack_ms", "decay_ms", "sustain", "release_ms"}
    assert set(BASELINE_AMP_ADSR.keys()) == keys
    assert BASELINE_AMP_ADSR["attack_ms"] >= 0
    assert 0.0 <= BASELINE_AMP_ADSR["sustain"] <= 1.0


def test_shape_labels_match_schema():
    assert SHAPE_LABELS == ["sine", "saw", "square", "triangle"]


def test_build_canonical_instance_shape():
    inst = build_canonical_instance(shape="saw", cutoff_hz=2000.0, resonance=0.5)
    assert inst["schema_version"] == "0.1"
    assert inst["engine"] == "subtractive"
    p = inst["params"]
    assert p["osc"]["1"]["shape"] == "saw"
    assert p["osc"]["1"]["level"] == 1.0
    assert p["filter"]["lp"]["cutoff_hz"] == 2000.0
    assert p["filter"]["lp"]["resonance"] == 0.5
    assert p["filter"]["lp"]["envelope_amount"] == 0.0
    assert p["voice"]["mode"] == "poly"
    assert p["envelope"]["amp"] == BASELINE_AMP_ADSR


def test_build_canonical_instance_validates():
    inst = build_canonical_instance(shape="square", cutoff_hz=440.0, resonance=0.0)
    validate_canonical(inst)  # raises on invalid


@pytest.mark.parametrize("shape", SHAPE_LABELS)
@pytest.mark.parametrize("cutoff_hz", [50.0, 100.0, 1000.0, 5000.0, 10000.0])
@pytest.mark.parametrize("resonance", [0.0, 0.5, 1.0])
def test_build_canonical_instance_parametric(shape, cutoff_hz, resonance):
    inst = build_canonical_instance(shape=shape, cutoff_hz=cutoff_hz, resonance=resonance)
    validate_canonical(inst)


def test_validate_canonical_rejects_invalid():
    bad = {"schema_version": "0.1", "engine": "subtractive", "params": {}}
    with pytest.raises(Exception):
        validate_canonical(bad)


def test_normalize_denormalize_roundtrip_cutoff():
    cutoff_hz = 2000.0
    norm = math.log(cutoff_hz / 50.0) / math.log(10000.0 / 50.0)
    out = normalize_params(shape="saw", cutoff_hz=cutoff_hz, resonance=0.5)
    assert math.isclose(out["cutoff_norm"], norm, rel_tol=1e-6)
    assert out["resonance"] == 0.5
    assert out["shape_label"] == 1  # saw == 1


def test_denormalize_returns_canonical_instance():
    inst = denormalize_predictions(
        shape_label=1, cutoff_norm=0.5, resonance=0.42, midi_pitches=[60, 64, 67]
    )
    validate_canonical(inst)
    assert inst["params"]["osc"]["1"]["shape"] == "saw"
    expected_cutoff = math.exp(
        math.log(50.0) + 0.5 * (math.log(10000.0) - math.log(50.0))
    )
    assert math.isclose(
        inst["params"]["filter"]["lp"]["cutoff_hz"], expected_cutoff, rel_tol=1e-4
    )


def test_normalize_then_denormalize_roundtrip_continuous():
    out = normalize_params(shape="square", cutoff_hz=2000.0, resonance=0.6)
    inst = denormalize_predictions(
        shape_label=out["shape_label"],
        cutoff_norm=out["cutoff_norm"],
        resonance=out["resonance"],
        midi_pitches=[60],
    )
    p = inst["params"]
    assert p["osc"]["1"]["shape"] == "square"
    assert math.isclose(p["filter"]["lp"]["cutoff_hz"], 2000.0, rel_tol=1e-4)
    assert math.isclose(p["filter"]["lp"]["resonance"], 0.6, abs_tol=1e-9)
