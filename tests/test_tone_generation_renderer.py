import numpy as np
import pytest

from audio_analysis_mcp.research.tone_generation.renderer import render_chord
from audio_analysis_mcp.research.tone_generation.schema_io import (
    build_canonical_instance,
)


SR = 44100
TOTAL_DURATION_S = 1.2  # 100 ms attack + 1000 ms hold + 100 ms release


def _params(shape: str = "saw", cutoff_hz: float = 2000.0, resonance: float = 0.5):
    return build_canonical_instance(
        shape=shape, cutoff_hz=cutoff_hz, resonance=resonance
    )


def test_render_chord_single_voice_shape_and_dtype():
    audio = render_chord(
        params_canonical=_params(),
        midi_pitches=[69],  # A4
        sample_rate=SR,
        total_duration_s=TOTAL_DURATION_S,
    )
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    expected_len = int(SR * TOTAL_DURATION_S)
    assert audio.size == expected_len
    assert not np.any(np.isnan(audio))
    assert not np.any(np.isinf(audio))


def test_render_chord_normalized_peak():
    audio = render_chord(
        params_canonical=_params(),
        midi_pitches=[69],
        sample_rate=SR,
        total_duration_s=TOTAL_DURATION_S,
    )
    peak = np.abs(audio).max()
    assert 0.85 <= peak <= 1.0, f"peak {peak} out of [0.85, 1.0]"


@pytest.mark.parametrize("n_voices", [1, 2, 3])
def test_render_chord_polyphony(n_voices: int):
    pitches = [60, 64, 67][:n_voices]
    audio = render_chord(
        params_canonical=_params(),
        midi_pitches=pitches,
        sample_rate=SR,
        total_duration_s=TOTAL_DURATION_S,
    )
    assert audio.size == int(SR * TOTAL_DURATION_S)
    assert not np.any(np.isnan(audio))
    assert np.abs(audio).max() <= 1.0


def test_render_chord_different_shapes_produce_different_audio():
    a = render_chord(
        params_canonical=_params(shape="sine"),
        midi_pitches=[69],
        sample_rate=SR,
        total_duration_s=TOTAL_DURATION_S,
    )
    b = render_chord(
        params_canonical=_params(shape="saw"),
        midi_pitches=[69],
        sample_rate=SR,
        total_duration_s=TOTAL_DURATION_S,
    )
    assert not np.allclose(a, b, atol=0.05)


def test_render_chord_rejects_empty_pitches():
    with pytest.raises(ValueError):
        render_chord(
            params_canonical=_params(),
            midi_pitches=[],
            sample_rate=SR,
            total_duration_s=TOTAL_DURATION_S,
        )
