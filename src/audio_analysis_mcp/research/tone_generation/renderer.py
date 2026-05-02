"""SignalFlow-based subtractive renderer for the tone-generation training MVP.

Single voice topology: osc -> LP filter -> fixed amp ADSR -> output.
Polyphonic chord = sum of N parallel voices in one AudioGraph, normalized.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import signalflow as sf_lib

_SHAPE_TO_OSC_CLASS: dict[str, str] = {
    "sine": "SineOscillator",
    "saw": "SawOscillator",
    "square": "SquareOscillator",
    "triangle": "TriangleOscillator",
}


def _midi_to_hz(midi: int) -> float:
    return float(440.0 * (2.0 ** ((midi - 69) / 12.0)))


def _build_voice(
    *,
    shape: str,
    cutoff_hz: float,
    resonance: float,
    midi_pitch: int,
    amp_adsr: dict[str, float],
) -> Any:
    """Build a single voice: osc -> LP -> amp env. Returns an unrooted SignalFlow node."""
    osc_class_name = _SHAPE_TO_OSC_CLASS[shape]
    if not hasattr(sf_lib, osc_class_name):
        raise RuntimeError(
            f"signalflow has no class {osc_class_name}; "
            "scratch/explore_subtractive_renderer.py needs re-running"
        )
    osc_class = getattr(sf_lib, osc_class_name)
    osc = osc_class(frequency=_midi_to_hz(midi_pitch))
    if not hasattr(sf_lib, "SVFilter"):
        raise RuntimeError(
            "signalflow has no SVFilter; "
            "scratch/explore_subtractive_renderer.py needs re-running"
        )
    # SignalFlow's SVFilter ctor: SVFilter(input, filter_type, cutoff, resonance).
    filtered = sf_lib.SVFilter(osc, "low_pass", cutoff_hz, resonance)
    # gate=1 is REQUIRED on every ADSREnvelope: SignalFlow 0.5.3's default is
    # gate=0, which produces silence. Verified empirically by
    # scratch/explore_subtractive_renderer.py test case 3.
    env = sf_lib.ADSREnvelope(
        attack=amp_adsr["attack_ms"] / 1000.0,
        decay=amp_adsr["decay_ms"] / 1000.0,
        sustain=amp_adsr["sustain"],
        release=amp_adsr["release_ms"] / 1000.0,
        gate=1,
    )
    return filtered * env


def render_chord(
    *,
    params_canonical: dict[str, Any],
    midi_pitches: list[int],
    sample_rate: int,
    total_duration_s: float,
) -> npt.NDArray[np.float32]:
    """Render N voices in parallel, summed, peak-normalized to ~0.95.

    Parameters
    ----------
    params_canonical : schema-conformant subtractive instance.
    midi_pitches : one MIDI pitch per voice. Length determines polyphony.
    sample_rate : in Hz.
    total_duration_s : total render length (covers attack + hold + release).

    Notes
    -----
    SignalFlow's ``AudioGraph`` is a process-wide singleton: constructing a
    second graph without destroying the first emits a warning and silently
    reuses the original. We make every call self-contained by destroying the
    graph at the end (verified empirically via
    ``scratch/explore_subtractive_renderer.py``).
    """
    if not midi_pitches:
        raise ValueError("midi_pitches must be non-empty")
    p = params_canonical["params"]
    shape = p["osc"]["1"]["shape"]
    cutoff_hz = float(p["filter"]["lp"]["cutoff_hz"])
    resonance = float(p["filter"]["lp"]["resonance"])
    amp_adsr = p["envelope"]["amp"]

    graph: Any = None
    try:
        # backend_name="null" forces miniaudio to use its NULL playback device,
        # bypassing CoreAudio / ALSA / PulseAudio device enumeration. This is
        # what makes the renderer work on headless CI runners (ubuntu-latest)
        # that have no real audio backend installed — passing output_device=None
        # alone is not enough; the graph still probes the host's default audio
        # backend and segfaults inside render_to_new_buffer when that backend
        # cannot enumerate any device. Verified empirically by
        # scratch/explore_signalflow_audio_backend.py.
        #
        # cfg.sample_rate propagates into miniaudio so the graph actually
        # renders at the requested rate (otherwise oscillator frequencies
        # would be interpreted at SignalFlow's default 44_100 Hz and the
        # buffer length we pass via num_frames would be inconsistent with
        # actual wall-clock duration). Verified empirically by
        # scratch/explore_signalflow_sample_rate.py: 1s of 1kHz sine
        # rendered at 22_050 / 44_100 / 88_200 all produce ~2000 zero
        # crossings with the matching buffer length.
        cfg = sf_lib.AudioGraphConfig()
        cfg.backend_name = "null"
        cfg.sample_rate = sample_rate
        graph = sf_lib.AudioGraph(config=cfg, start=False)
        voices = [
            _build_voice(
                shape=shape,
                cutoff_hz=cutoff_hz,
                resonance=resonance,
                midi_pitch=pitch,
                amp_adsr=amp_adsr,
            )
            for pitch in midi_pitches
        ]
        summed = voices[0]
        for v in voices[1:]:
            summed = summed + v
        summed.play()
        n_samples = int(round(sample_rate * total_duration_s))
        buf = graph.render_to_new_buffer(num_frames=n_samples)
        arr = np.asarray(buf.data, dtype=np.float32)
        audio = arr[0] if arr.ndim == 2 else arr
        if audio.size != n_samples:
            # SignalFlow rounding edge case - pad / truncate to expected length.
            audio = np.pad(audio, (0, max(0, n_samples - audio.size)))[:n_samples]
        peak = float(np.abs(audio).max())
        if peak > 1e-6:
            audio = audio * (0.95 / peak)
        return audio.astype(np.float32, copy=False)
    finally:
        # Tear down the global singleton so the next render_chord call gets a
        # fresh graph instead of silently reusing this one.
        if graph is not None:
            graph.destroy()
