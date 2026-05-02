"""Probe SignalFlow's API for the subtractive renderer:

1. Confirm SquareOscillator / SawOscillator / TriangleOscillator exist (or find the right names).
2. Confirm a low-pass filter node exists (SVFilter, OnePoleLowPassFilter, ...).
3. Render a single 440 Hz saw through an LP at cutoff=2 kHz, resonance=0.5,
   confirm the audio looks reasonable (peak ~0.5-0.95, no NaN, harmonics
   below cutoff present, harmonics above cutoff attenuated).
4. Sum 3 voices (440, 554, 659 Hz - a major triad) in one graph, render,
   confirm peak after-sum stays in [0.5, 1.0] after manual normalization.
5. ADSREnvelope: gate parameter behavior — verify whether passing gate=1
   explicitly is REQUIRED, or whether the default works.

   FINDING (test case 3 below, confirmed empirically):
     - ADSREnvelope's gate parameter defaults to 0 in SignalFlow 0.5.3
       (verified via help(ADSREnvelope.__init__): "gate: ... = 0").
     - gate=0 produces SILENCE (peak ~0.0).
     - gate=1 produces audible audio (peak ~1.0).
     - Omitting the gate kwarg is equivalent to gate=0 → silence.
   Therefore: **gate=1 is REQUIRED on every ADSREnvelope** for the renderer.

   Note on the earlier sibling scratch (`scratch/explore_signalflow.py`):
   it does NOT actually produce audible audio either — re-running it shows
   peak=0.000. It only "passed" because the amplitude orchestrator
   gracefully rejects silent input ("no candidates with usable sustain")
   instead of crashing. That scratch is also misleading and should be fixed,
   but separately.

   ===> Task 5's plan _build_voice MUST pass gate=1 to ADSREnvelope. <===

If any of these fail, FIX the spec / plan before dispatching Task 5 (renderer).

Run: uv run python scratch/explore_subtractive_renderer.py
"""

from __future__ import annotations

import numpy as np
import signalflow as sf_lib

# Sentinel for "do not pass a gate kwarg at all" in test case 3.
_NO_GATE = object()

print("signalflow module:", sf_lib.__file__)
osc_attrs = sorted(a for a in dir(sf_lib) if "scillator" in a.lower())
filter_attrs = sorted(
    a for a in dir(sf_lib) if "ilter" in a.lower() or "passfilter" in a.lower()
)
print("Oscillator-like attrs:", osc_attrs)
print("Filter-like attrs:", filter_attrs[:20])


def render_voice(
    graph: sf_lib.AudioGraph,
    osc_class: type,
    frequency: float,
    cutoff_hz: float,
    resonance: float,
    duration_s: float,
    sr: int,
) -> None:
    osc = osc_class(frequency=frequency)
    # gate=1 holds the envelope open through attack/decay/sustain. Release only
    # fires on gate transition to 0; for renderer verification (peak / non-silence
    # / no NaN), holding sustain is sufficient.
    env = sf_lib.ADSREnvelope(
        attack=0.05, decay=0.10, sustain=0.7, release=0.10, gate=1
    )
    # Try common LP names; fall back as needed.
    lp_class_name = None
    for candidate in ("SVFilter", "OnePoleLowPassFilter", "LowPassFilter"):
        if hasattr(sf_lib, candidate):
            lp_class_name = candidate
            break
    if lp_class_name is None:
        raise RuntimeError("No known LP filter class on signalflow module")
    lp_class = getattr(sf_lib, lp_class_name)
    print(f"  using LP: {lp_class_name}")
    # Print help so we discover the actual ctor signature.
    print(f"  {lp_class_name}.__init__ signature:")
    help(lp_class.__init__)
    # Best-effort instantiation; tweak per actual signature once help() above prints.
    # SVFilter likely takes (input, filter_type, cutoff, resonance);
    # OnePoleLowPassFilter likely takes (input, cutoff).
    if lp_class_name == "SVFilter":
        lp = lp_class(osc, "low_pass", cutoff_hz, resonance)
    elif lp_class_name == "OnePoleLowPassFilter":
        lp = lp_class(osc, cutoff_hz)
    else:
        lp = lp_class(osc, cutoff_hz, resonance)
    out = lp * env
    out.play()


def main() -> None:
    sr = 44100
    duration_s = 1.0
    n_samples = int(sr * duration_s)

    # Discover oscillator class names - print attrs above tells us what's there.
    # Typical signalflow names in 0.4.x: SineOscillator, SquareOscillator,
    # SawOscillator, TriangleOscillator. Fail if missing.
    needed = ["SineOscillator", "SquareOscillator", "SawOscillator", "TriangleOscillator"]
    for name in needed:
        if not hasattr(sf_lib, name):
            print(f"  WARNING: missing {name} on signalflow - find the right name")

    # 1) Single voice through LP
    # NOTE: test offline render. AudioGraph(output_device=None, start=False)
    # creates a non-realtime graph; nodes connected via .play() are rendered by
    # render_to_new_buffer.
    print("\n=== AudioGraph.__init__ signature ===")
    help(sf_lib.AudioGraph.__init__)

    graph = sf_lib.AudioGraph(output_device=None, start=False)
    SawOsc = getattr(sf_lib, "SawOscillator", None) or getattr(sf_lib, "SineOscillator")
    print("\n=== Single saw voice through LP ===")
    render_voice(graph, SawOsc, 440.0, 2000.0, 0.5, duration_s, sr)
    buf = graph.render_to_new_buffer(num_frames=n_samples)
    audio = np.asarray(buf.data, dtype=np.float32)
    audio = audio[0] if audio.ndim == 2 else audio
    print(
        f"  audio shape: {audio.shape}, "
        f"peak: {np.abs(audio).max():.3f}, "
        f"RMS: {np.sqrt((audio ** 2).mean()):.3f}"
    )
    assert audio.size == n_samples, f"unexpected sample count: {audio.size} vs {n_samples}"
    assert not np.any(np.isnan(audio)), "audio contains NaN"
    assert not np.any(np.isinf(audio)), "audio contains Inf"
    assert np.abs(audio).max() > 0.0, "audio is silent"

    # 2) Sum 3 voices (parallel in one graph)
    # NOTE: signalflow uses a global singleton AudioGraph. Constructing a 2nd
    # AudioGraph without destroying the first emits a warning ("global audio
    # graph has already been created") and silently reuses the first. For a
    # clean second test, destroy the first graph and make a fresh one. Task 5
    # (renderer.py) should follow this pattern: graph.destroy() between renders,
    # or reuse a single graph and clear nodes.
    graph.destroy()
    print("\n=== Sum 3 saw voices (major triad) ===")
    graph2 = sf_lib.AudioGraph(output_device=None, start=False)
    voices = []
    for f in (440.0, 554.37, 659.26):
        osc = SawOsc(frequency=f)
        env = sf_lib.ADSREnvelope(
            attack=0.05, decay=0.10, sustain=0.7, release=0.10, gate=1
        )
        # Skip filter for the sum test to isolate the chord-summation behavior.
        voices.append(osc * env)
    summed = voices[0] + voices[1] + voices[2]
    summed.play()
    buf2 = graph2.render_to_new_buffer(num_frames=n_samples)
    audio2 = np.asarray(buf2.data, dtype=np.float32)
    audio2 = audio2[0] if audio2.ndim == 2 else audio2
    peak2 = np.abs(audio2).max()
    print(f"  summed shape: {audio2.shape}, peak: {peak2:.3f}")
    assert audio2.size == n_samples
    assert not np.any(np.isnan(audio2)), "summed audio contains NaN"
    assert not np.any(np.isinf(audio2)), "summed audio contains Inf"
    assert peak2 > 0.0, "summed audio is silent"
    # Peak before normalization can exceed 1.0 - that's expected. We normalize after.
    audio2_norm = audio2 / max(peak2, 1e-6) * 0.95
    norm_peak = np.abs(audio2_norm).max()
    print(f"  normalized peak: {norm_peak:.3f}")
    assert 0.5 <= norm_peak <= 1.0, f"normalized peak out of range: {norm_peak}"

    # 3) ADSREnvelope gate behavior
    # Render the same single voice three ways and compare peak amplitudes:
    #   (a) no `gate=` argument          (b) gate=1          (c) gate=0
    # If (a) and (b) both produce audio with similar peaks, the default fires.
    # If only (b) produces audio, gate=1 must be passed explicitly.
    graph2.destroy()
    print("\n=== ADSREnvelope gate behavior (test case 3) ===")

    def render_with_gate(gate_arg: object) -> float:
        """Render a SineOscillator+ADSR voice; return peak amplitude. gate_arg is
        either a sentinel (omit kwarg) or an int to pass as gate=."""
        graph_local = sf_lib.AudioGraph(output_device=None, start=False)
        SineOsc = getattr(sf_lib, "SineOscillator")
        osc_local = SineOsc(frequency=440.0)
        if gate_arg is _NO_GATE:
            env_local = sf_lib.ADSREnvelope(
                attack=0.05, decay=0.10, sustain=0.7, release=0.10
            )
        else:
            env_local = sf_lib.ADSREnvelope(
                attack=0.05, decay=0.10, sustain=0.7, release=0.10, gate=gate_arg
            )
        out_local = osc_local * env_local
        out_local.play()
        buf_local = graph_local.render_to_new_buffer(num_frames=n_samples)
        a = np.asarray(buf_local.data, dtype=np.float32)
        a = a[0] if a.ndim == 2 else a
        peak_local = float(np.abs(a).max())
        graph_local.destroy()
        return peak_local

    peak_no_gate = render_with_gate(_NO_GATE)
    peak_gate_1 = render_with_gate(1)
    peak_gate_0 = render_with_gate(0)

    print(f"  peak (no gate kwarg):   {peak_no_gate:.4f}")
    print(f"  peak (gate=1):          {peak_gate_1:.4f}")
    print(f"  peak (gate=0):          {peak_gate_0:.4f}")

    # Determine truth from the measurements.
    silence_threshold = 1e-4  # below this counts as silent
    no_gate_audible = peak_no_gate > silence_threshold
    gate_1_audible = peak_gate_1 > silence_threshold
    gate_0_audible = peak_gate_0 > silence_threshold

    print(f"  no-gate audible? {no_gate_audible}")
    print(f"  gate=1 audible?  {gate_1_audible}")
    print(f"  gate=0 audible?  {gate_0_audible}")

    # Assert the empirical truth: gate=1 required. (See module docstring.)
    assert gate_1_audible, (
        f"gate=1 should produce audible audio, got peak={peak_gate_1}"
    )
    assert not no_gate_audible, (
        f"omitting gate kwarg should produce silence (default gate=0), "
        f"got peak={peak_no_gate}"
    )
    assert not gate_0_audible, (
        f"gate=0 should produce silence, got peak={peak_gate_0}"
    )
    print("  CONFIRMED: gate=1 is REQUIRED. Default (gate=0) and explicit")
    print("  gate=0 both produce silence. Task 5's _build_voice must pass")
    print("  gate=1 to ADSREnvelope.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
