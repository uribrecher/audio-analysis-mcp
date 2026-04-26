"""Probe SignalFlow's API: render a known-ADSR note and run it through analyze_amplitude.

The plan's slow integration test (Task 8) renders an ASR-shaped note via
SignalFlow with attack=20ms, sustain=600ms, release=200ms at 440Hz, runs
analyze_amplitude on a fabricated single-cluster triage, and asserts ADSR
recovery within tolerance. This scratch confirms:
  1. SignalFlow installed and importable on this platform.
  2. The exact constructor names work (ASREnvelope vs ASR, etc.).
  3. render_to_buffer (or whatever its offline-render API is) returns audio
     we can pass to analyze_amplitude.
  4. analyze_amplitude actually recovers the ADSR within reasonable tolerance.

If any of these fail, fix the plan's T8 test code BEFORE dispatching the
implementer.

Run: uv run python scratch/explore_signalflow.py
"""

from pathlib import Path
import tempfile

import numpy as np
import signalflow as sf_lib
import soundfile as sf

print("signalflow module:", sf_lib.__file__)
print("signalflow attrs (first 60):", sorted([a for a in dir(sf_lib) if not a.startswith("_")])[:60])
print()
print("ADSREnvelope help:")
help(sf_lib.ADSREnvelope.__init__)
print()


def main() -> None:
    sr = 44100
    duration = 1.5
    attack_s, sustain_s, release_s = 0.020, 0.600, 0.200

    # Try the API forms the plan assumes.
    print("=== Constructing SignalFlow graph ===")
    AudioGraph = sf_lib.AudioGraph
    SineOscillator = sf_lib.SineOscillator
    ADSREnvelope = sf_lib.ADSREnvelope
    print(f"  AudioGraph signature hint: try output_device=None, start=False")

    # Use ADSR (4-segment) so envelope has a peak at attack-end and a sustain plateau below it.
    # ASR (3-segment) makes argmax land randomly in the flat plateau and breaks fit_adsr.
    decay_s = 0.10
    sustain_level = 0.6
    graph = AudioGraph(output_device=None, start=False)
    osc = SineOscillator(frequency=440.0)
    env = ADSREnvelope(attack=attack_s, decay=decay_s, sustain=sustain_level, release=release_s)
    out = osc * env
    print(f"  graph: {graph}")
    print(f"  osc * env type: {type(out)}")

    # SignalFlow 0.4.x API: connect node to graph via .play(), then render the graph.
    # `render_to_buffer(buffer)` writes graph output INTO an existing buffer.
    # `render_to_new_buffer(num_frames)` allocates and returns a new Buffer.
    out.play()
    n_samples = int(sr * duration)
    buffer = graph.render_to_new_buffer(num_frames=n_samples)
    print(f"  render_to_new_buffer returned: type={type(buffer)}")
    if hasattr(buffer, "data"):
        data = np.asarray(buffer.data, dtype=np.float32)
        print(f"  buffer.data shape: {data.shape}")
        # Buffer is (channels, samples) — take channel 0.
        audio = data[0] if data.ndim == 2 else data
    else:
        audio = np.asarray(buffer, dtype=np.float32)

    print()
    print(f"=== Rendered audio ===")
    print(f"  size: {audio.size} samples ({audio.size / sr:.2f}s)")
    print(f"  peak amplitude: {abs(audio).max():.3f}")
    print(f"  RMS: {np.sqrt((audio ** 2).mean()):.3f}")

    # Show the envelope shape (RMS over 5ms windows) so we can see what the ADSR fitter sees.
    win = int(0.005 * sr)
    n_frames = audio.size // win
    rms_envelope = np.array([
        np.sqrt(np.mean(audio[i*win:(i+1)*win] ** 2))
        for i in range(n_frames)
    ])
    print(f"  RMS envelope (200Hz, first 50 frames @ 5ms each):")
    for i in range(0, min(50, len(rms_envelope)), 5):
        print(f"    t={i*5}ms: {rms_envelope[i]:.4f}")
    print(f"  argmax frame: {int(np.argmax(rms_envelope))} (= {int(np.argmax(rms_envelope)) * 5}ms)")
    print(f"  envelope min in sustain region (frames 10-120): {rms_envelope[10:120].min():.4f}, max: {rms_envelope[10:120].max():.4f}")

    # Save and run through analyze_amplitude.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wav_path = tmp_path / "rendered.wav"
        sf.write(wav_path, audio, sr)
        print(f"  wrote {wav_path}")

        # Fabricate a single-cluster triage.
        from audio_analysis_mcp.schemas import (
            CandidateCluster, CandidateNote, NoteEvent, NoteTriageFileData,
        )
        cluster = CandidateCluster(
            kind="single", score=3.0,
            start_time=0.0, end_time=float(audio.size) / sr,
            start_freq=200.0, end_freq=2000.0,
            members=[CandidateNote(
                note=NoteEvent(start_time=0.0, end_time=float(audio.size) / sr,
                               pitch_midi=69, amplitude=1.0, pitch_bends=None),
                score=3.0, start_time=0.0, end_time=float(audio.size) / sr,
                start_freq=200.0, end_freq=2000.0,
            )],
        )
        triage_path = tmp_path / "triage.json"
        triage_path.write_text(NoteTriageFileData(
            polyphony_profile=[], candidates=[cluster],
        ).model_dump_json(indent=2))

        from audio_analysis_mcp.analysis.amplitude import analyze_amplitude
        result = analyze_amplitude(
            audio=audio, sample_rate=sr,
            triage_path=triage_path, output_dir=tmp_path / "amp",
        )

        print()
        print("=== Orchestrator result ===")
        print(f"  rejected_reason: {result.rejected_reason}")
        print(f"  candidates: {len(result.candidates)}")
        print(f"  is_consistent: {result.is_consistent}")
        if result.candidates:
            adsr = result.candidates[0].adsr
            print(f"  recovered ADSR: attack={adsr.attack_ms:.0f}ms decay={adsr.decay_ms:.0f}ms sustain={adsr.sustain_level:.2f} release={adsr.release_ms:.0f}ms")
            print()
            print("=== Tolerance check ===")
            print(f"  attack:  expected 20, got {adsr.attack_ms:.1f},  |Δ|={abs(adsr.attack_ms - 20.0):.1f}, tol 25 → {'OK' if abs(adsr.attack_ms - 20.0) < 25.0 else 'FAIL'}")
            print(f"  release: expected 200, got {adsr.release_ms:.1f}, |Δ|={abs(adsr.release_ms - 200.0):.1f}, tol 60 → {'OK' if abs(adsr.release_ms - 200.0) < 60.0 else 'FAIL'}")
            print(f"  sustain: expected >0.5, got {adsr.sustain_level:.3f} → {'OK' if adsr.sustain_level > 0.5 else 'FAIL'}")


if __name__ == "__main__":
    main()
