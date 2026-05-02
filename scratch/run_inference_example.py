"""Inference harness for the trained subtractive tone-generation MVP checkpoint.

By default, this renders a synthetic test chord with KNOWN params via the project's
own SignalFlow renderer, then runs the trained model on that audio, and prints both
ground-truth and prediction side-by-side. That makes it a self-test you can run
without any external audio.

Pass `--wav <path>` to run inference on a real recording instead. You must also pass
`--midi-pitches` (the MIDI notes being played) since the model conditions on them.

Run:
    uv run python scratch/run_inference_example.py
    uv run python scratch/run_inference_example.py \\
        --wav my_chord.wav --midi-pitches 60 64 67

Constraints inherited from the trained model:
    - sample rate must be 44_100 Hz
    - sustain region must contain ≥ 300 ms of audio starting 100 ms after note-on
    - 1-3 voices, MIDI pitches inside [36, 84]
    - synth audio only (synthetic→real gap not yet addressed)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_analysis_mcp.research.tone_generation.dataset import (
    _PITCH_WINDOW,
    _SLICE_DURATION_S,
    _SLICE_OFFSET_S,
    _audio_to_mel,
    _pitch_multihot,
)
from audio_analysis_mcp.research.tone_generation.model import ToneGenerationCNN
from audio_analysis_mcp.research.tone_generation.renderer import render_chord
from audio_analysis_mcp.research.tone_generation.schema_io import (
    SHAPE_LABELS,
    build_canonical_instance,
    denormalize_predictions,
    validate_canonical,
)


_DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "scratch" / "tone_gen_checkpoints" / "checkpoint.pt"
_SAMPLE_RATE = 44_100
_TOTAL_DURATION_S = 1.2
# Required minimum: 100ms attack-skip + 300ms sustain slice = 0.40s.
_MIN_DURATION_S = _SLICE_OFFSET_S + _SLICE_DURATION_S

# Synthetic test-mode defaults: a known C-major triad with definite synth params.
_TEST_PITCHES = [60, 64, 67]  # C4, E4, G4
_TEST_PARAMS = {
    "shape": "saw",
    "cutoff_hz": 1500.0,
    "resonance": 0.6,
}


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_audio(wav_path: Path) -> np.ndarray:
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != _SAMPLE_RATE:
        raise SystemExit(f"sample rate must be {_SAMPLE_RATE}, got {sr} ({wav_path})")
    if len(audio) < int(_MIN_DURATION_S * _SAMPLE_RATE):
        raise SystemExit(
            f"audio too short: {len(audio) / _SAMPLE_RATE:.2f}s; "
            f"need at least {_MIN_DURATION_S}s (100ms attack-skip + 300ms sustain slice). "
            f"Note-on must be aligned to t=0 in the file."
        )
    return audio.astype(np.float32, copy=False)


def _render_test_chord() -> np.ndarray:
    instance = build_canonical_instance(
        shape=_TEST_PARAMS["shape"],
        cutoff_hz=_TEST_PARAMS["cutoff_hz"],
        resonance=_TEST_PARAMS["resonance"],
    )
    return render_chord(
        params_canonical=instance,
        midi_pitches=_TEST_PITCHES,
        sample_rate=_SAMPLE_RATE,
        total_duration_s=_TOTAL_DURATION_S,
    )


def _print_prediction_table(predicted: dict, ground_truth: dict | None) -> None:
    p = predicted["params"]
    pred = {
        "shape": p["osc"]["1"]["shape"],
        "cutoff_hz": p["filter"]["lp"]["cutoff_hz"],
        "resonance": p["filter"]["lp"]["resonance"],
    }
    if ground_truth is None:
        print(f"\nprediction:")
        print(f"  shape     = {pred['shape']}")
        print(f"  cutoff_hz = {pred['cutoff_hz']:.1f}")
        print(f"  resonance = {pred['resonance']:.3f}")
        return
    print(f"\n{'param':<12} {'ground truth':<18} {'predicted':<18} {'delta':<10}")
    print("-" * 60)
    print(f"{'shape':<12} {ground_truth['shape']:<18} {pred['shape']:<18} {'✓' if ground_truth['shape'] == pred['shape'] else '✗'}")
    cutoff_err_pct = 100 * (pred["cutoff_hz"] - ground_truth["cutoff_hz"]) / ground_truth["cutoff_hz"]
    print(f"{'cutoff_hz':<12} {ground_truth['cutoff_hz']:<18.1f} {pred['cutoff_hz']:<18.1f} {cutoff_err_pct:+.1f}%")
    print(f"{'resonance':<12} {ground_truth['resonance']:<18.3f} {pred['resonance']:<18.3f} {pred['resonance'] - ground_truth['resonance']:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--wav", type=Path, default=None, help="Audio file (44.1 kHz mono); omit for synthetic self-test")
    parser.add_argument("--midi-pitches", type=int, nargs="+", default=None, help="MIDI pitches being played (1-3 ints in [36,84]); required when --wav is provided")
    parser.add_argument("--full-instance", action="store_true", help="Print the full canonical schema instance (not just the 3 free params)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found at {args.checkpoint} — train the model first")

    if args.wav is None:
        # Synthetic self-test mode.
        print(f"synthetic self-test mode (no --wav given)")
        print(f"rendering test chord: pitches={_TEST_PITCHES}, params={_TEST_PARAMS}")
        audio = _render_test_chord()
        midi_pitches = list(_TEST_PITCHES)
        ground_truth = _TEST_PARAMS
    else:
        if not args.midi_pitches:
            raise SystemExit("--midi-pitches is required when --wav is provided")
        if not (1 <= len(args.midi_pitches) <= 3):
            raise SystemExit(f"need 1-3 pitches, got {len(args.midi_pitches)}")
        for p in args.midi_pitches:
            if not (36 <= p <= 84):
                raise SystemExit(f"pitch {p} outside trained range [36, 84]")
        if len(args.midi_pitches) > 1:
            if len(set(args.midi_pitches)) != len(args.midi_pitches):
                raise SystemExit(f"MIDI pitches must be distinct: {args.midi_pitches}")
            if max(args.midi_pitches) - min(args.midi_pitches) > _PITCH_WINDOW:
                raise SystemExit(
                    f"MIDI pitches span {max(args.midi_pitches) - min(args.midi_pitches)} semitones; "
                    f"training distribution caps multi-voice chords at {_PITCH_WINDOW} semitones"
                )
        audio = _load_audio(args.wav)
        midi_pitches = list(args.midi_pitches)
        ground_truth = None

    device = _select_device()
    print(f"device: {device}")

    model = ToneGenerationCNN().to(device)
    try:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except Exception as exc:  # noqa: BLE001 — fallback path for legacy/custom-class checkpoints
        print(f"warning: torch.load(weights_only=True) failed ({exc!r}); falling back to weights_only=False")
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    mel = _audio_to_mel(audio, _SAMPLE_RATE).unsqueeze(0).to(device)
    pitch_mh = _pitch_multihot(midi_pitches).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(mel, pitch_mh)

    instance = denormalize_predictions(
        shape_label=int(out["shape_logits"].argmax(dim=1).item()),
        cutoff_norm=float(out["cutoff_norm"].item()),
        resonance=float(out["resonance"].item()),
        midi_pitches=midi_pitches,
    )
    validate_canonical(instance)

    _print_prediction_table(instance, ground_truth)

    if args.full_instance:
        print("\nfull canonical instance:")
        print(json.dumps(instance, indent=2))

    # Show the model's softmax confidence over shape, since it's useful info.
    shape_probs = torch.softmax(out["shape_logits"], dim=1).squeeze(0).tolist()
    print(f"\nshape softmax: " + ", ".join(f"{lbl}={p:.3f}" for lbl, p in zip(SHAPE_LABELS, shape_probs)))


if __name__ == "__main__":
    main()
