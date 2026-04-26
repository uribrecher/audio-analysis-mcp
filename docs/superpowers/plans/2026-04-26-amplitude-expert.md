# Amplitude Expert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the amplitude expert from `reverse-synth-research/amplitude/amplitude-envelope-research-plan.md` as a Python module + MCP tool inside `audio-analysis-mcp`. Given audio + transcribed MIDI, produce an ADSR-triage classification, a canonical ADSR estimate, and an isolated sustain slice for the downstream tone-generation expert.

**Architecture:** Pure-logic functions live in `analysis/` (one module per phase: `adsr_triage`, `envelope`, `adsr_fit`, `sustain_isolation`), composed by an `analysis/amplitude.py` orchestrator. A thin MCP wrapper in `tools/amplitude_analyze.py` handles file I/O and workspace paths. Outputs (envelope curve, sustain slice) are written to disk as `.npy` / `.wav`; the MCP result returns paths and the canonical ADSR struct. The modulation expert is intentionally not served by this stage — see the research plan.

**Tech Stack:** Python 3.11+, `numpy`, `librosa` (already a dep), `soundfile`, `pydantic`, `pytest`. SignalFlow is added as a `dev` dep for one slow integration test only — runtime code does not import it.

**Scope (v1):** Triage (monophonic / block_chord / arpeggio→reject), one envelope-extraction method (RMS sliding window), heuristic ADSR fit, sustain isolation, MCP tool. Comparing multiple envelope-extraction methods, fitting non-monophonic arpeggios per-note, and the natural-envelope passthrough heuristic for acoustic instruments are deferred to v2.

---

## File Structure

**Create:**
- `src/audio_analysis_mcp/analysis/adsr_triage.py` — onset-density classification
- `src/audio_analysis_mcp/analysis/envelope.py` — RMS sliding-window envelope extraction
- `src/audio_analysis_mcp/analysis/adsr_fit.py` — heuristic ADSR fit from an envelope curve
- `src/audio_analysis_mcp/analysis/sustain_isolation.py` — trim audio to sustained region
- `src/audio_analysis_mcp/analysis/amplitude.py` — orchestrator: triage → envelope → fit → isolate
- `src/audio_analysis_mcp/tools/amplitude_analyze.py` — MCP tool wrapper
- `tests/test_adsr_triage.py`
- `tests/test_envelope.py`
- `tests/test_adsr_fit.py`
- `tests/test_sustain_isolation.py`
- `tests/test_amplitude.py` — orchestrator integration (numpy fixtures)
- `tests/test_amplitude_signalflow.py` — slow end-to-end with rendered synth audio

**Modify:**
- `src/audio_analysis_mcp/schemas.py` — add `AmplitudeTriage` enum, `AmplitudeAnalyzeResult` model
- `src/audio_analysis_mcp/workspace.py` — add `job_amplitude_dir(...)` helper
- `src/audio_analysis_mcp/server.py` — import the new tool module so `@mcp.tool()` registers it
- `pyproject.toml` — add `signalflow` to `[tool.uv.dev-dependencies]`

---

## Conventions

- All modules use the existing `from audio_analysis_mcp.schemas import ...` pattern.
- Pure-logic functions take and return numpy arrays / pydantic models — no file I/O.
- The MCP tool reads files, calls the orchestrator, writes outputs to the job workspace, returns a JSON-serialized pydantic result.
- Tests follow the existing `tests/test_*.py` style — synchronous functions, numpy fixtures, no async.
- After every task: run `pytest -m "not slow"` and `mypy src/` and verify both pass before commit.
- Slow tests are marked `@pytest.mark.slow` and excluded from the default CI run.

---

## Task 1: Schemas

**Files:**
- Modify: `src/audio_analysis_mcp/schemas.py`
- Test: `tests/test_amplitude.py` (created here, expanded later)

- [ ] **Step 1: Write the failing test**

Create `tests/test_amplitude.py`:

```python
from audio_analysis_mcp.schemas import (
    AmplitudeTriage,
    AmplitudeAnalyzeResult,
    ADSREstimate,
)


def test_amplitude_triage_values():
    assert AmplitudeTriage.MONOPHONIC.value == "monophonic"
    assert AmplitudeTriage.BLOCK_CHORD.value == "block_chord"
    assert AmplitudeTriage.ARPEGGIO.value == "arpeggio"
    assert AmplitudeTriage.REJECTED.value == "rejected"


def test_amplitude_analyze_result_minimal():
    result = AmplitudeAnalyzeResult(
        adsr_triage=AmplitudeTriage.MONOPHONIC,
        adsr=ADSREstimate(
            attack_ms=12.0,
            decay_ms=380.0,
            sustain_level=0.62,
            release_ms=220.0,
        ),
        envelope_curve_path="/tmp/envelope.npy",
        sustain_slice_path="/tmp/sustain.wav",
    )
    assert result.adsr_triage == AmplitudeTriage.MONOPHONIC
    assert result.sustain_slice_path == "/tmp/sustain.wav"


def test_amplitude_analyze_result_rejected_has_no_sustain():
    result = AmplitudeAnalyzeResult(
        adsr_triage=AmplitudeTriage.REJECTED,
        adsr=None,
        envelope_curve_path="/tmp/envelope.npy",
        sustain_slice_path=None,
    )
    assert result.adsr is None
    assert result.sustain_slice_path is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_amplitude.py -v`
Expected: FAIL with `ImportError: cannot import name 'AmplitudeTriage'`

- [ ] **Step 3: Add the schemas**

Append to `src/audio_analysis_mcp/schemas.py`:

```python
from enum import Enum


class AmplitudeTriage(str, Enum):
    MONOPHONIC = "monophonic"
    BLOCK_CHORD = "block_chord"
    ARPEGGIO = "arpeggio"
    REJECTED = "rejected"


class AmplitudeAnalyzeResult(BaseModel):
    adsr_triage: AmplitudeTriage
    adsr: ADSREstimate | None
    envelope_curve_path: str
    sustain_slice_path: str | None
```

(The `Enum` import goes at the top of the file alongside the existing `from pydantic import BaseModel`. `ADSREstimate` is already defined.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_amplitude.py -v && uv run mypy src/`
Expected: 3 passed, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/schemas.py tests/test_amplitude.py
git commit -m "amplitude: add AmplitudeTriage enum and AmplitudeAnalyzeResult schema"
```

---

## Task 2: ADSR-Triage

Classifies a list of `NoteEvent`s as monophonic / block_chord / arpeggio. Per scope decision, arpeggio is downgraded to "rejected" for v1 (no per-note segmentation yet).

**Triage rules:**
- 0 notes → REJECTED.
- Max note velocity across all notes < 0.1 → REJECTED (signal-to-noise ratio too poor for reliable ADSR fitting).
- 1 note → MONOPHONIC.
- ≥2 notes, all start times within 30 ms of each other AND all end times within 30 ms of each other → BLOCK_CHORD.
- Otherwise compute onsets-per-second over the clip span; >3 → ARPEGGIO (→ REJECTED in v1); ≤3 → MONOPHONIC (sequential single notes treated as analyzable; the longest one will be used downstream).

**Files:**
- Create: `src/audio_analysis_mcp/analysis/adsr_triage.py`
- Test: `tests/test_adsr_triage.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_adsr_triage.py`:

```python
from audio_analysis_mcp.analysis.adsr_triage import classify_adsr_triage
from audio_analysis_mcp.schemas import AmplitudeTriage, NoteEvent


def _note(start: float, end: float, pitch: int = 60, amp: float = 0.8) -> NoteEvent:
    return NoteEvent(
        start_time=start,
        end_time=end,
        pitch_midi=pitch,
        amplitude=amp,
        pitch_bends=None,
    )


def test_empty_notes_rejected():
    assert classify_adsr_triage([]) == AmplitudeTriage.REJECTED


def test_single_note_is_monophonic():
    assert classify_adsr_triage([_note(0.0, 1.0)]) == AmplitudeTriage.MONOPHONIC


def test_simultaneous_chord_is_block_chord():
    notes = [_note(0.0, 1.0, 60), _note(0.01, 1.0, 64), _note(0.02, 0.99, 67)]
    assert classify_adsr_triage(notes) == AmplitudeTriage.BLOCK_CHORD


def test_dense_arpeggio_is_rejected():
    # 8 notes in 1 second = 8 onsets/sec
    notes = [_note(i * 0.125, i * 0.125 + 0.1) for i in range(8)]
    assert classify_adsr_triage(notes) == AmplitudeTriage.REJECTED


def test_sequential_two_notes_is_monophonic():
    notes = [_note(0.0, 0.5), _note(0.6, 1.1)]
    assert classify_adsr_triage(notes) == AmplitudeTriage.MONOPHONIC
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_adsr_triage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'audio_analysis_mcp.analysis.adsr_triage'`

- [ ] **Step 3: Implement the classifier**

Create `src/audio_analysis_mcp/analysis/adsr_triage.py`:

```python
from audio_analysis_mcp.schemas import AmplitudeTriage, NoteEvent

_BLOCK_CHORD_TOLERANCE_S = 0.030
_ARPEGGIO_ONSETS_PER_SECOND = 3.0


def classify_adsr_triage(notes: list[NoteEvent]) -> AmplitudeTriage:
    if not notes:
        return AmplitudeTriage.REJECTED
    if len(notes) == 1:
        return AmplitudeTriage.MONOPHONIC

    starts = [n.start_time for n in notes]
    ends = [n.end_time for n in notes]
    if (max(starts) - min(starts) <= _BLOCK_CHORD_TOLERANCE_S
            and max(ends) - min(ends) <= _BLOCK_CHORD_TOLERANCE_S):
        return AmplitudeTriage.BLOCK_CHORD

    span = max(ends) - min(starts)
    if span <= 0:
        return AmplitudeTriage.REJECTED
    onsets_per_second = len(notes) / span
    if onsets_per_second > _ARPEGGIO_ONSETS_PER_SECOND:
        return AmplitudeTriage.REJECTED  # arpeggio: per-note segmentation deferred to v2
    return AmplitudeTriage.MONOPHONIC
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_adsr_triage.py -v && uv run mypy src/`
Expected: 5 passed, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/analysis/adsr_triage.py tests/test_adsr_triage.py
git commit -m "amplitude: add ADSR-triage onset-density classifier"
```

---

## Task 3: Envelope Extraction (RMS Sliding Window)

Compute a smoothed amplitude envelope from mono audio using RMS over a sliding window. Hop length is the step between successive RMS values. The output envelope is sampled at `sample_rate / hop_length` Hz.

**Files:**
- Create: `src/audio_analysis_mcp/analysis/envelope.py`
- Test: `tests/test_envelope.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_envelope.py`:

```python
import numpy as np

from audio_analysis_mcp.analysis.envelope import (
    extract_rms_envelope,
    EnvelopeResult,
)


SR = 22050


def _adsr_test_signal(
    attack_s: float = 0.02,
    decay_s: float = 0.10,
    sustain_level: float = 0.6,
    sustain_s: float = 0.5,
    release_s: float = 0.15,
    sr: int = SR,
    freq: float = 220.0,
) -> np.ndarray:
    """Sine carrier multiplied by a four-segment piecewise envelope. Peak amplitude = 1.0."""
    n_attack = int(attack_s * sr)
    n_decay = int(decay_s * sr)
    n_sustain = int(sustain_s * sr)
    n_release = int(release_s * sr)
    env = np.concatenate([
        np.linspace(0.0, 1.0, n_attack, endpoint=False),
        np.linspace(1.0, sustain_level, n_decay, endpoint=False),
        np.full(n_sustain, sustain_level),
        np.linspace(sustain_level, 0.0, n_release, endpoint=True),
    ])
    t = np.arange(env.size) / sr
    return (env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_envelope_result_shape():
    audio = _adsr_test_signal()
    result = extract_rms_envelope(audio, sample_rate=SR)
    assert isinstance(result, EnvelopeResult)
    assert result.envelope.ndim == 1
    assert result.envelope_sample_rate > 0
    # librosa.feature.rms with center=False produces this exact frame count
    expected_len = (audio.size - result.frame_length) // result.hop_length + 1
    assert result.envelope.size == expected_len


def test_envelope_tracks_amplitude_shape():
    audio = _adsr_test_signal()
    result = extract_rms_envelope(audio, sample_rate=SR)
    env = result.envelope
    # Peak should be near the attack→decay boundary, not at the very end (release)
    peak_idx = int(np.argmax(env))
    peak_time = peak_idx / result.envelope_sample_rate
    assert 0.01 < peak_time < 0.05, f"peak at {peak_time}s outside attack region"
    # Final sample should be near zero (release ended)
    assert env[-1] < 0.05


def test_envelope_silence_has_low_rms():
    silence = np.zeros(SR, dtype=np.float32)
    result = extract_rms_envelope(silence, sample_rate=SR)
    assert float(result.envelope.max()) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_envelope.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'audio_analysis_mcp.analysis.envelope'`

- [ ] **Step 3: Implement the extractor**

Create `src/audio_analysis_mcp/analysis/envelope.py`:

```python
import librosa
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

_DEFAULT_FRAME_LENGTH_MS = 20.0
_DEFAULT_HOP_LENGTH_MS = 5.0


class EnvelopeResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    envelope: npt.NDArray[np.float32]   # 1-D, RMS values
    envelope_sample_rate: float          # frames per second
    hop_length: int                      # samples between frames
    frame_length: int                    # samples per RMS window


def extract_rms_envelope(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    frame_length_ms: float = _DEFAULT_FRAME_LENGTH_MS,
    hop_length_ms: float = _DEFAULT_HOP_LENGTH_MS,
) -> EnvelopeResult:
    if audio.ndim != 1:
        raise ValueError(f"audio must be mono (1-D), got shape {audio.shape}")
    frame_length = max(1, int(round(frame_length_ms * sample_rate / 1000.0)))
    hop_length = max(1, int(round(hop_length_ms * sample_rate / 1000.0)))
    rms = librosa.feature.rms(
        y=audio.astype(np.float32),
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    )[0]
    return EnvelopeResult(
        envelope=rms.astype(np.float32),
        envelope_sample_rate=sample_rate / hop_length,
        hop_length=hop_length,
        frame_length=frame_length,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_envelope.py -v && uv run mypy src/`
Expected: 3 passed, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/analysis/envelope.py tests/test_envelope.py
git commit -m "amplitude: add RMS sliding-window envelope extractor"
```

---

## Task 4: ADSR Fitting

Heuristic four-segment fit on an envelope curve. `sustain_level` is computed as a fraction of the envelope's peak (`sustain_rms / envelope.max()`), which is naturally velocity-invariant — both peak and sustain scale linearly with hit-strength, so the ratio recovers the engine-intrinsic shape regardless of how hard the note was struck. Velocity is *not* an input to this stage; it's used by triage (Task 2) to reject low-SNR notes before fitting.

**Algorithm:**
1. Find peak of envelope → defines attack endpoint.
2. Attack time = peak time minus the time the envelope first crossed `0.05 * peak`.
3. Find sustain region: starting from the peak, slide a 50 ms window forward; the sustain region is the longest contiguous span where the windowed standard deviation is below `0.05 * peak`. Sustain stops when stddev exceeds threshold or envelope drops below `0.10 * peak`.
4. Sustain level = mean of sustain region, divided by velocity, clamped to [0, 1].
5. Decay time = sustain start - peak time.
6. Release time = time from sustain end until envelope drops below `0.05 * peak`.

If no sustain region of ≥30 ms is found, the note is treated as a "no sustain" pluck: sustain_level=0, sustain start = sustain end = the moment envelope drops below `0.5 * peak` after the peak.

**Files:**
- Create: `src/audio_analysis_mcp/analysis/adsr_fit.py`
- Test: `tests/test_adsr_fit.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_adsr_fit.py`:

```python
import numpy as np

from audio_analysis_mcp.analysis.adsr_fit import fit_adsr, ADSRFit


def _piecewise_envelope(
    attack_s: float,
    decay_s: float,
    sustain_level: float,
    sustain_s: float,
    release_s: float,
    sr: float = 200.0,  # 5ms hop → 200 Hz envelope rate
) -> np.ndarray:
    n_a = int(attack_s * sr)
    n_d = int(decay_s * sr)
    n_s = int(sustain_s * sr)
    n_r = int(release_s * sr)
    return np.concatenate([
        np.linspace(0.0, 1.0, n_a, endpoint=False),
        np.linspace(1.0, sustain_level, n_d, endpoint=False),
        np.full(n_s, sustain_level),
        np.linspace(sustain_level, 0.0, n_r, endpoint=True),
    ]).astype(np.float32)


def test_fit_returns_canonical_struct():
    env = _piecewise_envelope(0.02, 0.10, 0.6, 0.5, 0.15)
    fit = fit_adsr(env, envelope_sample_rate=200.0, peak_velocity=1.0)
    assert isinstance(fit, ADSRFit)
    assert fit.attack_ms > 0
    assert fit.decay_ms > 0
    assert fit.release_ms > 0
    assert 0.0 <= fit.sustain_level <= 1.0


def test_fit_recovers_known_adsr_within_tolerance():
    env = _piecewise_envelope(0.02, 0.10, 0.6, 0.5, 0.15)
    fit = fit_adsr(env, envelope_sample_rate=200.0, peak_velocity=1.0)
    # ±10 ms attack, ±25 ms for decay/release (envelope rate is 5ms)
    assert abs(fit.attack_ms - 20.0) < 10.0
    assert abs(fit.decay_ms - 100.0) < 25.0
    assert abs(fit.sustain_level - 0.6) < 0.05
    assert abs(fit.release_ms - 150.0) < 25.0


def test_fit_returns_sustain_window_indices():
    env = _piecewise_envelope(0.02, 0.10, 0.6, 0.5, 0.15)
    fit = fit_adsr(env, envelope_sample_rate=200.0, peak_velocity=1.0)
    assert fit.sustain_start_idx < fit.sustain_end_idx
    assert fit.sustain_end_idx <= env.size


def test_fit_normalizes_by_velocity():
    # Same shape, struck at half velocity → sustain_level should match the original
    env = _piecewise_envelope(0.02, 0.10, 0.6, 0.5, 0.15) * 0.5
    fit = fit_adsr(env, envelope_sample_rate=200.0, peak_velocity=0.5)
    assert abs(fit.sustain_level - 0.6) < 0.05


def test_fit_pluck_with_no_sustain():
    # Attack + immediate decay to zero, no sustain
    env = np.concatenate([
        np.linspace(0.0, 1.0, 5),
        np.linspace(1.0, 0.0, 40),
    ]).astype(np.float32)
    fit = fit_adsr(env, envelope_sample_rate=200.0, peak_velocity=1.0)
    assert fit.sustain_level < 0.1
    assert fit.sustain_start_idx == fit.sustain_end_idx


def test_fit_handles_exponential_decay():
    # Real synth decays are exponential, not linear. Validate the fitter still recovers ADSR.
    env_sr = 200.0
    n_a, n_d, n_s, n_r = 4, 20, 100, 30
    sus = 0.6
    tau = 0.05  # 50ms time constant
    t_decay = np.arange(n_d) / env_sr
    decay_curve = sus + (1.0 - sus) * np.exp(-t_decay / tau)
    env = np.concatenate([
        np.linspace(0.0, 1.0, n_a, endpoint=False),
        decay_curve,
        np.full(n_s, sus),
        np.linspace(sus, 0.0, n_r, endpoint=True),
    ]).astype(np.float32)
    fit = fit_adsr(env, envelope_sample_rate=env_sr, peak_velocity=1.0)
    assert abs(fit.attack_ms - 20.0) < 10.0
    assert abs(fit.decay_ms - 100.0) < 25.0
    assert abs(fit.sustain_level - 0.6) < 0.05
    assert abs(fit.release_ms - 150.0) < 25.0


def test_fit_sustain_runs_to_envelope_end():
    # Sustain runs to the end of the envelope (no release segment).
    # Exercises the for-else fallback. Exclusive end semantics → sustain_end_idx == envelope.size.
    env = np.concatenate([
        np.linspace(0.0, 1.0, 4, endpoint=False),
        np.linspace(1.0, 0.6, 20, endpoint=False),
        np.full(120, 0.6),
    ]).astype(np.float32)
    fit = fit_adsr(env, envelope_sample_rate=200.0, peak_velocity=1.0)
    assert fit.sustain_end_idx == env.size
    # No release segment → release_ms should be 0
    assert fit.release_ms == 0.0
    # Sustain length should reflect the full plateau (allow attack+decay slack)
    sustain_frames = fit.sustain_end_idx - fit.sustain_start_idx
    assert sustain_frames >= 100  # at least 500ms of the 600ms plateau
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_adsr_fit.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the fitter**

Create `src/audio_analysis_mcp/analysis/adsr_fit.py`:

```python
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

_ATTACK_THRESHOLD = 0.05      # fraction of peak that defines "note start"
_RELEASE_THRESHOLD = 0.05     # fraction of peak that defines "silence"
_SUSTAIN_DROP_THRESHOLD = 0.10  # fraction of peak below which sustain ends
_SUSTAIN_STDDEV_THRESHOLD = 0.02  # fraction of peak — flatness gate (validated by scratch/explore_adsr_fit.py)
_SUSTAIN_WINDOW_MS = 50.0
_MIN_SUSTAIN_MS = 30.0
_PLUCK_FALLBACK_FRACTION = 0.5  # for sustain-less notes, sustain marker = drop below 50% peak


class ADSRFit(BaseModel):
    attack_ms: float
    decay_ms: float
    sustain_level: float           # 0..1, normalized by peak_velocity
    release_ms: float
    sustain_start_idx: int         # index into envelope
    sustain_end_idx: int           # index into envelope (exclusive)


def _frame_to_ms(n_frames: int, envelope_sample_rate: float) -> float:
    return 1000.0 * n_frames / envelope_sample_rate


def fit_adsr(
    envelope: npt.NDArray[np.float32],
    envelope_sample_rate: float,
    peak_velocity: float,
) -> ADSRFit:
    if envelope.ndim != 1 or envelope.size == 0:
        raise ValueError("envelope must be a non-empty 1-D array")
    if peak_velocity <= 0:
        raise ValueError("peak_velocity must be > 0")

    peak = float(envelope.max())
    if peak <= 0:
        return ADSRFit(0.0, 0.0, 0.0, 0.0, 0, 0)

    peak_idx = int(np.argmax(envelope))

    # Attack: first index where envelope crosses _ATTACK_THRESHOLD * peak
    attack_thresh = _ATTACK_THRESHOLD * peak
    above = np.where(envelope[:peak_idx + 1] >= attack_thresh)[0]
    attack_start_idx = int(above[0]) if above.size > 0 else 0
    attack_frames = peak_idx - attack_start_idx
    attack_ms = _frame_to_ms(attack_frames, envelope_sample_rate)

    # Sustain region: slide window from peak forward; flat where stddev < threshold AND value above drop floor
    window_frames = max(1, int(round(_SUSTAIN_WINDOW_MS * envelope_sample_rate / 1000.0)))
    stddev_thresh = _SUSTAIN_STDDEV_THRESHOLD * peak
    drop_floor = _SUSTAIN_DROP_THRESHOLD * peak

    sustain_start_idx = peak_idx
    # Walk forward to find the first frame where stddev is below threshold AND value >= drop_floor
    found_start = False
    for i in range(peak_idx, envelope.size - window_frames):
        window = envelope[i:i + window_frames]
        if window.std() < stddev_thresh and window.mean() >= drop_floor:
            sustain_start_idx = i
            found_start = True
            break

    sustain_end_idx = sustain_start_idx
    if found_start:
        for i in range(sustain_start_idx, envelope.size - window_frames):
            window = envelope[i:i + window_frames]
            if window.std() >= stddev_thresh or window.mean() < drop_floor:
                sustain_end_idx = i
                break
        else:
            # Sustain runs to the end of the envelope. Exclusive end = envelope.size.
            sustain_end_idx = envelope.size

    sustain_duration_ms = _frame_to_ms(sustain_end_idx - sustain_start_idx, envelope_sample_rate)

    if not found_start or sustain_duration_ms < _MIN_SUSTAIN_MS:
        # Pluck fallback: locate where envelope drops below 50% of peak after the peak.
        # Early return for readability — none of the sustain-region math applies here.
        below = np.where(envelope[peak_idx:] < _PLUCK_FALLBACK_FRACTION * peak)[0]
        marker = peak_idx + int(below[0]) if below.size > 0 else envelope.size - 1
        decay_ms = _frame_to_ms(marker - peak_idx, envelope_sample_rate)
        return ADSRFit(
            attack_ms=attack_ms,
            decay_ms=decay_ms,
            sustain_level=0.0,
            release_ms=0.0,
            sustain_start_idx=marker,
            sustain_end_idx=marker,
        )

    sustain_level_raw = float(envelope[sustain_start_idx:sustain_end_idx].mean())
    sustain_level = float(np.clip(sustain_level_raw / peak_velocity, 0.0, 1.0))
    decay_ms = _frame_to_ms(sustain_start_idx - peak_idx, envelope_sample_rate)

    # Release: from sustain_end_idx until envelope < _RELEASE_THRESHOLD * peak
    release_thresh = _RELEASE_THRESHOLD * peak
    tail = envelope[sustain_end_idx:]
    drops = np.where(tail < release_thresh)[0]
    release_frames = int(drops[0]) if drops.size > 0 else tail.size
    release_ms = _frame_to_ms(release_frames, envelope_sample_rate)

    return ADSRFit(
        attack_ms=attack_ms,
        decay_ms=decay_ms,
        sustain_level=sustain_level,
        release_ms=release_ms,
        sustain_start_idx=sustain_start_idx,
        sustain_end_idx=sustain_end_idx,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_adsr_fit.py -v && uv run mypy src/`
Expected: 5 passed, mypy clean. If a tolerance test fails, tighten/loosen the constants in the algorithm — do not loosen the test, the tolerances were chosen to be honest.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/analysis/adsr_fit.py tests/test_adsr_fit.py
git commit -m "amplitude: add heuristic ADSR fitter with velocity normalization"
```

---

## Task 5: Sustain Isolation

Trim the audio to the sustain region identified by the ADSR fit. Operates in audio samples, not envelope frames — converts the envelope frame indices using the hop length.

**Rules:**
- Require ≥100 ms of sustain. If shorter, return `None` (caller will fall back to using the whole note unmodified).
- Otherwise return an audio slice of the sustain region.

**Files:**
- Create: `src/audio_analysis_mcp/analysis/sustain_isolation.py`
- Test: `tests/test_sustain_isolation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sustain_isolation.py`:

```python
import numpy as np

from audio_analysis_mcp.analysis.sustain_isolation import isolate_sustain


SR = 22050


def test_isolates_sustain_region():
    # 1-second signal; sustain region is envelope frames 200..600 with hop=5ms → audio samples 22050*1.0..22050*3.0
    audio = np.ones(SR * 4, dtype=np.float32)
    hop_length = int(0.005 * SR)
    sustain = isolate_sustain(
        audio,
        sample_rate=SR,
        sustain_start_idx=200,
        sustain_end_idx=600,
        envelope_hop_length=hop_length,
    )
    assert sustain is not None
    expected_samples = (600 - 200) * hop_length
    assert sustain.size == expected_samples


def test_returns_none_when_too_short():
    audio = np.ones(SR, dtype=np.float32)
    hop_length = int(0.005 * SR)
    # 10 frames * 5ms = 50ms — below 100ms minimum
    sustain = isolate_sustain(
        audio,
        sample_rate=SR,
        sustain_start_idx=100,
        sustain_end_idx=110,
        envelope_hop_length=hop_length,
    )
    assert sustain is None


def test_clips_to_audio_bounds():
    audio = np.ones(SR, dtype=np.float32)
    hop_length = int(0.005 * SR)
    # Request a slice that runs past end of audio — should clip
    sustain = isolate_sustain(
        audio,
        sample_rate=SR,
        sustain_start_idx=100,
        sustain_end_idx=10_000,
        envelope_hop_length=hop_length,
    )
    assert sustain is not None
    assert sustain.size <= audio.size
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sustain_isolation.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the isolator**

Create `src/audio_analysis_mcp/analysis/sustain_isolation.py`:

```python
import numpy as np
import numpy.typing as npt

_MIN_SUSTAIN_MS = 100.0


def isolate_sustain(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    sustain_start_idx: int,
    sustain_end_idx: int,
    envelope_hop_length: int,
) -> npt.NDArray[np.float32] | None:
    """Trim audio to the sustained region identified by ADSR fitting.

    Returns None if the sustain region is shorter than 100 ms (caller falls back
    to the unmodified note).
    """
    if sustain_end_idx <= sustain_start_idx:
        return None
    duration_ms = 1000.0 * (sustain_end_idx - sustain_start_idx) * envelope_hop_length / sample_rate
    if duration_ms < _MIN_SUSTAIN_MS:
        return None

    start = sustain_start_idx * envelope_hop_length
    end = sustain_end_idx * envelope_hop_length
    start = max(0, start)
    end = min(audio.size, end)
    if end <= start:
        return None
    return audio[start:end].astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sustain_isolation.py -v && uv run mypy src/`
Expected: 3 passed, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/analysis/sustain_isolation.py tests/test_sustain_isolation.py
git commit -m "amplitude: add sustain-region isolation with 100ms minimum"
```

---

## Task 6: Amplitude Orchestrator

Compose the four phases. Pure logic, takes audio + notes + workspace paths to write outputs to. Returns an `AmplitudeAnalyzeResult`.

**Behavior:**
- Run triage first. If REJECTED, write the envelope curve only and return early with `adsr=None, sustain_slice_path=None`.
- Otherwise extract envelope, fit ADSR (using the loudest note's velocity as `peak_velocity`), isolate sustain. If sustain too short, set `sustain_slice_path=None`.
- Always write the envelope curve as `.npy` (debug aid).
- Write the sustain slice as `.wav` if produced.

**Files:**
- Create: `src/audio_analysis_mcp/analysis/amplitude.py`
- Modify: `tests/test_amplitude.py` (extend with orchestrator tests)

- [ ] **Step 1: Extend the failing tests**

Append to `tests/test_amplitude.py`:

```python
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_analysis_mcp.analysis.amplitude import analyze_amplitude
from audio_analysis_mcp.schemas import NoteEvent


SR = 22050


def _adsr_signal() -> np.ndarray:
    # 20ms attack, 100ms decay to 0.6, 500ms sustain, 150ms release at 220Hz
    attack = np.linspace(0, 1, int(0.02 * SR), endpoint=False)
    decay = np.linspace(1.0, 0.6, int(0.10 * SR), endpoint=False)
    sustain = np.full(int(0.5 * SR), 0.6)
    release = np.linspace(0.6, 0.0, int(0.15 * SR), endpoint=True)
    env = np.concatenate([attack, decay, sustain, release])
    t = np.arange(env.size) / SR
    return (env * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _mono_note(duration_s: float) -> list[NoteEvent]:
    return [NoteEvent(
        start_time=0.0,
        end_time=duration_s,
        pitch_midi=57,
        amplitude=1.0,
        pitch_bends=None,
    )]


def test_orchestrator_monophonic_writes_outputs(tmp_path: Path):
    audio = _adsr_signal()
    notes = _mono_note(audio.size / SR)
    result = analyze_amplitude(
        audio=audio,
        sample_rate=SR,
        notes=notes,
        output_dir=tmp_path,
    )
    assert result.adsr_triage.value == "monophonic"
    assert result.adsr is not None
    assert Path(result.envelope_curve_path).exists()
    assert result.sustain_slice_path is not None
    assert Path(result.sustain_slice_path).exists()


def test_orchestrator_rejected_for_arpeggio(tmp_path: Path):
    audio = _adsr_signal()
    arpeggio_notes = [
        NoteEvent(start_time=i * 0.125, end_time=i * 0.125 + 0.1,
                  pitch_midi=60 + i, amplitude=0.8, pitch_bends=None)
        for i in range(8)
    ]
    result = analyze_amplitude(
        audio=audio,
        sample_rate=SR,
        notes=arpeggio_notes,
        output_dir=tmp_path,
    )
    assert result.adsr_triage.value == "rejected"
    assert result.adsr is None
    assert result.sustain_slice_path is None
    assert Path(result.envelope_curve_path).exists()


def test_orchestrator_recovers_known_adsr(tmp_path: Path):
    audio = _adsr_signal()
    notes = _mono_note(audio.size / SR)
    result = analyze_amplitude(
        audio=audio,
        sample_rate=SR,
        notes=notes,
        output_dir=tmp_path,
    )
    assert result.adsr is not None
    assert abs(result.adsr.attack_ms - 20.0) < 15.0
    assert abs(result.adsr.sustain_level - 0.6) < 0.10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_amplitude.py -v`
Expected: existing schema tests pass; new orchestrator tests fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the orchestrator**

Create `src/audio_analysis_mcp/analysis/amplitude.py`:

```python
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

from audio_analysis_mcp.analysis.adsr_triage import classify_adsr_triage
from audio_analysis_mcp.analysis.envelope import extract_rms_envelope
from audio_analysis_mcp.analysis.adsr_fit import fit_adsr
from audio_analysis_mcp.analysis.sustain_isolation import isolate_sustain
from audio_analysis_mcp.schemas import (
    ADSREstimate,
    AmplitudeAnalyzeResult,
    AmplitudeTriage,
    NoteEvent,
)


def analyze_amplitude(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    notes: list[NoteEvent],
    output_dir: Path,
) -> AmplitudeAnalyzeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    envelope_path = output_dir / "envelope.npy"
    sustain_path = output_dir / "sustain.wav"

    triage = classify_adsr_triage(notes)
    env_result = extract_rms_envelope(audio, sample_rate=sample_rate)
    np.save(envelope_path, env_result.envelope)

    if triage == AmplitudeTriage.REJECTED:
        return AmplitudeAnalyzeResult(
            adsr_triage=triage,
            adsr=None,
            envelope_curve_path=str(envelope_path),
            sustain_slice_path=None,
        )

    peak_velocity = max((n.amplitude for n in notes), default=1.0)
    fit = fit_adsr(
        env_result.envelope,
        envelope_sample_rate=env_result.envelope_sample_rate,
        peak_velocity=peak_velocity,
    )
    adsr = ADSREstimate(
        attack_ms=fit.attack_ms,
        decay_ms=fit.decay_ms,
        sustain_level=fit.sustain_level,
        release_ms=fit.release_ms,
    )

    sustain = isolate_sustain(
        audio,
        sample_rate=sample_rate,
        sustain_start_idx=fit.sustain_start_idx,
        sustain_end_idx=fit.sustain_end_idx,
        envelope_hop_length=env_result.hop_length,
    )
    sustain_slice_path: str | None = None
    if sustain is not None:
        sf.write(sustain_path, sustain, sample_rate)
        sustain_slice_path = str(sustain_path)

    return AmplitudeAnalyzeResult(
        adsr_triage=triage,
        adsr=adsr,
        envelope_curve_path=str(envelope_path),
        sustain_slice_path=sustain_slice_path,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_amplitude.py -v && uv run mypy src/`
Expected: all amplitude tests pass, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/analysis/amplitude.py tests/test_amplitude.py
git commit -m "amplitude: add orchestrator composing triage, envelope, fit, isolation"
```

---

## Task 7: MCP Tool

Wrap the orchestrator as an MCP tool. Mirrors the `note_triage` tool pattern: takes paths, resolves the workspace job context, writes outputs to a job-scoped amplitude directory, returns JSON-serialized result.

**Files:**
- Create: `src/audio_analysis_mcp/tools/amplitude_analyze.py`
- Modify: `src/audio_analysis_mcp/workspace.py` (add `job_amplitude_dir`)
- Modify: `src/audio_analysis_mcp/server.py` (import the new tool module)
- Test: `tests/test_mcp_tools.py` (extend) and a focused tool test

- [ ] **Step 1: Write failing tests**

Append to `tests/test_amplitude.py`:

```python
import json

from audio_analysis_mcp.workspace import Workspace


def test_workspace_job_amplitude_dir(tmp_path: Path):
    ws = Workspace(root=tmp_path)
    d = ws.job_amplitude_dir("myjob", stem="other", preset="htdemucs")
    assert d.exists()
    assert d.relative_to(tmp_path) == Path("jobs/myjob/amplitude/other_htdemucs")


def test_mcp_tool_runs_end_to_end(tmp_path: Path, monkeypatch):
    from audio_analysis_mcp import server as server_module
    from audio_analysis_mcp.tools.amplitude_analyze import amplitude_analyze
    import audio_analysis_mcp.tools.amplitude_analyze as tool_module

    ws = Workspace(root=tmp_path)
    monkeypatch.setattr(server_module, "get_workspace", lambda: ws)
    monkeypatch.setattr(tool_module, "get_workspace", lambda: ws)

    job_dir = ws.job_stems_dir("myjob", "htdemucs")
    audio_path = job_dir / "other.wav"
    audio = _adsr_signal()
    sf.write(audio_path, audio, SR)

    notes = _mono_note(audio.size / SR)
    notes_path = ws.job_dir("myjob") / "notes.json"
    notes_path.write_text(json.dumps([n.model_dump() for n in notes]))

    result_json = amplitude_analyze(
        audio_path=str(audio_path),
        notes_path=str(notes_path),
    )
    payload = json.loads(result_json)
    assert payload["adsr_triage"] == "monophonic"
    assert payload["adsr"]["attack_ms"] > 0
    assert Path(payload["envelope_curve_path"]).exists()
    assert payload["sustain_slice_path"] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_amplitude.py -v`
Expected: new tests fail with `AttributeError: 'Workspace' object has no attribute 'job_amplitude_dir'` and `ModuleNotFoundError`.

- [ ] **Step 3: Add the workspace helper**

Add to `src/audio_analysis_mcp/workspace.py` next to the other `job_*_dir` methods:

```python
    def job_amplitude_dir(self, job_name: str, stem: str, preset: str) -> Path:
        return self._ensure(f"jobs/{job_name}/amplitude/{stem}_{preset}")
```

- [ ] **Step 4: Implement the MCP tool**

Create `src/audio_analysis_mcp/tools/amplitude_analyze.py`:

```python
from pathlib import Path

import soundfile as sf
from pydantic import TypeAdapter

from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.workspace import resolve_job_context
from audio_analysis_mcp.analysis.amplitude import analyze_amplitude
from audio_analysis_mcp.schemas import NoteEvent


@mcp.tool()
def amplitude_analyze(
    audio_path: str,
    notes_path: str,
) -> str:
    """Analyze amplitude envelope: triage, ADSR fit, and sustain isolation.

    Inputs:
      audio_path:  WAV file at jobs/<job>/stems/<preset>/<stem>.wav
      notes_path:  JSON list of NoteEvent from note_transcribe

    Writes envelope curve (.npy) and isolated sustain slice (.wav) into
    jobs/<job>/amplitude/<stem>_<preset>/. Returns JSON-serialized
    AmplitudeAnalyzeResult.
    """
    ws = get_workspace()
    ctx = resolve_job_context(audio_path, ws)
    if ctx.stem is None or ctx.preset is None:
        raise ValueError(
            f"Expected a stem file (jobs/<job>/stems/<preset>/<stem>.wav), got: {audio_path}"
        )

    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("float32")

    adapter = TypeAdapter(list[NoteEvent])
    notes = adapter.validate_json(Path(notes_path).read_text())

    output_dir = ws.job_amplitude_dir(ctx.job_name, ctx.stem, ctx.preset)
    result = analyze_amplitude(
        audio=audio,
        sample_rate=int(sample_rate),
        notes=notes,
        output_dir=output_dir,
    )
    return result.model_dump_json(indent=2)
```

- [ ] **Step 5: Register the tool in the server**

Open `src/audio_analysis_mcp/server.py`, find the block of `from audio_analysis_mcp.tools import ...` lines (or `import audio_analysis_mcp.tools.<name>` lines), and add:

```python
import audio_analysis_mcp.tools.amplitude_analyze  # noqa: F401  (registers @mcp.tool)
```

(Match whichever import style the file already uses for the other tools — do not duplicate or reorder existing imports.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_amplitude.py -v && uv run pytest -m "not slow" && uv run mypy src/`
Expected: amplitude tests pass; full non-slow suite green; mypy clean.

- [ ] **Step 7: Commit**

```bash
git add \
  src/audio_analysis_mcp/tools/amplitude_analyze.py \
  src/audio_analysis_mcp/workspace.py \
  src/audio_analysis_mcp/server.py \
  tests/test_amplitude.py
git commit -m "amplitude: add amplitude_analyze MCP tool and workspace dir"
```

---

## Task 8: SignalFlow Integration Test (slow)

One realistic check: render a synth note with known ADSR via SignalFlow, run it through the orchestrator, verify the recovered ADSR is within tolerance. Marked `@pytest.mark.slow` and excluded from the default CI run per repo convention.

**Files:**
- Modify: `pyproject.toml` (add `signalflow` to dev deps)
- Create: `tests/test_amplitude_signalflow.py`

- [ ] **Step 1: Add SignalFlow as a dev dependency**

In `pyproject.toml`, locate the dev-dependencies section (in this repo it appears under `[tool.uv]` or a `dependency-groups.dev` block — match the existing style). Add:

```toml
"signalflow>=0.4.0",
```

Then run:

```bash
uv sync --dev
```

Expected: SignalFlow installs without error. If it fails on macOS, the fallback is `brew install portaudio` first, then re-run `uv sync --dev`.

- [ ] **Step 2: Write the slow test**

Create `tests/test_amplitude_signalflow.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_analysis_mcp.analysis.amplitude import analyze_amplitude
from audio_analysis_mcp.schemas import NoteEvent


@pytest.mark.slow
def test_signalflow_rendered_adsr_recovered(tmp_path: Path):
    from signalflow import AudioGraph, SineOscillator, ASREnvelope

    sr = 44100
    duration = 1.5  # seconds
    attack_s, sustain_s, release_s = 0.020, 0.600, 0.200

    graph = AudioGraph(output_device=None, start=False)
    osc = SineOscillator(frequency=440.0)
    env = ASREnvelope(attack=attack_s, sustain=sustain_s, release=release_s)
    out = osc * env

    n_samples = int(sr * duration)
    buffer = graph.render_to_buffer(out, num_frames=n_samples)
    audio = np.asarray(buffer.data[0], dtype=np.float32)

    audio_path = tmp_path / "signalflow_note.wav"
    sf.write(audio_path, audio, sr)

    notes = [NoteEvent(
        start_time=0.0,
        end_time=duration,
        pitch_midi=69,
        amplitude=1.0,
        pitch_bends=None,
    )]

    result = analyze_amplitude(
        audio=audio,
        sample_rate=sr,
        notes=notes,
        output_dir=tmp_path,
    )

    assert result.adsr_triage.value == "monophonic"
    assert result.adsr is not None
    # ASR envelope has no decay — fit should produce small decay_ms
    assert abs(result.adsr.attack_ms - 20.0) < 25.0
    assert abs(result.adsr.release_ms - 200.0) < 60.0
    assert result.adsr.sustain_level > 0.5
    assert result.sustain_slice_path is not None
```

(If SignalFlow's API names differ in the installed version — `ASREnvelope` vs `ASR` vs `Envelope` — update the import and constructor to match. To confirm available names, write a small script at `scratch/explore_signalflow.py` that imports `signalflow` and prints `dir(signalflow)`, then run it via `uv run python scratch/explore_signalflow.py`.)

- [ ] **Step 3: Run the slow test explicitly**

Run: `uv run pytest tests/test_amplitude_signalflow.py -v -m slow`
Expected: PASS. If tolerances are breached, adjust the test tolerances (these are realistic-synth-audio bounds, not algorithmic correctness bounds — they should be looser than the numpy-fixture tests in Task 4).

Then verify it does NOT run in the default suite:

Run: `uv run pytest -m "not slow" -v`
Expected: the SignalFlow test is skipped/not collected.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock tests/test_amplitude_signalflow.py
git commit -m "amplitude: add SignalFlow-rendered ADSR integration test (slow)"
```

---

## Task 9: Wire Into Repo Docs

Two small doc touches so future readers know the amplitude expert exists.

**Files:**
- Modify: `audio-analysis-mcp/audio-pipeline-plan.md`
- Modify: `audio-analysis-mcp/CLAUDE.md`

- [ ] **Step 1: Add the tool to the pipeline plan**

Open `audio-analysis-mcp/audio-pipeline-plan.md` and, in the file structure tree (around the `tools/` and `analysis/` listings), add:

```
        amplitude_analyze.py           # ADSR estimate + sustain isolation (amplitude expert)
```

and

```
        adsr_triage.py                 # Onset-density triage for ADSR analyzability
        envelope.py                    # RMS sliding-window envelope
        adsr_fit.py                    # Heuristic four-segment ADSR fit
        sustain_isolation.py           # Trim audio to sustain region
        amplitude.py                   # Orchestrator (triage → envelope → fit → isolate)
```

Match the indentation of the surrounding entries.

- [ ] **Step 2: Add a one-liner to CLAUDE.md tool list**

In `audio-analysis-mcp/CLAUDE.md`, the architecture section mentions tool registration. No specific tool list to update there — skip if there isn't one. (If `CLAUDE.md` has grown a tool list since this plan was written, add `amplitude_analyze` to it.)

- [ ] **Step 3: Final verification**

Run:
```bash
uv run pytest -m "not slow" -v
uv run mypy src/
```
Expected: full non-slow suite green; mypy clean.

- [ ] **Step 4: Commit**

```bash
git add audio-analysis-mcp/audio-pipeline-plan.md
git commit -m "amplitude: document amplitude_analyze in pipeline plan"
```

---

## Out of Scope (v2 and Beyond)

Explicitly deferred — do not add to this plan:

- Multi-method envelope-extraction comparison (Hilbert, peak follower) — only RMS in v1.
- Per-note arpeggio segmentation — arpeggios are rejected in v1.
- Acoustic-instrument passthrough heuristic (piano/Rhodes intrinsic decay) — open question in research plan, depends on engine pre-classifier.
- Modulation-friendly audio output — research plan explicitly punts this until the modulation expert decides what it needs.
- Non-linear velocity curves per engine — v1 assumes linear.
- Synthetic dataset generation script for cross-expert evaluation — belongs to a shared dataset plan, not this expert.

---

## Self-Review Notes

- Spec coverage: research plan's three responsibilities (ADSR estimate, sustain slice for tone-gen, no modulation output) all covered. Triage Phase 0 → Task 2. Phase 1 envelope → Task 3 (RMS only, comparison deferred per scope). Phase 2 fit → Task 4 (with velocity normalization per research plan inputs). Phase 3 sustain isolation → Task 5. Orchestrator + MCP tool → Tasks 6–7. SignalFlow validation → Task 8.
- Type consistency: `ADSRFit` (analysis-internal pydantic with extra index fields) vs `ADSREstimate` (canonical pydantic, just the four ADSR values) are distinct on purpose — the orchestrator translates. Both pydantic for codebase consistency. `AmplitudeAnalyzeResult` matches the result dict shape from the research plan, with `triage` renamed to `adsr_triage` per the latest research-plan revision.
- No placeholders: every task has runnable test code, runnable implementation code, and exact commands.