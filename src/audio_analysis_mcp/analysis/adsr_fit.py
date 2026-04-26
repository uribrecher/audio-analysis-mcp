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
    sustain_level: float           # 0..1, ratio of sustain RMS to envelope peak (naturally velocity-invariant)
    release_ms: float
    sustain_start_idx: int         # index into envelope
    sustain_end_idx: int           # index into envelope (exclusive)


def _frame_to_ms(n_frames: int, envelope_sample_rate: float) -> float:
    return 1000.0 * n_frames / envelope_sample_rate


def fit_adsr(
    envelope: npt.NDArray[np.float32],
    envelope_sample_rate: float,
) -> ADSRFit:
    if envelope.ndim != 1 or envelope.size == 0:
        raise ValueError("envelope must be a non-empty 1-D array")

    peak = float(envelope.max())
    if peak <= 0:
        return ADSRFit(
            attack_ms=0.0, decay_ms=0.0, sustain_level=0.0, release_ms=0.0,
            sustain_start_idx=0, sustain_end_idx=0,
        )

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
    sustain_level = float(np.clip(sustain_level_raw / peak, 0.0, 1.0))
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
