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