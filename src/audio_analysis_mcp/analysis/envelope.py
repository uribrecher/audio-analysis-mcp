import librosa
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

_DEFAULT_FRAME_LENGTH_MS = 20.0
_DEFAULT_HOP_LENGTH_MS = 5.0


class EnvelopeResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    envelope: npt.NDArray[np.float32]            # 1-D, RMS values
    envelope_sample_rate: float     # frames per second
    hop_length: int                 # samples between frames
    frame_length: int               # samples per RMS window


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