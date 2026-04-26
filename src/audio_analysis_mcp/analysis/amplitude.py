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