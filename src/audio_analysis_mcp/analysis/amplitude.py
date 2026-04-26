from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

from audio_analysis_mcp.analysis.envelope import extract_rms_envelope
from audio_analysis_mcp.analysis.adsr_fit import fit_adsr
from audio_analysis_mcp.analysis.sustain_isolation import isolate_sustain
from audio_analysis_mcp.schemas import (
    ADSREstimate,
    AmplitudeAnalyzeResult,
    AmplitudeCandidate,
    NoteTriageFileData,
)


_DIVERGENCE_THRESHOLD = 0.15  # validated in scratch/explore_consensus.py
_REJECTED_REASON = "no candidates with usable sustain"


def analyze_amplitude(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    triage_path: Path,
    output_dir: Path,
) -> AmplitudeAnalyzeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_data = NoteTriageFileData.model_validate_json(Path(triage_path).read_text())

    candidates: list[AmplitudeCandidate] = []
    for idx, cluster in enumerate(file_data.candidates):
        if cluster.kind == "arpeggio":
            continue  # defensive — triage filters these, but skip if any leak through

        cluster_dir = output_dir / f"cluster_{idx:02d}_{cluster.kind}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        start_sample = max(0, int(cluster.start_time * sample_rate))
        end_sample = min(audio.size, int(cluster.end_time * sample_rate))
        if end_sample <= start_sample:
            continue
        cluster_audio = audio[start_sample:end_sample]

        env_result = extract_rms_envelope(cluster_audio, sample_rate=sample_rate)
        envelope_path = cluster_dir / "envelope.npy"
        np.save(envelope_path, env_result.envelope)

        fit = fit_adsr(env_result.envelope, envelope_sample_rate=env_result.envelope_sample_rate)
        if fit.sustain_level == 0.0:
            continue  # pluck — skip per scope decision

        adsr = ADSREstimate(
            attack_ms=fit.attack_ms,
            decay_ms=fit.decay_ms,
            sustain_level=fit.sustain_level,
            release_ms=fit.release_ms,
        )

        sustain = isolate_sustain(
            cluster_audio,
            sample_rate=sample_rate,
            sustain_start_idx=fit.sustain_start_idx,
            sustain_end_idx=fit.sustain_end_idx,
            envelope_hop_length=env_result.hop_length,
        )
        sustain_path: str | None = None
        if sustain is not None:
            slice_path = cluster_dir / "sustain.wav"
            sf.write(slice_path, sustain, sample_rate)
            sustain_path = str(slice_path)

        # Debug-only sustain duration; NOT part of the ADSR profile or the
        # divergence metric — see plan.
        sustain_duration_ms = (
            (fit.sustain_end_idx - fit.sustain_start_idx)
            * env_result.hop_length
            * 1000.0
            / sample_rate
        )

        candidates.append(AmplitudeCandidate(
            cluster_index=idx,
            kind=cluster.kind,
            score=cluster.score,
            adsr=adsr,
            sustain_duration_ms=round(sustain_duration_ms, 2),
            envelope_curve_path=str(envelope_path),
            sustain_slice_path=sustain_path,
        ))

    if not candidates:
        return AmplitudeAnalyzeResult(
            candidates=[], consensus_adsr=None,
            divergence_score=0.0, is_consistent=False,
            rejected_reason=_REJECTED_REASON,
        )

    vectors = np.array([
        [c.adsr.attack_ms / 1000.0,
         c.adsr.decay_ms / 1000.0,
         c.adsr.sustain_level,
         c.adsr.release_ms / 1000.0]
        for c in candidates
    ])

    if len(vectors) == 1:
        max_dist = 0.0
    else:
        diffs = vectors[:, None, :] - vectors[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        max_dist = float(dists.max())

    is_consistent = max_dist < _DIVERGENCE_THRESHOLD
    consensus: ADSREstimate | None = None
    if is_consistent:
        mean_vec = vectors.mean(axis=0)
        consensus = ADSREstimate(
            attack_ms=float(mean_vec[0] * 1000.0),
            decay_ms=float(mean_vec[1] * 1000.0),
            sustain_level=float(mean_vec[2]),
            release_ms=float(mean_vec[3] * 1000.0),
        )

    return AmplitudeAnalyzeResult(
        candidates=candidates,
        consensus_adsr=consensus,
        divergence_score=round(max_dist, 4),
        is_consistent=is_consistent,
        rejected_reason=None,
    )
