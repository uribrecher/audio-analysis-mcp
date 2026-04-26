from audio_analysis_mcp.schemas import (
    ADSREstimate,
    AmplitudeCandidate,
    AmplitudeAnalyzeResult,
)


def _candidate(idx: int = 0) -> AmplitudeCandidate:
    return AmplitudeCandidate(
        cluster_index=idx,
        kind="single",
        score=2.5,
        adsr=ADSREstimate(attack_ms=20.0, decay_ms=100.0, sustain_level=0.6, release_ms=150.0),
        envelope_curve_path=f"/tmp/c{idx}/envelope.npy",
        sustain_slice_path=f"/tmp/c{idx}/sustain.wav",
    )


def test_amplitude_candidate_single():
    c = _candidate()
    assert c.kind == "single"
    assert c.adsr.sustain_level == 0.6


def test_amplitude_candidate_chord_no_sustain():
    c = AmplitudeCandidate(
        cluster_index=3, kind="chord", score=1.8,
        adsr=ADSREstimate(attack_ms=15.0, decay_ms=60.0, sustain_level=0.4, release_ms=80.0),
        envelope_curve_path="/tmp/c3/envelope.npy",
        sustain_slice_path=None,
    )
    assert c.sustain_slice_path is None


def test_amplitude_analyze_result_consistent():
    result = AmplitudeAnalyzeResult(
        candidates=[_candidate(0), _candidate(1)],
        consensus_adsr=ADSREstimate(attack_ms=20.0, decay_ms=100.0, sustain_level=0.6, release_ms=150.0),
        divergence_score=0.05,
        is_consistent=True,
        rejected_reason=None,
    )
    assert result.is_consistent
    assert result.consensus_adsr is not None


def test_amplitude_analyze_result_rejected():
    result = AmplitudeAnalyzeResult(
        candidates=[],
        consensus_adsr=None,
        divergence_score=0.0,
        is_consistent=False,
        rejected_reason="no candidates with usable sustain",
    )
    assert result.candidates == []
    assert result.consensus_adsr is None
    assert result.rejected_reason == "no candidates with usable sustain"
