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