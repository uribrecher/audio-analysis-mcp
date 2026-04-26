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