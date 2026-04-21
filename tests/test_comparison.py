from pathlib import Path
from audio_analysis_mcp.analysis.comparison import compare_audio


def test_identical_signals_low_distance(sine_440_wav: Path):
    result = compare_audio(str(sine_440_wav), str(sine_440_wav))
    assert result.mel_spectrogram_distance < 0.01


def test_different_frequencies_high_distance(sine_440_wav: Path, sine_880_wav: Path):
    result = compare_audio(str(sine_440_wav), str(sine_880_wav))
    assert result.mel_spectrogram_distance > 0.1


def test_sine_vs_square_different_timbre(sine_440_wav: Path, square_440_wav: Path):
    result = compare_audio(str(sine_440_wav), str(square_440_wav))
    assert result.mel_spectrogram_distance > 0.05
    # Square wave has more high-frequency energy
    high_band = next(b for b in result.band_diffs if "high" in b.band)
    assert high_band.diff_db != 0


def test_band_diffs_present(sine_440_wav: Path, sine_880_wav: Path):
    result = compare_audio(str(sine_440_wav), str(sine_880_wav))
    assert len(result.band_diffs) >= 3
    band_names = {b.band for b in result.band_diffs}
    assert any("low" in b for b in band_names)
    assert any("mid" in b for b in band_names)
    assert any("high" in b for b in band_names)


def test_clap_is_none(sine_440_wav: Path):
    """CLAP returns None (deferred to Phase 2)."""
    result = compare_audio(str(sine_440_wav), str(sine_440_wav))
    assert result.clap_cosine_similarity is None
