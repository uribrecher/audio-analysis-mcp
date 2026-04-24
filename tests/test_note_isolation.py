"""Tests for analysis.note_isolation — STFT time-frequency masking."""
from pathlib import Path

import numpy as np
import librosa
import pytest

from audio_analysis_mcp.analysis.note_isolation import isolate_note


def test_single_sine_in_box(sine_440_wav: Path):
    """A 440Hz sine isolated with a box that includes 440Hz → high energy output."""
    y_isolated, sr = isolate_note(
        audio_path=str(sine_440_wav),
        start_time=0.0,
        end_time=1.0,
        start_freq=400.0,
        end_freq=500.0,
    )
    rms = float(np.sqrt(np.mean(y_isolated**2)))
    assert rms > 0.1  # significant energy preserved


def test_isolate_one_of_two_freqs(two_freq_wav: Path):
    """440Hz + 1000Hz signal, isolate 440Hz → 1000Hz attenuated."""
    y_isolated, sr = isolate_note(
        audio_path=str(two_freq_wav),
        start_time=0.0,
        end_time=1.0,
        start_freq=400.0,
        end_freq=500.0,
    )
    # Check that 1000Hz is strongly attenuated
    S = np.abs(librosa.stft(y_isolated))
    freqs = librosa.fft_frequencies(sr=sr)
    bin_440 = int(np.argmin(np.abs(freqs - 440)))
    bin_1000 = int(np.argmin(np.abs(freqs - 1000)))
    energy_440 = float(np.mean(S[bin_440, :]))
    energy_1000 = float(np.mean(S[bin_1000, :]))
    assert energy_440 > energy_1000 * 5  # 440Hz should dominate


def test_time_window_slicing(sequential_notes_wav: Path):
    """3-second audio (3 notes), isolate second 1-2s → duration ~1s."""
    y_isolated, sr = isolate_note(
        audio_path=str(sequential_notes_wav),
        start_time=1.0,
        end_time=2.0,
        start_freq=20.0,
        end_freq=10000.0,
    )
    duration = len(y_isolated) / sr
    assert duration == pytest.approx(1.0, abs=0.05)


def test_output_duration_matches_window(sine_440_wav: Path):
    """Requested window of 0.5s → output is ~0.5s."""
    y_isolated, sr = isolate_note(
        audio_path=str(sine_440_wav),
        start_time=0.2,
        end_time=0.7,
        start_freq=20.0,
        end_freq=10000.0,
    )
    duration = len(y_isolated) / sr
    assert duration == pytest.approx(0.5, abs=0.05)


def test_invalid_time_range(sine_440_wav: Path):
    """start_time >= end_time → ValueError."""
    with pytest.raises(ValueError):
        isolate_note(
            audio_path=str(sine_440_wav),
            start_time=0.5,
            end_time=0.5,
            start_freq=100.0,
            end_freq=1000.0,
        )


def test_invalid_freq_range(sine_440_wav: Path):
    """start_freq >= end_freq → ValueError."""
    with pytest.raises(ValueError):
        isolate_note(
            audio_path=str(sine_440_wav),
            start_time=0.0,
            end_time=1.0,
            start_freq=1000.0,
            end_freq=100.0,
        )
