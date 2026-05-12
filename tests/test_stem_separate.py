from pathlib import Path
from unittest.mock import patch, MagicMock, call
import torch
import pytest
from audio_analysis_mcp.tools.stem_separate import stem_separate_impl, PRESETS

SOURCES = ["drums", "bass", "other", "vocals"]


def _make_mock_model():
    model = MagicMock()
    model.sources = SOURCES
    model.samplerate = 44100
    model.audio_channels = 2
    return model


def _make_mock_audio_file():
    af = MagicMock()
    af.read.return_value = torch.zeros(2, 44100)
    return af


def _fake_save(tensor, path, samplerate):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("fake")


def test_stem_separate_returns_all_stems(sine_440_wav: Path, tmp_path: Path):
    model = _make_mock_model()
    sources_tensor = torch.zeros(1, 4, 2, 44100)

    with patch("audio_analysis_mcp.tools.stem_separate.get_demucs_model", return_value=model), \
         patch("audio_analysis_mcp.tools.stem_separate.apply_model", return_value=sources_tensor), \
         patch("audio_analysis_mcp.tools.stem_separate.AudioFile", return_value=_make_mock_audio_file()), \
         patch("audio_analysis_mcp.tools.stem_separate.save_audio", side_effect=_fake_save):
        result = stem_separate_impl(str(sine_440_wav), tmp_path, preset_name="medium")

    assert len(result.stems) == 4
    assert {s.stem for s in result.stems} == set(SOURCES)
    assert result.cached is False
    assert result.preset == "medium"
    assert result.model == "htdemucs_6s"


def test_stem_separate_uses_cache(sine_440_wav: Path, tmp_path: Path):
    model = _make_mock_model()
    sources_tensor = torch.zeros(1, 4, 2, 44100)

    with patch("audio_analysis_mcp.tools.stem_separate.get_demucs_model", return_value=model) as mock_get, \
         patch("audio_analysis_mcp.tools.stem_separate.apply_model", return_value=sources_tensor) as mock_apply, \
         patch("audio_analysis_mcp.tools.stem_separate.AudioFile", return_value=_make_mock_audio_file()), \
         patch("audio_analysis_mcp.tools.stem_separate.save_audio", side_effect=_fake_save):
        stem_separate_impl(str(sine_440_wav), tmp_path, preset_name="fast")
        result2 = stem_separate_impl(str(sine_440_wav), tmp_path, preset_name="fast")

    assert result2.cached is True
    assert result2.preset == "fast"
    # Model loaded only on first call; cache hit avoids get_model on second call
    assert mock_get.call_count == 1
    assert mock_apply.call_count == 1


def test_stem_separate_failure_raises(sine_440_wav: Path, tmp_path: Path):
    model = _make_mock_model()

    with patch("audio_analysis_mcp.tools.stem_separate.get_demucs_model", return_value=model), \
         patch("audio_analysis_mcp.tools.stem_separate.apply_model", side_effect=RuntimeError("OOM")), \
         patch("audio_analysis_mcp.tools.stem_separate.AudioFile", return_value=_make_mock_audio_file()):
        with pytest.raises(RuntimeError, match="OOM"):
            stem_separate_impl(str(sine_440_wav), tmp_path, preset_name="medium")


def test_stem_separate_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not found"):
        stem_separate_impl("/nonexistent/file.wav", tmp_path, preset_name="medium")


def test_invalid_preset_raises(sine_440_wav: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown preset"):
        stem_separate_impl(str(sine_440_wav), tmp_path, preset_name="ultra")


def test_preset_passes_correct_args_to_apply_model(sine_440_wav: Path, tmp_path: Path):
    """Each preset should pass its specific shifts/overlap/progress to apply_model."""
    for preset_name, preset in PRESETS.items():
        model = _make_mock_model()
        sources_tensor = torch.zeros(1, 4, 2, 44100)

        with patch("audio_analysis_mcp.tools.stem_separate.get_demucs_model", return_value=model), \
             patch("audio_analysis_mcp.tools.stem_separate.apply_model", return_value=sources_tensor) as mock_apply, \
             patch("audio_analysis_mcp.tools.stem_separate.AudioFile", return_value=_make_mock_audio_file()), \
             patch("audio_analysis_mcp.tools.stem_separate.save_audio", side_effect=_fake_save):
            stem_separate_impl(str(sine_440_wav), tmp_path / preset_name, preset_name=preset_name)

        _, kwargs = mock_apply.call_args
        assert kwargs["shifts"] == preset.shifts, f"{preset_name}: wrong shifts"
        assert kwargs["overlap"] == preset.overlap, f"{preset_name}: wrong overlap"
        assert kwargs["progress"] is True, f"{preset_name}: progress not enabled"


def test_progress_monotonic_across_shifts_with_same_total(sine_440_wav: Path, tmp_path: Path):
    """Three back-to-back tqdm bars on the same audio share a `total`. The
    progress sink must still detect each new bar (via `current` resetting)
    and emit a monotonically non-decreasing fraction — the previous
    implementation looked for `total` changes and stuck at ~33% across all
    three shifts.
    """
    model = _make_mock_model()
    # htdemucs-shaped: 1 sub-model (model.models attr only present on bag-of-N
    # variants; without it we fall back to 1). Combined with shifts=3 → 3 runs.
    del model.models
    sources_tensor = torch.zeros(1, 4, 2, 44100)

    captured_fractions: list[float] = []

    def progress(stage: str, fraction: float, detail: str | None) -> None:
        captured_fractions.append(fraction)

    TOTAL = 403  # same total for every bar — the case that broke before
    def fake_apply_model(*args, **kwargs):
        # Drive the installed sink with three bars at the same `total`.
        from audio_analysis_mcp import _demucs_progress
        sink = getattr(_demucs_progress._progress_sink, "sink", None)
        assert sink is not None, "stem_separate_impl should install a sink before apply_model"
        for _run in range(3):
            for current in (10, 100, 200, TOTAL):
                sink(current, TOTAL)
        return sources_tensor

    with patch("audio_analysis_mcp.tools.stem_separate.get_demucs_model", return_value=model), \
         patch("audio_analysis_mcp.tools.stem_separate.apply_model", side_effect=fake_apply_model), \
         patch("audio_analysis_mcp.tools.stem_separate.AudioFile", return_value=_make_mock_audio_file()), \
         patch("audio_analysis_mcp.tools.stem_separate.save_audio", side_effect=_fake_save):
        stem_separate_impl(
            str(sine_440_wav), tmp_path / "medium", preset_name="medium", progress=progress,
        )

    # Pull only the "run" stage emissions (load_model / write / done land in
    # different buckets and don't share the monotonic invariant we're testing).
    run_fractions = captured_fractions[1:-2]  # drop the load_model / write / done bookends
    # Monotonic non-decreasing across the whole run phase.
    for prev, nxt in zip(run_fractions, run_fractions[1:]):
        assert nxt + 1e-9 >= prev, f"progress regressed: {prev:.4f} -> {nxt:.4f}"
    # The last run-phase emission should reach near the 0.95 ceiling, not get
    # stuck at ~0.33 (which is the symptom the buggy implementation produced).
    assert run_fractions[-1] > 0.9, (
        f"progress topped out at {run_fractions[-1]:.4f}; counter never advanced across shifts"
    )


def test_different_presets_use_separate_cache(sine_440_wav: Path, tmp_path: Path):
    """Fast and medium presets should not share cache entries.

    In the job-centric workspace, each preset gets its own directory
    (handled by job_stems_dir). stem_separate_impl uses stems_dir directly
    as the cache directory, so different dirs = no cache sharing.
    """
    model = _make_mock_model()
    sources_tensor = torch.zeros(1, 4, 2, 44100)

    with patch("audio_analysis_mcp.tools.stem_separate.get_demucs_model", return_value=model), \
         patch("audio_analysis_mcp.tools.stem_separate.apply_model", return_value=sources_tensor) as mock_apply, \
         patch("audio_analysis_mcp.tools.stem_separate.AudioFile", return_value=_make_mock_audio_file()), \
         patch("audio_analysis_mcp.tools.stem_separate.save_audio", side_effect=_fake_save):
        result_fast = stem_separate_impl(str(sine_440_wav), tmp_path / "fast", preset_name="fast")
        result_medium = stem_separate_impl(str(sine_440_wav), tmp_path / "medium", preset_name="medium")

    # Both should be uncached since they use different directories
    assert result_fast.cached is False
    assert result_medium.cached is False
    assert mock_apply.call_count == 2
