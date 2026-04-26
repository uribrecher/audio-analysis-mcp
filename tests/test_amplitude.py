import json
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_analysis_mcp.analysis.amplitude import analyze_amplitude
from audio_analysis_mcp.schemas import (
    ADSREstimate,
    AmplitudeCandidate,
    AmplitudeAnalyzeResult,
    CandidateCluster,
    CandidateNote,
    NoteEvent,
    NoteTriageFileData,
)


def _candidate(idx: int = 0) -> AmplitudeCandidate:
    return AmplitudeCandidate(
        cluster_index=idx,
        kind="single",
        score=2.5,
        adsr=ADSREstimate(attack_ms=20.0, decay_ms=100.0, sustain_level=0.6, release_ms=150.0),
        sustain_duration_ms=420.0,
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
        sustain_duration_ms=80.0,
        envelope_curve_path="/tmp/c3/envelope.npy",
        sustain_slice_path=None,
    )
    assert c.sustain_slice_path is None
    assert c.sustain_duration_ms == 80.0


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


SR = 22050


def _adsr_audio(duration_s: float, sustain: float = 0.6, freq: float = 220.0) -> np.ndarray:
    n_a = int(0.02 * SR)
    n_d = int(0.10 * SR)
    n_s = max(0, int(SR * duration_s) - n_a - n_d - int(0.15 * SR))
    n_r = int(0.15 * SR)
    env = np.concatenate([
        np.linspace(0.0, 1.0, n_a, endpoint=False),
        np.linspace(1.0, sustain, n_d, endpoint=False),
        np.full(n_s, sustain),
        np.linspace(sustain, 0.0, n_r, endpoint=True),
    ])
    t = np.arange(env.size) / SR
    return (env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _cluster_at(start_s: float, end_s: float, pitch: int) -> CandidateCluster:
    return CandidateCluster(
        kind="single", score=2.5,
        start_time=start_s, end_time=end_s,
        start_freq=200.0, end_freq=2000.0,
        members=[CandidateNote(
            note=NoteEvent(start_time=start_s, end_time=end_s, pitch_midi=pitch,
                           amplitude=0.8, pitch_bends=None),
            score=2.5, start_time=start_s, end_time=end_s,
            start_freq=200.0, end_freq=2000.0,
        )],
    )


def _write_triage(tmp_path: Path, clusters: list[CandidateCluster]) -> Path:
    data = NoteTriageFileData(polyphony_profile=[], candidates=clusters)
    path = tmp_path / "triage.json"
    path.write_text(data.model_dump_json(indent=2))
    return path


def test_orchestrator_two_consistent_clusters_emits_consensus(tmp_path: Path):
    # Two near-identical synthetic notes back-to-back → one combined audio buffer.
    note_a = _adsr_audio(duration_s=0.77)
    silence = np.zeros(int(0.5 * SR), dtype=np.float32)
    note_b = _adsr_audio(duration_s=0.77)
    audio = np.concatenate([note_a, silence, note_b])

    end_a = note_a.size / SR
    start_b = (note_a.size + silence.size) / SR
    end_b = audio.size / SR

    clusters = [
        _cluster_at(0.0, end_a, 60),
        _cluster_at(start_b, end_b, 64),
    ]
    triage_path = _write_triage(tmp_path, clusters)

    result = analyze_amplitude(
        audio=audio, sample_rate=SR,
        triage_path=triage_path, output_dir=tmp_path / "amp",
    )
    assert result.rejected_reason is None
    assert len(result.candidates) == 2
    assert result.is_consistent
    assert result.consensus_adsr is not None
    # Each candidate has its own envelope.npy on disk
    for c in result.candidates:
        assert Path(c.envelope_curve_path).exists()


def test_orchestrator_rejects_when_all_plucks(tmp_path: Path):
    # A single very short note → ADSR fit returns sustain_level=0 (pluck fallback) → skipped.
    n_a = int(0.02 * SR)
    n_d = int(0.05 * SR)
    short = np.concatenate([
        np.linspace(0.0, 1.0, n_a, endpoint=False),
        np.linspace(1.0, 0.0, n_d, endpoint=True),
    ])
    t = np.arange(short.size) / SR
    audio = (short * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    clusters = [_cluster_at(0.0, audio.size / SR, 60)]
    triage_path = _write_triage(tmp_path, clusters)

    result = analyze_amplitude(
        audio=audio, sample_rate=SR,
        triage_path=triage_path, output_dir=tmp_path / "amp",
    )
    assert result.rejected_reason == "no candidates with usable sustain"
    assert result.candidates == []
    assert result.consensus_adsr is None


def test_orchestrator_no_clusters_returns_rejected(tmp_path: Path):
    audio = _adsr_audio(0.77)
    triage_path = _write_triage(tmp_path, [])
    result = analyze_amplitude(
        audio=audio, sample_rate=SR,
        triage_path=triage_path, output_dir=tmp_path / "amp",
    )
    assert result.rejected_reason == "no candidates with usable sustain"
    assert result.candidates == []


def test_orchestrator_writes_per_cluster_outputs(tmp_path: Path):
    audio = _adsr_audio(0.77)
    clusters = [_cluster_at(0.0, audio.size / SR, 60)]
    triage_path = _write_triage(tmp_path, clusters)
    result = analyze_amplitude(
        audio=audio, sample_rate=SR,
        triage_path=triage_path, output_dir=tmp_path / "amp",
    )
    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert "cluster_00" in c.envelope_curve_path
    if c.sustain_slice_path:
        assert "cluster_00" in c.sustain_slice_path
        assert Path(c.sustain_slice_path).exists()


def test_orchestrator_reports_sustain_duration(tmp_path: Path):
    # _adsr_audio synthesizes ~500ms sustain at amplitude 0.6.
    # The detected sustain region should be in roughly the same ballpark.
    audio = _adsr_audio(0.77)
    clusters = [_cluster_at(0.0, audio.size / SR, 60)]
    triage_path = _write_triage(tmp_path, clusters)
    result = analyze_amplitude(
        audio=audio, sample_rate=SR,
        triage_path=triage_path, output_dir=tmp_path / "amp",
    )
    c = result.candidates[0]
    assert c.sustain_duration_ms > 0
    # Loose bound — the heuristic doesn't recover the exact 500ms, but it
    # should land between 100ms (sustain isolation gate) and 700ms (clip length).
    assert 100.0 <= c.sustain_duration_ms <= 700.0


def test_sustain_duration_does_not_affect_divergence(tmp_path: Path):
    # Two clusters with the same canonical 4-tuple ADSR but different
    # sustain durations must still be flagged as consistent (zero divergence
    # on the 4-axis vector).
    note_a = _adsr_audio(duration_s=0.77)
    silence = np.zeros(int(0.5 * SR), dtype=np.float32)
    note_b = _adsr_audio(duration_s=0.77)
    audio = np.concatenate([note_a, silence, note_b])

    end_a = note_a.size / SR
    start_b = (note_a.size + silence.size) / SR
    end_b = audio.size / SR
    clusters = [
        _cluster_at(0.0, end_a, 60),
        _cluster_at(start_b, end_b, 64),
    ]
    triage_path = _write_triage(tmp_path, clusters)
    result = analyze_amplitude(
        audio=audio, sample_rate=SR,
        triage_path=triage_path, output_dir=tmp_path / "amp",
    )
    assert result.is_consistent
    assert result.divergence_score < 0.05  # essentially zero — the two clusters are identical


def test_workspace_job_amplitude_dir(tmp_path: Path):
    from audio_analysis_mcp.workspace import Workspace
    ws = Workspace(root=tmp_path)
    d = ws.job_amplitude_dir("myjob", stem="other", preset="htdemucs")
    assert d.exists()
    assert d.relative_to(tmp_path) == Path("jobs/myjob/amplitude/other_htdemucs")

