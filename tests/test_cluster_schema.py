from audio_analysis_mcp.schemas import (
    CandidateCluster,
    CandidateNote,
    NoteEvent,
    NoteTriageFileData,
    NoteTriageResult,
    PolyphonyWindow,
)


def _candidate_note(start: float, end: float, pitch: int = 60) -> CandidateNote:
    return CandidateNote(
        note=NoteEvent(start_time=start, end_time=end, pitch_midi=pitch, amplitude=0.8, pitch_bends=None),
        score=1.0,
        start_time=start,
        end_time=end,
        start_freq=200.0,
        end_freq=2000.0,
    )


def test_candidate_cluster_single_note():
    cluster = CandidateCluster(
        kind="single",
        score=2.5,
        start_time=0.0,
        end_time=1.0,
        start_freq=200.0,
        end_freq=2000.0,
        members=[_candidate_note(0.0, 1.0)],
    )
    assert cluster.kind == "single"
    assert len(cluster.members) == 1


def test_candidate_cluster_chord():
    cluster = CandidateCluster(
        kind="chord",
        score=2.0,
        start_time=0.0,
        end_time=1.0,
        start_freq=200.0,
        end_freq=2400.0,
        members=[_candidate_note(0.0, 1.0, 60), _candidate_note(0.0, 1.0, 64), _candidate_note(0.0, 1.0, 67)],
    )
    assert cluster.kind == "chord"
    assert len(cluster.members) == 3


def test_note_triage_file_data_holds_clusters():
    data = NoteTriageFileData(
        polyphony_profile=[PolyphonyWindow(start_time=0.0, end_time=0.5, note_count=1)],
        candidates=[CandidateCluster(
            kind="single", score=1.0,
            start_time=0.0, end_time=1.0, start_freq=200.0, end_freq=2000.0,
            members=[_candidate_note(0.0, 1.0)],
        )],
    )
    assert isinstance(data.candidates[0], CandidateCluster)


def test_note_triage_result_holds_clusters():
    result = NoteTriageResult(
        triage_path="/tmp/triage.json",
        candidate_count=1,
        top_candidates=[CandidateCluster(
            kind="single", score=1.0,
            start_time=0.0, end_time=1.0, start_freq=200.0, end_freq=2000.0,
            members=[_candidate_note(0.0, 1.0)],
        )],
    )
    assert result.top_candidates[0].kind == "single"