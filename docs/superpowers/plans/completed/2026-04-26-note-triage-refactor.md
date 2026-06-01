# Note-Triage Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `note_triage` from per-note scoring to a 3-pass `cluster → score → select` pipeline, and update `note_isolate` to operate on clusters (one audio slice per cluster) instead of single notes. Also clean up obsolete amplitude-triage code that this refactor supersedes.

**Architecture:** `triage_notes` becomes three explicit passes. Pass 1 groups notes into single / chord / arpeggio clusters by temporal-overlap rules. Pass 2 scores each cluster (kind bonus + duration + velocity + temporal isolation + pitch diversity). Pass 3 drops arpeggios and returns the top-N clusters. `note_isolate` consumes a `CandidateCluster` and returns a single time-sliced wav (no per-member splitting; chords are isolated as one block per the design). Optional `start_time`/`end_time` params let callers triage a single song region.

**Tech Stack:** Python 3.11+, `numpy`, `librosa`, `pydantic`, `pytest`. No new dependencies.

**Scope (this plan):** note_triage refactor, note_isolate update, schema changes, tear-down of `analysis/adsr_triage.py` and the `AmplitudeTriage` enum. Amplitude-expert orchestrator/tool/tests are out of scope here — they will be redone in the revised `2026-04-26-amplitude-expert.md` after this plan lands.

---

## File Map

**Modify:**
- `src/audio_analysis_mcp/schemas.py` — add `CandidateCluster`, update `NoteTriageFileData` and `NoteTriageResult`. Remove `AmplitudeTriage` enum and `AmplitudeAnalyzeResult` (the latter will be re-added in the amplitude-expert plan with a different shape).
- `src/audio_analysis_mcp/analysis/note_triage.py` — replace per-note pipeline with the 3-pass cluster pipeline.
- `src/audio_analysis_mcp/analysis/note_isolation.py` — accept a `CandidateCluster`, output one time slice; freq band uses encompassing range (still applied for noise reduction on single notes; for chords the band is wide).
- `src/audio_analysis_mcp/tools/note_triage.py` — return `top_candidates: list[CandidateCluster]`.
- `src/audio_analysis_mcp/tools/note_isolate.py` — read `CandidateCluster` from triage JSON.
- `src/audio_analysis_mcp/analysis/amplitude.py` — remove (will be rewritten in amplitude-expert plan against the new triage contract).
- `tests/test_note_triage.py` — replace per-note assertions with cluster-level ones.
- `tests/test_note_isolation.py` — feed CandidateCluster, assert one wav per call.
- `tests/test_mcp_tools.py` — update e2e flow for new schemas.
- `tests/test_amplitude.py` — remove (will be re-added in amplitude-expert plan).

**Delete:**
- `src/audio_analysis_mcp/analysis/adsr_triage.py`
- `tests/test_adsr_triage.py`

---

## Task 1: Cleanup — remove obsolete amplitude-triage code

**Files:**
- Delete: `src/audio_analysis_mcp/analysis/adsr_triage.py`
- Delete: `tests/test_adsr_triage.py`
- Modify: `src/audio_analysis_mcp/schemas.py` — remove `AmplitudeTriage` enum and `AmplitudeAnalyzeResult` model
- Delete: `src/audio_analysis_mcp/analysis/amplitude.py`
- Delete: `tests/test_amplitude.py`

This commit lands first so the rest of the plan starts from a clean slate. The amplitude module and tests are intentionally removed entirely — they will be rewritten against the new cluster contract by `2026-04-26-amplitude-expert.md`.

- [ ] **Step 1: Verify what we're deleting**

Run: `ls -la src/audio_analysis_mcp/analysis/adsr_triage.py src/audio_analysis_mcp/analysis/amplitude.py tests/test_adsr_triage.py tests/test_amplitude.py`
Expected: all four files exist.

- [ ] **Step 2: Delete the four files**

```bash
git rm src/audio_analysis_mcp/analysis/adsr_triage.py
git rm src/audio_analysis_mcp/analysis/amplitude.py
git rm tests/test_adsr_triage.py
git rm tests/test_amplitude.py
```

- [ ] **Step 3: Remove the obsolete schemas**

Open `src/audio_analysis_mcp/schemas.py` and delete:

- The `from enum import Enum` import (no longer needed).
- The entire `AmplitudeTriage` enum class.
- The entire `AmplitudeAnalyzeResult` model.

Leave `ADSREstimate` (it's still used downstream).

- [ ] **Step 4: Verify nothing else imports the removed types**

Run: `grep -rn "AmplitudeTriage\|AmplitudeAnalyzeResult\|adsr_triage\|analysis.amplitude" src tests`
Expected: zero matches. If any remain, they need to be cleaned up before the commit.

- [ ] **Step 5: Run remaining test suite**

Run: `uv run pytest tests/ -v && uv run mypy src/`
Expected: every test still passes (we removed self-contained tests; nothing else depends on the removed code), mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/audio_analysis_mcp/schemas.py
git commit -m "amplitude: remove obsolete adsr_triage and AmplitudeTriage (superseded by cluster-based triage)"
```

---

## Task 2: Schemas — add `CandidateCluster`

**Files:**
- Modify: `src/audio_analysis_mcp/schemas.py`
- Test: extend an existing test file or create `tests/test_cluster_schema.py`

The new type:

```python
ClusterKind = Literal["single", "chord", "arpeggio"]

class CandidateCluster(BaseModel):
    kind: ClusterKind                      # arpeggios are filtered in pass 3 but the type allows it
    score: float
    start_time: float                       # padded encompassing range
    end_time: float
    start_freq: float                       # min over members
    end_freq: float                         # max over members
    members: list[CandidateNote]            # constituent notes (1 for single, ≥2 for chord, ≥3 for arpeggio)
```

`NoteTriageFileData` and `NoteTriageResult` change to hold `list[CandidateCluster]` instead of `list[CandidateNote]`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_cluster_schema.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cluster_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'CandidateCluster' from 'audio_analysis_mcp.schemas'`.

- [ ] **Step 3: Add the schema and update existing types**

In `src/audio_analysis_mcp/schemas.py`:

1. Add `from typing import Literal` at the top alongside the existing imports.
2. Add the `ClusterKind` alias and `CandidateCluster` class:

```python
ClusterKind = Literal["single", "chord", "arpeggio"]


class CandidateCluster(BaseModel):
    kind: ClusterKind
    score: float
    start_time: float
    end_time: float
    start_freq: float
    end_freq: float
    members: list[CandidateNote]
```

3. Change `NoteTriageFileData.candidates` from `list[CandidateNote]` to `list[CandidateCluster]`.
4. Change `NoteTriageResult.top_candidates` from `list[CandidateNote]` to `list[CandidateCluster]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cluster_schema.py -v && uv run mypy src/`
Expected: 4 passed, mypy clean. mypy will likely flag the existing callers of `NoteTriageFileData` / `NoteTriageResult` (in `analysis/note_triage.py`, `tools/note_triage.py`, `tools/note_isolate.py`) — those are the next tasks. **Confirm the mypy errors are limited to those three files**; if anything else surfaces, stop and report.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/schemas.py tests/test_cluster_schema.py
git commit -m "triage: add CandidateCluster schema; NoteTriageFileData/Result now hold clusters"
```

---

## Task 3: Clustering pass — `_cluster_notes`

Pure-logic helper that groups a list of `NoteEvent`s into `CandidateCluster`s of kind `single` / `chord` / `arpeggio`.

**Rules:**
- Sort notes by `start_time`.
- **Chord cluster:** any group of 2+ notes whose start times are within `_CHORD_TOLERANCE_S` (30 ms) of each other AND whose end times are within `_CHORD_TOLERANCE_S` of each other → one chord cluster.
- **Arpeggio cluster:** any sequence of 3+ notes (not in any chord) whose successive onset gaps are all ≤ `_ARPEGGIO_GAP_S` (150 ms) → one arpeggio cluster.
- **Single:** any remaining note → its own single-note cluster.

The function returns clusters with provisional `score=0.0` (filled in by Task 4) and provisional `start_freq`/`end_freq` derived from members.

**Files:**
- Modify: `src/audio_analysis_mcp/analysis/note_triage.py`
- Test: `tests/test_note_triage.py` (extend; do not delete existing tests yet — they'll be rewritten in Task 5)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_note_triage.py`:

```python
from audio_analysis_mcp.analysis.note_triage import _cluster_notes


def _ev(start: float, end: float, pitch: int = 60, amp: float = 0.8) -> NoteEvent:
    return NoteEvent(
        start_time=start, end_time=end, pitch_midi=pitch, amplitude=amp, pitch_bends=None
    )


def test_cluster_empty():
    assert _cluster_notes([]) == []


def test_cluster_single_note():
    clusters = _cluster_notes([_ev(0.0, 1.0)])
    assert len(clusters) == 1
    assert clusters[0].kind == "single"
    assert len(clusters[0].members) == 1


def test_cluster_chord_three_simultaneous():
    notes = [_ev(0.0, 1.0, 60), _ev(0.01, 1.0, 64), _ev(0.02, 0.99, 67)]
    clusters = _cluster_notes(notes)
    assert len(clusters) == 1
    assert clusters[0].kind == "chord"
    assert len(clusters[0].members) == 3


def test_cluster_arpeggio_six_notes():
    # 6 notes spaced 100 ms apart: each onset gap is 100 ms ≤ 150 ms threshold
    notes = [_ev(i * 0.1, i * 0.1 + 0.4, 60 + i) for i in range(6)]
    clusters = _cluster_notes(notes)
    assert len(clusters) == 1
    assert clusters[0].kind == "arpeggio"
    assert len(clusters[0].members) == 6


def test_cluster_arpeggio_minimum_size_3():
    # Two sequential notes are NOT an arpeggio → two singles
    notes = [_ev(0.0, 0.4, 60), _ev(0.45, 0.85, 62)]
    clusters = _cluster_notes(notes)
    assert {c.kind for c in clusters} == {"single"}
    assert len(clusters) == 2


def test_cluster_mixed_chord_and_single():
    # A chord at t=0..1 + a sequential single note at t=2..3
    notes = [_ev(0.0, 1.0, 60), _ev(0.0, 1.0, 64), _ev(2.0, 3.0, 72)]
    clusters = _cluster_notes(notes)
    assert len(clusters) == 2
    assert {c.kind for c in clusters} == {"chord", "single"}


def test_cluster_arpeggio_breaks_on_long_gap():
    # 4 notes with onsets at 0.0, 0.1, 0.5 (gap 400ms > 150ms), 0.6 → first 2 too short for arpeggio,
    # last 2 too short → 4 singles
    notes = [_ev(0.0, 0.1), _ev(0.1, 0.2), _ev(0.5, 0.6), _ev(0.6, 0.7)]
    clusters = _cluster_notes(notes)
    assert all(c.kind == "single" for c in clusters)
    assert len(clusters) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_note_triage.py -v`
Expected: existing tests still pass (until we replace `triage_notes` in Task 5); the new ones fail with `ImportError: cannot import name '_cluster_notes'`.

- [ ] **Step 3: Implement `_cluster_notes`**

Add to `src/audio_analysis_mcp/analysis/note_triage.py` (above `triage_notes`):

```python
_CHORD_TOLERANCE_S = 0.030
_ARPEGGIO_GAP_S = 0.150
_ARPEGGIO_MIN_SIZE = 3


def _build_candidate_note(note: NoteEvent) -> CandidateNote:
    start_freq, end_freq = _freq_bounds(note.pitch_midi)
    padded_start = max(0.0, note.start_time - TIME_PADDING)
    padded_end = note.end_time + TIME_PADDING
    return CandidateNote(
        note=note,
        score=0.0,                       # filled in by Task 4
        start_time=padded_start,
        end_time=padded_end,
        start_freq=round(start_freq, 2),
        end_freq=round(end_freq, 2),
    )


def _cluster_notes(notes: list[NoteEvent]) -> list[CandidateCluster]:
    """Pass 1: group notes into single / chord / arpeggio clusters."""
    if not notes:
        return []

    notes_sorted = sorted(notes, key=lambda n: n.start_time)

    # Step 1: collect chord groups (greedy left-to-right).
    used: set[int] = set()
    chord_groups: list[list[int]] = []
    for i, n in enumerate(notes_sorted):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(notes_sorted)):
            if j in used:
                continue
            other = notes_sorted[j]
            if (abs(other.start_time - n.start_time) <= _CHORD_TOLERANCE_S
                    and abs(other.end_time - n.end_time) <= _CHORD_TOLERANCE_S):
                group.append(j)
        if len(group) >= 2:
            for k in group:
                used.add(k)
            chord_groups.append(group)

    # Step 2: from the remaining notes, find arpeggio runs.
    remaining_indices = [i for i in range(len(notes_sorted)) if i not in used]
    arpeggio_groups: list[list[int]] = []
    run: list[int] = []
    for idx in remaining_indices:
        if not run:
            run = [idx]
            continue
        prev_idx = run[-1]
        gap = notes_sorted[idx].start_time - notes_sorted[prev_idx].start_time
        if 0.0 <= gap <= _ARPEGGIO_GAP_S:
            run.append(idx)
        else:
            if len(run) >= _ARPEGGIO_MIN_SIZE:
                arpeggio_groups.append(run)
                for k in run:
                    used.add(k)
            run = [idx]
    if len(run) >= _ARPEGGIO_MIN_SIZE:
        arpeggio_groups.append(run)
        for k in run:
            used.add(k)

    # Step 3: emit clusters in start-time order.
    cluster_specs: list[tuple[str, list[int]]] = []
    cluster_specs.extend(("chord", g) for g in chord_groups)
    cluster_specs.extend(("arpeggio", g) for g in arpeggio_groups)
    for i in range(len(notes_sorted)):
        if i not in used:
            cluster_specs.append(("single", [i]))

    cluster_specs.sort(key=lambda spec: notes_sorted[spec[1][0]].start_time)

    clusters: list[CandidateCluster] = []
    for kind, indices in cluster_specs:
        members = [_build_candidate_note(notes_sorted[i]) for i in indices]
        start_time = min(m.start_time for m in members)
        end_time = max(m.end_time for m in members)
        start_freq = min(m.start_freq for m in members)
        end_freq = max(m.end_freq for m in members)
        clusters.append(CandidateCluster(
            kind=kind,                              # type: ignore[arg-type]
            score=0.0,
            start_time=start_time,
            end_time=end_time,
            start_freq=start_freq,
            end_freq=end_freq,
            members=members,
        ))

    return clusters
```

Update the import block at the top of the file to include `CandidateCluster`:

```python
from audio_analysis_mcp.schemas import (
    NoteEvent,
    PolyphonyWindow,
    CandidateNote,
    CandidateCluster,
    NoteTriageFileData,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_note_triage.py -v && uv run mypy src/`
Expected: previously-passing tests still pass; the 7 new clustering tests pass; mypy still flags the soon-to-be-rewritten `triage_notes` callers but no new errors. Confirm.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/analysis/note_triage.py tests/test_note_triage.py
git commit -m "triage: add _cluster_notes (pass 1 — chord/arpeggio/single grouping)"
```

---

## Task 4: Cluster scoring — `_score_cluster`

Pure-logic scoring of a `CandidateCluster`. Higher = better.

**Components (all preserved from the prior per-note scoring, lifted to cluster level):**

| Component | Formula | Weight |
|---|---|---|
| `kind_bonus` | single=3.0, chord=2.0, arpeggio=0.0 | (constant, not weighted) |
| `poly_score` | `1 / max(mean_polyphony_over_cluster_window, 1.0)` | ×2.0 |
| `dur_score` | `log1p(min(cluster_duration, 2.0))` | ×1.0 |
| `gap_score` | `log1p(temporal_gap_to_nearest_other_cluster)` | ×0.5 |
| `velocity_score` | mean velocity over members | ×1.0 |

`score = kind_bonus + poly_score×2 + dur_score + gap_score×0.5 + velocity_score`

The pitch-diversity penalty is applied later in pass 3 (selection), not here, because it's a comparative penalty (cluster vs. already-selected).

**Files:**
- Modify: `src/audio_analysis_mcp/analysis/note_triage.py`
- Test: `tests/test_note_triage.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_note_triage.py`:

```python
from audio_analysis_mcp.analysis.note_triage import _score_cluster, _build_polyphony_profile


def test_score_single_outscores_chord_outscores_arpeggio():
    # Three identical-shape clusters differing only in kind.
    clusters = []
    for kind in ("single", "chord", "arpeggio"):
        members = [_build_candidate_note_for_test(0.0, 1.0, 60)]
        clusters.append(CandidateCluster(
            kind=kind, score=0.0,
            start_time=0.0, end_time=1.0, start_freq=200.0, end_freq=2000.0,
            members=members,
        ))
    profile = _build_polyphony_profile([_ev(0.0, 1.0)])
    s_single = _score_cluster(clusters[0], profile, clusters)
    s_chord = _score_cluster(clusters[1], profile, clusters)
    s_arp = _score_cluster(clusters[2], profile, clusters)
    assert s_single > s_chord > s_arp


def test_score_velocity_helps():
    soft = CandidateCluster(
        kind="single", score=0.0, start_time=0.0, end_time=1.0,
        start_freq=200.0, end_freq=2000.0,
        members=[_build_candidate_note_for_test(0.0, 1.0, 60, amp=0.1)],
    )
    loud = CandidateCluster(
        kind="single", score=0.0, start_time=2.0, end_time=3.0,
        start_freq=200.0, end_freq=2000.0,
        members=[_build_candidate_note_for_test(2.0, 3.0, 60, amp=0.9)],
    )
    profile = _build_polyphony_profile([_ev(0.0, 1.0), _ev(2.0, 3.0)])
    assert _score_cluster(loud, profile, [soft, loud]) > _score_cluster(soft, profile, [soft, loud])


def test_score_longer_duration_helps():
    short = CandidateCluster(
        kind="single", score=0.0, start_time=0.0, end_time=0.6,
        start_freq=200.0, end_freq=2000.0,
        members=[_build_candidate_note_for_test(0.0, 0.6, 60)],
    )
    long_c = CandidateCluster(
        kind="single", score=0.0, start_time=2.0, end_time=4.0,
        start_freq=200.0, end_freq=2000.0,
        members=[_build_candidate_note_for_test(2.0, 4.0, 60)],
    )
    profile = _build_polyphony_profile([_ev(0.0, 0.6), _ev(2.0, 4.0)])
    assert _score_cluster(long_c, profile, [short, long_c]) > _score_cluster(short, profile, [short, long_c])


def _build_candidate_note_for_test(start: float, end: float, pitch: int, amp: float = 0.8) -> CandidateNote:
    return CandidateNote(
        note=NoteEvent(start_time=start, end_time=end, pitch_midi=pitch, amplitude=amp, pitch_bends=None),
        score=0.0, start_time=start, end_time=end, start_freq=200.0, end_freq=2000.0,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_note_triage.py -v`
Expected: new score tests fail with `ImportError: cannot import name '_score_cluster'`.

- [ ] **Step 3: Implement `_score_cluster`**

Add to `src/audio_analysis_mcp/analysis/note_triage.py`:

```python
_KIND_BONUS = {"single": 3.0, "chord": 2.0, "arpeggio": 0.0}


def _cluster_polyphony(cluster: CandidateCluster, profile: list[PolyphonyWindow]) -> float:
    """Average polyphony count across windows that overlap with the cluster's time range."""
    overlapping = [
        w for w in profile if w.start_time < cluster.end_time and w.end_time > cluster.start_time
    ]
    if not overlapping:
        return 1.0
    return sum(w.note_count for w in overlapping) / len(overlapping)


def _cluster_temporal_gap(cluster: CandidateCluster, all_clusters: list[CandidateCluster]) -> float:
    """Minimum time gap to the nearest neighboring cluster (seconds)."""
    min_gap = float("inf")
    for other in all_clusters:
        if other is cluster:
            continue
        gap = max(0.0, max(other.start_time - cluster.end_time, cluster.start_time - other.end_time))
        min_gap = min(min_gap, gap)
    return min_gap if min_gap != float("inf") else 1.0


def _score_cluster(
    cluster: CandidateCluster,
    profile: list[PolyphonyWindow],
    all_clusters: list[CandidateCluster],
) -> float:
    """Score a cluster for ADSR-analysis suitability. Higher = better."""
    duration = cluster.end_time - cluster.start_time
    poly = _cluster_polyphony(cluster, profile)
    gap = _cluster_temporal_gap(cluster, all_clusters)
    velocity = sum(m.note.amplitude for m in cluster.members) / max(len(cluster.members), 1)

    poly_score = 1.0 / max(poly, 1.0)
    dur_score = float(np.log1p(min(duration, 2.0)))
    gap_score = float(np.log1p(gap))
    kind_bonus = _KIND_BONUS[cluster.kind]

    return kind_bonus + poly_score * 2.0 + dur_score + gap_score * 0.5 + velocity * 1.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_note_triage.py -v && uv run mypy src/`
Expected: 3 new score tests pass; clustering tests still pass; mypy clean for the new code (the rewrite of `triage_notes` is still pending, so any pre-existing errors in that function will remain).

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/analysis/note_triage.py tests/test_note_triage.py
git commit -m "triage: add _score_cluster (pass 2 — kind/poly/duration/velocity/gap)"
```

---

## Task 5: Refactor `triage_notes` to 3-pass + add time-window params

Replace the body of `triage_notes` with the 3-pass pipeline and add optional `start_time` / `end_time` parameters that filter notes to a region of the song.

**Files:**
- Modify: `src/audio_analysis_mcp/analysis/note_triage.py`
- Replace: existing tests in `tests/test_note_triage.py` that asserted the old per-note shape

- [ ] **Step 1: Rewrite the existing tests**

Open `tests/test_note_triage.py` and DELETE the existing tests that asserted the old per-note shape (everything that used `result.candidates[i].note` directly without going through `members`). Keep the tests added in Tasks 3 and 4.

Then append new top-level tests:

```python
from audio_analysis_mcp.analysis.note_triage import triage_notes


def test_triage_returns_clusters_sorted_by_score():
    notes = [
        _ev(0.0, 0.3, 60, amp=0.2),       # short low-velocity single → low score
        _ev(2.0, 4.0, 64, amp=0.9),       # long high-velocity single → high score
        _ev(6.0, 7.0, 67, amp=0.6),       # medium single
    ]
    data = triage_notes(notes, min_duration=0.0, max_candidates=10)
    assert len(data.candidates) == 3
    # Highest score first
    assert data.candidates[0].kind == "single"
    assert data.candidates[0].members[0].note.pitch_midi == 64


def test_triage_filters_arpeggios():
    # Two singles + one arpeggio of 4 notes
    notes = [
        _ev(0.0, 1.0, 60),                       # single
        _ev(2.0, 3.0, 64),                       # single
        _ev(5.0, 5.2, 60), _ev(5.15, 5.35, 62),  # part of arpeggio
        _ev(5.30, 5.50, 64), _ev(5.45, 5.65, 65),
    ]
    data = triage_notes(notes, min_duration=0.0, max_candidates=10)
    assert all(c.kind != "arpeggio" for c in data.candidates)


def test_triage_min_duration_filters_short_notes():
    notes = [_ev(0.0, 0.3, 60), _ev(1.0, 2.0, 64)]
    data = triage_notes(notes, min_duration=0.5)
    assert len(data.candidates) == 1
    assert data.candidates[0].members[0].note.pitch_midi == 64


def test_triage_respects_time_window():
    notes = [_ev(0.0, 1.0, 60), _ev(5.0, 6.0, 64), _ev(10.0, 11.0, 67)]
    data = triage_notes(notes, min_duration=0.0, start_time=4.0, end_time=8.0, max_candidates=10)
    assert len(data.candidates) == 1
    assert data.candidates[0].members[0].note.pitch_midi == 64


def test_triage_respects_max_candidates():
    notes = [_ev(i * 2.0, i * 2.0 + 1.0, 60 + i) for i in range(15)]
    data = triage_notes(notes, min_duration=0.0, max_candidates=5)
    assert len(data.candidates) == 5


def test_triage_pitch_diversity_penalty():
    # Two equal-shape singles at the same pitch → second one penalized.
    # Plus one cluster at a different pitch → should win over the second same-pitch one.
    notes = [
        _ev(0.0, 1.0, 60, amp=0.9),
        _ev(2.0, 3.0, 60, amp=0.9),
        _ev(4.0, 5.0, 72, amp=0.5),
    ]
    data = triage_notes(notes, min_duration=0.0, max_candidates=3)
    pitches = [c.members[0].note.pitch_midi for c in data.candidates]
    # The first same-pitch instance ranks above the diverse-pitch cluster (because amp=0.9 vs 0.5),
    # but the second same-pitch instance is penalized below the diverse one.
    assert pitches.index(60) < pitches.index(72) < pitches.index(60, pitches.index(60) + 1)


def test_triage_empty_notes_returns_empty():
    data = triage_notes([])
    assert data.candidates == []
    assert data.polyphony_profile == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_note_triage.py -v`
Expected: the new top-level `triage_notes` tests fail (current implementation returns per-note, not cluster shape).

- [ ] **Step 3: Replace `triage_notes` implementation**

In `src/audio_analysis_mcp/analysis/note_triage.py`, replace the entire body of `triage_notes` (and remove the now-unused `_score_note`, `_polyphony_at`, `_temporal_gap` helpers) with:

```python
def triage_notes(
    notes: list[NoteEvent],
    min_duration: float = 0.5,
    max_candidates: int = 10,
    start_time: float | None = None,
    end_time: float | None = None,
) -> NoteTriageFileData:
    """Three-pass triage: cluster → score → select.

    Optional `start_time`/`end_time` filter notes to a song region (callers
    already detected the region; this module does not detect song parts).
    """
    # Filter to time window
    if start_time is not None:
        notes = [n for n in notes if n.end_time > start_time]
    if end_time is not None:
        notes = [n for n in notes if n.start_time < end_time]

    # Filter by minimum duration (existing pluck filter)
    notes = [n for n in notes if (n.end_time - n.start_time) >= min_duration]

    profile = _build_polyphony_profile(notes)

    if not notes:
        return NoteTriageFileData(polyphony_profile=profile, candidates=[])

    # Pass 1: clustering
    clusters = _cluster_notes(notes)

    # Pass 2: score each cluster
    scored: list[tuple[CandidateCluster, float]] = [
        (c, _score_cluster(c, profile, clusters)) for c in clusters
    ]

    # Pass 3: drop arpeggios, sort, apply pitch-diversity penalty greedily, take top N
    scored = [(c, s) for c, s in scored if c.kind != "arpeggio"]
    scored.sort(key=lambda x: x[1], reverse=True)

    selected: list[tuple[CandidateCluster, float]] = []
    selected_pitches: list[int] = []
    for cluster, score in scored:
        if len(selected) >= max_candidates:
            break
        rep_pitch = cluster.members[0].note.pitch_midi  # highest-scoring member by sort order
        if any(abs(rep_pitch - p) <= 2 for p in selected_pitches):
            score *= 0.5
        selected.append((cluster, score))
        selected_pitches.append(rep_pitch)

    selected.sort(key=lambda x: x[1], reverse=True)
    selected = selected[:max_candidates]

    candidates = [
        c.model_copy(update={"score": round(s, 4)}) for c, s in selected
    ]
    return NoteTriageFileData(polyphony_profile=profile, candidates=candidates)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_note_triage.py -v && uv run mypy src/`
Expected: all tests in this file pass; mypy clean for `note_triage.py` itself; mypy may still flag `tools/note_triage.py` and `tools/note_isolate.py` callers (next tasks).

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/analysis/note_triage.py tests/test_note_triage.py
git commit -m "triage: replace per-note pipeline with 3-pass cluster/score/select"
```

---

## Task 6: Update `tools/note_triage.py` MCP tool

The MCP tool needs to expose the new `start_time` / `end_time` parameters and return clusters in `top_candidates`.

**Files:**
- Modify: `src/audio_analysis_mcp/tools/note_triage.py`
- Test: `tests/test_mcp_tools.py` (touch only the note_triage test; isolate tests in Task 8)

- [ ] **Step 1: Rewrite the failing test**

Open `tests/test_mcp_tools.py`. Find the existing `note_triage` MCP test and replace it with one that:
- Builds a `notes.json` with three notes (one short, one long, one chord pair).
- Calls `note_triage(audio_path, notes_path)`.
- Parses the returned JSON.
- Asserts `top_candidates[0]` has fields `kind`, `score`, `members`.
- Asserts the long single note is at index 0 of `top_candidates`.

Concrete replacement:

```python
def test_note_triage_returns_clusters(tmp_path: Path, monkeypatch):
    from audio_analysis_mcp import server as server_module
    from audio_analysis_mcp.tools.note_triage import note_triage
    import audio_analysis_mcp.tools.note_triage as tool_module

    ws = Workspace(root=tmp_path)
    monkeypatch.setattr(server_module, "get_workspace", lambda: ws)
    monkeypatch.setattr(tool_module, "get_workspace", lambda: ws)

    job_dir = ws.job_stems_dir("song", "htdemucs")
    audio_path = job_dir / "other.wav"
    sf.write(audio_path, np.zeros(22050, dtype=np.float32), 22050)

    notes = [
        NoteEvent(start_time=0.0, end_time=0.3, pitch_midi=60, amplitude=0.5, pitch_bends=None),
        NoteEvent(start_time=2.0, end_time=4.0, pitch_midi=64, amplitude=0.9, pitch_bends=None),
        NoteEvent(start_time=6.0, end_time=7.0, pitch_midi=67, amplitude=0.6, pitch_bends=None),
        NoteEvent(start_time=6.0, end_time=7.0, pitch_midi=71, amplitude=0.6, pitch_bends=None),
    ]
    notes_path = ws.job_dir("song") / "notes.json"
    notes_path.write_text(json.dumps([n.model_dump() for n in notes]))

    result_json = note_triage(audio_path=str(audio_path), notes_path=str(notes_path))
    payload = json.loads(result_json)
    assert payload["candidate_count"] >= 1
    top = payload["top_candidates"][0]
    assert "kind" in top and "members" in top and "score" in top
```

(Imports at top of the test file may need `import json`, `import numpy as np`, `import soundfile as sf`, `from audio_analysis_mcp.workspace import Workspace`, `from audio_analysis_mcp.schemas import NoteEvent` — match whatever the existing file has.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_mcp_tools.py::test_note_triage_returns_clusters -v`
Expected: FAIL — either the test is not yet present, or the existing tool returns per-note shape and the new assertions fail.

- [ ] **Step 3: Update the tool**

Open `src/audio_analysis_mcp/tools/note_triage.py`. Update the `note_triage` MCP tool function to expose the new parameters:

```python
@mcp.tool()
def note_triage(
    audio_path: str,
    notes_path: str,
    min_duration: float = 0.5,
    max_candidates: int = 10,
    start_time: float | None = None,
    end_time: float | None = None,
) -> str:
    """Triage notes into ranked clusters (single / chord) for downstream analysis.

    notes_path must be the JSON file from note_transcribe.
    Optional start_time/end_time filter notes to a song region.
    """
    ws = get_workspace()
    ctx = resolve_job_context(audio_path, ws)
    if ctx.stem is None or ctx.preset is None:
        raise ValueError(
            f"Expected a stem file (jobs/<job>/stems/<preset>/<stem>.wav), got: {audio_path}"
        )

    adapter = TypeAdapter(list[NoteEvent])
    notes_json = Path(notes_path).read_text()
    notes = adapter.validate_json(notes_json)

    file_data = triage_notes(
        notes=notes,
        min_duration=min_duration,
        max_candidates=max_candidates,
        start_time=start_time,
        end_time=end_time,
    )

    triage_dir = ws.job_triage_dir(ctx.job_name, ctx.stem, ctx.preset)
    triage_path = triage_dir / "triage.json"
    triage_path.write_text(file_data.model_dump_json(indent=2))

    return NoteTriageResult(
        triage_path=str(triage_path),
        candidate_count=len(file_data.candidates),
        top_candidates=file_data.candidates[:5],
    ).model_dump_json(indent=2)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_mcp_tools.py -v && uv run mypy src/`
Expected: the new note_triage test passes; the note_isolate tests in this file likely still fail (Task 7); mypy may flag `tools/note_isolate.py`.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/tools/note_triage.py tests/test_mcp_tools.py
git commit -m "triage: MCP tool exposes start_time/end_time and returns clusters"
```

---

## Task 7: Update `note_isolate` to operate on a `CandidateCluster`

`note_isolate` currently consumes `CandidateNote` to do time-frequency masking. The new behavior: take a `CandidateCluster`, produce one wav per cluster (time slice from the stem; the cluster's encompassing freq band still drives the mask, which is fine for single-note clusters and harmless-but-wide for chords).

The choice of which cluster to isolate (the user passes an index, same as today) does not change.

**Files:**
- Modify: `src/audio_analysis_mcp/analysis/note_isolation.py`
- Modify: `src/audio_analysis_mcp/tools/note_isolate.py`
- Modify: `tests/test_note_isolation.py`
- Modify: `tests/test_mcp_tools.py` (the note_isolate e2e test)

- [ ] **Step 1: Rewrite the unit tests**

Open `tests/test_note_isolation.py` and update it so the function under test is called with a `CandidateCluster` (single or chord). For the chord case, assert that the output wav's duration equals `(cluster.end_time - cluster.start_time) * sample_rate` samples (within ±1 hop). Replace any test that constructed a `CandidateNote` directly with one that constructs a `CandidateCluster` containing one or more `CandidateNote` members.

The exact test names and bodies depend on what's there today; the contract to assert is:
- Input: a stem path + a `CandidateCluster` (+ output path).
- Output: one wav file at `output_path`, mono, sample-rate matching the input stem, time-bounded by the cluster.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_note_isolation.py -v`
Expected: FAIL — current implementation doesn't accept a cluster.

- [ ] **Step 3: Update `analysis/note_isolation.py`**

Change the signature of the public function (today it takes a `CandidateNote`; rename if needed but minimize churn — prefer adapting in place) to take a `CandidateCluster`. The masking math operates on `cluster.start_time`/`end_time`/`start_freq`/`end_freq`. No per-member splitting.

- [ ] **Step 4: Update `tools/note_isolate.py`**

The MCP tool reads `triage.json` (now containing clusters). It looks up the requested cluster by index and passes it to the analysis function. The output filename should reflect the cluster (e.g., `cluster_<idx>_<kind>.wav` instead of the old `<pitch>_<idx>.wav`).

- [ ] **Step 5: Update the e2e test**

In `tests/test_mcp_tools.py`, find the `note_isolate` test. Update it so the upstream `note_triage` produces clusters and `note_isolate` consumes a cluster index. Assert the resulting wav exists and is non-empty.

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/ -v && uv run mypy src/`
Expected: full suite green, mypy clean.

- [ ] **Step 7: Commit**

```bash
git add src/audio_analysis_mcp/analysis/note_isolation.py \
        src/audio_analysis_mcp/tools/note_isolate.py \
        tests/test_note_isolation.py \
        tests/test_mcp_tools.py
git commit -m "isolate: consume CandidateCluster (one wav per cluster)"
```

---

## Task 8: Final verification

- [ ] **Step 1: Full suite + mypy**

Run:
```
uv run pytest tests/ -v -m "not slow"
uv run mypy src/
```
Expected: all non-slow tests pass; mypy clean.

- [ ] **Step 2: Verify obsolete code is gone**

Run: `grep -rn "AmplitudeTriage\|adsr_triage\|analysis.amplitude\b" src tests`
Expected: zero matches.

- [ ] **Step 3: No commit needed if everything was committed task-by-task.**

---

## Deferred (not in this plan)

- The amplitude expert orchestrator + MCP tool + tests will be (re-)added by `2026-04-26-amplitude-expert.md`, against the new `CandidateCluster` contract.
- Future per-note isolation inside a chord (for a chord-specific timbre model) — out of scope; the user has explicitly deferred it.

---

## Self-Review

- **Spec coverage:** every clarification from the design discussion is mapped to a task — 3-pass restructure (Tasks 3–5), velocity in scoring (Task 4), time-window params (Task 5), arpeggio drop (Task 5), cluster-level isolation (Task 7), obsolete-code teardown (Task 1).
- **Type consistency:** `ClusterKind` is a `Literal` alias defined once in `schemas.py` and referenced everywhere. `_KIND_BONUS` keys must match it (the test in Task 4 transitively verifies all three).
- **No placeholders:** every step shows the exact code, file path, and command. The two places that say "the exact test names depend on what's there today" (Task 7 Steps 1, 5) are pointers because the existing test file's names aren't constant — the *contract to assert* is fully specified.