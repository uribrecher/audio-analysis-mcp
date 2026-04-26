from audio_analysis_mcp.schemas import AmplitudeTriage, NoteEvent

_BLOCK_CHORD_TOLERANCE_S = 0.030
_ARPEGGIO_ONSETS_PER_SECOND = 3.0
_MIN_VELOCITY = 0.1  # below this, signal-to-noise ratio is too poor for reliable ADSR fitting


def classify_adsr_triage(notes: list[NoteEvent]) -> AmplitudeTriage:
    if not notes:
        return AmplitudeTriage.REJECTED
    if max((n.amplitude for n in notes), default=0.0) < _MIN_VELOCITY:
        return AmplitudeTriage.REJECTED
    if len(notes) == 1:
        return AmplitudeTriage.MONOPHONIC

    starts = [n.start_time for n in notes]
    ends = [n.end_time for n in notes]
    if (max(starts) - min(starts) <= _BLOCK_CHORD_TOLERANCE_S
            and max(ends) - min(ends) <= _BLOCK_CHORD_TOLERANCE_S):
        return AmplitudeTriage.BLOCK_CHORD

    span = max(ends) - min(starts)
    if span <= 0:
        return AmplitudeTriage.REJECTED
    onsets_per_second = len(notes) / span
    if onsets_per_second > _ARPEGGIO_ONSETS_PER_SECOND:
        return AmplitudeTriage.REJECTED  # arpeggio: per-note segmentation deferred to v2
    return AmplitudeTriage.MONOPHONIC
