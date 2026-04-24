# TODO

- [ ] Fix `note_isolate` — most isolated notes are silent or barely audible; only high-frequency notes (e.g. F5) produce audible output. Likely an issue with the STFT time-frequency masking approach for low-frequency notes.
- [ ] `note_triage`: add RMS energy threshold — filter out candidates below a minimum amplitude to reject stem separation artifacts (e.g. bridge sections where Demucs produces noise instead of music)
- [ ] `note_triage`: add chord-aware mode — instead of only preferring monophonic windows, also select "stable chord windows" where polyphony is consistent over a sustained period. Songs like Jump are mostly chords (polyphony 3-5), so the current algorithm skips all the musically interesting content.
- [ ] `note_triage`: accept optional `start_time`/`end_time` parameters to focus on a known section of the song, allowing the user to skip problematic regions (e.g. bridge with bad stem separation)
