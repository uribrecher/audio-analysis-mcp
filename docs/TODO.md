# TODO

- [ ] Fix `note_isolate` — most isolated notes are silent or barely audible; only high-frequency notes (e.g. F5) produce audible output. Likely an issue with the STFT time-frequency masking approach for low-frequency notes.
- [ ] `note_triage`: add RMS energy threshold — filter out candidates below a minimum amplitude to reject stem separation artifacts (e.g. bridge sections where Demucs produces noise instead of music)
- [x] `note_triage`: add chord-aware mode — done in 2026-04-26-note-triage-refactor: 3-pass cluster→score→select with overlap-based clustering produces single/chord/arpeggio cluster candidates.
- [x] `note_triage`: accept optional `start_time`/`end_time` parameters — done.
- [ ] `note_triage`: handle stem-separation leakage. When a stem is fed to triage (e.g. `other.wav`), low-pitch notes (C3, G3, etc.) often turn out to be bass leaking from the bass stem rather than real keyboard content. Effects: triage scores those leaked notes highly because they appear as "isolated singles" in the keyboard stem (no concurrent keyboard notes overlapping them). Two possible mitigations:
   - (a) Optional `min_pitch` parameter on triage — drop notes below it. Cheap, but a heuristic.
   - (b) Cross-stem leakage filter — given the bass-stem transcription, exclude `other` notes whose pitch+time match a `bass` note within tolerance. More principled.
   - (c) Better upstream: try Demucs `htdemucs_ft` preset and see if leakage decreases.
- [ ] Quantization-aware chord clustering. Real chords transcribed from audio have onset jitter (~50-200ms across members). The current overlap-based clustering handles tight chords but classifies "rolled" chords (where members don't all share a sounding moment) as arpeggios. A BPM/time-signature-aware quantization step before triage could collapse onset jitter to a beat grid and recover those chords. Out of scope for v1.
