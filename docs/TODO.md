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

## amplitude_analyze: not ready for production

Manual testing on Van Halen "Jump" synth chords (clusters 00, 01, 03 in the [0,30]s window) showed the heuristic ADSR fitter is meaningfully off in every component. Plotted envelopes are at `scratch/plot_amplitude_envelopes.py` and `extract_cluster_audio.py` reproduces the audio slices used.

Concrete issues observed:

- [ ] **Attack-end is taken at RMS peak, but should be at the RMS slope inflection (2nd-derivative crossing).** On synth tones with flat-ish plateaus, peak-based attack is wildly inflated. Cluster 01 ("5-note C @ 8.9s"): recovered attack=205ms, audio's actual attack ends at ~40ms.
- [ ] **Decay segment is forced into the model.** Real synth patches often have no decay — sustain begins at the attack-end inflection. The fitter currently always reports a decay value.
- [ ] **Sustain end is detected too late.** The algorithm walks until envelope variance exceeds threshold OR mean drops below 10% of peak. With noise floor riding above 10% of peak, sustain region extends well into the actual release. Cluster 00: sustain end recovered at ~510ms, audio loses energy at ~300ms.
- [ ] **Release length is dominated by noise floor.** Algorithm walks the tail until envelope < 5% of peak — a threshold the noise floor never falls under. Cluster 00 recovered release=469ms (impossibly long for that audio).
- [ ] **Cluster 03 specifically picks up substantial noise.** The triage gave us a clean 4-note G chord but its 0.3s-margin slice is noisy. Likely a stem-separation artifact at that moment combined with no noise gating in the analyzer.

Next research direction: look for prior work on robust ADSR estimation from real audio. Likely needs:

- Noise floor estimation + signal-vs-noise gating before envelope analysis (so sustain/release boundaries are detected against signal energy, not RMS-of-noise).
- Inflection-based segmentation (find slope-change points in the envelope, then assign attack/decay/sustain/release to the segments) instead of threshold-crossing heuristics.
- Optional handling of "no decay" / "no sustain" engines (set those to zero rather than forcing values).

Until that lands: `amplitude_analyze` returns scaffolding-grade outputs only. Useful for downstream pipeline plumbing but the ADSR numbers should not be trusted for engine matching.
