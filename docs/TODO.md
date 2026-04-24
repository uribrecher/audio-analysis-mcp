# TODO

- [ ] Add offset/duration parameters to `import_audio` tool — allow importing a segment of an audio file instead of the whole thing
- [ ] Fix `note_isolate` — most isolated notes are silent or barely audible; only high-frequency notes (e.g. F5) produce audible output. Likely an issue with the STFT time-frequency masking approach for low-frequency notes.
