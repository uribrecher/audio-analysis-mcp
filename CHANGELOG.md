## v0.1.1 (2026-06-19)

### Refactor

- remove tone-generation training pipeline + signalflow dep (#61)

## v0.1.0 (2026-06-17)

### Feat

- **service**: add /jobs/triage SSE endpoint (#26)
- **analysis**: triage_notes_by_sections — per-section variant (#25)
- **service**: add /jobs/transcribe SSE endpoint (Basic Pitch -> MIDI) (#24)
- HTTP+SSE service for slow audio-analysis ops (#19)
- add structure_analyze MCP tool backed by SongFormer (#18)

### Fix

- cap .glow width on mobile to prevent horizontal overflow (#45)
- **service**: drop pipeline singletons after each request; fix stems progress (#23)
