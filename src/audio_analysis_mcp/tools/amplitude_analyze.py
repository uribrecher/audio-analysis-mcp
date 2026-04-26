from pathlib import Path

import soundfile as sf

from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.workspace import resolve_job_context
from audio_analysis_mcp.analysis.amplitude import analyze_amplitude


@mcp.tool()
def amplitude_analyze(
    audio_path: str,
    triage_path: str,
) -> str:
    """Per-cluster ADSR analysis with cross-candidate consistency check.

    NOT READY FOR PRODUCTION. Manual testing on real synth audio (Van
    Halen "Jump") showed the heuristic ADSR fitter is meaningfully off in
    every component. See docs/TODO.md for the full list. Top issues:

      - Attack end is taken at the RMS peak, but on real synth tones the
        true attack ends earlier — at the RMS-slope inflection (where the
        second derivative crosses zero). Peak-based attack is inflated.
      - Decay segment is forced into the model. Many real synth patches
        have no decay; sustain begins at the attack-end inflection.
      - Sustain end is detected too late because noise floor sustains the
        envelope past the real signal's drop-off.
      - Release length is consequently dominated by noise floor — values
        come out 2-5× too long.

    Use the result for debugging / scaffolding only. Real ADSR estimation
    needs noise-floor handling and inflection-based segmentation; planned
    research follows prior work.

    Inputs:
      audio_path:  WAV file at jobs/<job>/stems/<preset>/<stem>.wav
      triage_path: triage.json from note_triage (containing CandidateClusters)

    Writes per-cluster envelope.npy and sustain.wav under
    jobs/<job>/amplitude/<stem>_<preset>/cluster_<idx>_<kind>/.
    Returns JSON-serialized AmplitudeAnalyzeResult.
    """
    ws = get_workspace()
    ctx = resolve_job_context(audio_path, ws)
    if ctx.stem is None or ctx.preset is None:
        raise ValueError(
            f"Expected a stem file (jobs/<job>/stems/<preset>/<stem>.wav), got: {audio_path}"
        )

    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("float32")

    output_dir = ws.job_amplitude_dir(ctx.job_name, ctx.stem, ctx.preset)
    result = analyze_amplitude(
        audio=audio,
        sample_rate=int(sample_rate),
        triage_path=Path(triage_path),
        output_dir=output_dir,
    )
    return result.model_dump_json(indent=2)
