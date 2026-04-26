"""Plot the envelope curves the amplitude analyzer produced for the
van-halen-jump triage, alongside the audio waveform of each cluster slice.

Re-runs the same envelope/fit/isolate pipeline so the plot can show the
detected ADSR boundaries (attack-end, sustain-start, sustain-end) and the
raw audio waveform underneath.

Run: uv run python scratch/plot_amplitude_envelopes.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from audio_analysis_mcp.analysis.envelope import extract_rms_envelope
from audio_analysis_mcp.analysis.adsr_fit import fit_adsr
from audio_analysis_mcp.schemas import NoteTriageFileData

JOB = Path(
    "/Users/uribrecher/.audio-analysis-mcp/workspace/jobs/"
    "van-halen-jump-lyrics-bugg-lyrics"
)
TRIAGE = JOB / "triage/other_fast/triage.json"
STEM = JOB / "stems/fast/other.wav"
OUT_PNG = Path("/tmp/van-halen-amplitude-envelopes.png")


def main() -> None:
    audio, sr = sf.read(STEM, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("float32")
    print(f"audio: {audio.size} samples @ {sr}Hz ({audio.size / sr:.2f}s)")

    file_data = NoteTriageFileData.model_validate_json(TRIAGE.read_text())
    # Restrict to the 3 clusters that produced ADSR (indices 0, 1, 3 from earlier run).
    target_indices = [0, 1, 3]

    fig, axes = plt.subplots(len(target_indices), 1, figsize=(14, 4 * len(target_indices)))
    if len(target_indices) == 1:
        axes = [axes]

    for ax, cluster_idx in zip(axes, target_indices):
        cluster = file_data.candidates[cluster_idx]
        start_sample = max(0, int(cluster.start_time * sr))
        end_sample = min(audio.size, int(cluster.end_time * sr))
        clip = audio[start_sample:end_sample]

        env_result = extract_rms_envelope(clip, sample_rate=sr)
        fit = fit_adsr(env_result.envelope, envelope_sample_rate=env_result.envelope_sample_rate)

        # Time axes
        t_audio_ms = np.arange(clip.size) / sr * 1000.0
        t_env_ms = np.arange(env_result.envelope.size) / env_result.envelope_sample_rate * 1000.0

        peak_idx = int(np.argmax(env_result.envelope))
        peak_ms = peak_idx / env_result.envelope_sample_rate * 1000.0
        sustain_start_ms = fit.sustain_start_idx / env_result.envelope_sample_rate * 1000.0
        sustain_end_ms = fit.sustain_end_idx / env_result.envelope_sample_rate * 1000.0
        # The "attack-start" frame: first frame the envelope crossed 5% of peak.
        peak_v = float(env_result.envelope.max())
        above = np.where(env_result.envelope[:peak_idx + 1] >= 0.05 * peak_v)[0]
        attack_start_ms = (int(above[0]) if above.size > 0 else 0) / env_result.envelope_sample_rate * 1000.0
        # Release-end: first frame past sustain_end where envelope < 5% peak
        tail = env_result.envelope[fit.sustain_end_idx:]
        drops = np.where(tail < 0.05 * peak_v)[0]
        release_end_ms = ((fit.sustain_end_idx + int(drops[0])) if drops.size > 0
                          else env_result.envelope.size - 1) / env_result.envelope_sample_rate * 1000.0

        # Plot raw audio (light) + envelope (bold)
        ax.plot(t_audio_ms, clip, color="#bbbbbb", linewidth=0.5, label="audio waveform")
        ax.plot(t_env_ms, env_result.envelope, color="#1f77b4", linewidth=2.0, label="RMS envelope")
        # ADSR boundaries
        ax.axvline(attack_start_ms, color="#2ca02c", linestyle=":", label=f"attack start ({attack_start_ms:.0f}ms)")
        ax.axvline(peak_ms, color="#ff7f0e", linestyle="--", label=f"peak / attack end ({peak_ms:.0f}ms)")
        ax.axvline(sustain_start_ms, color="#d62728", linestyle="--", label=f"sustain start ({sustain_start_ms:.0f}ms)")
        ax.axvline(sustain_end_ms, color="#9467bd", linestyle="--", label=f"sustain end ({sustain_end_ms:.0f}ms)")
        ax.axvline(release_end_ms, color="#8c564b", linestyle=":", label=f"release end ({release_end_ms:.0f}ms)")

        title = (
            f"cluster #{cluster_idx} {cluster.kind} t=[{cluster.start_time:.2f},{cluster.end_time:.2f}]s | "
            f"A={fit.attack_ms:.0f}ms D={fit.decay_ms:.0f}ms S={fit.sustain_level:.2f} R={fit.release_ms:.0f}ms"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time (ms within slice)")
        ax.set_ylabel("amplitude")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=110)
    print(f"saved → {OUT_PNG}")


if __name__ == "__main__":
    main()