from dataclasses import dataclass
from pathlib import Path
import re


DEFAULT_ROOT = Path.home() / ".audio-analysis-mcp" / "workspace"


def sanitize_job_name(filename: str) -> str:
    """Convert a filename to a filesystem-safe, human-readable slug."""
    # Remove file extension
    name = Path(filename).stem
    # Lowercase
    name = name.lower()
    # Replace non-alphanumeric chars with hyphens
    name = re.sub(r"[^a-z0-9]+", "-", name)
    # Collapse multiple hyphens
    name = re.sub(r"-+", "-", name)
    # Trim leading/trailing hyphens
    name = name.strip("-")
    if not name:
        raise ValueError(f"Cannot derive a job name from filename: {filename}")
    return name


@dataclass
class JobContext:
    job_name: str
    stem: str | None = None
    preset: str | None = None


def resolve_job_context(path: str, workspace: "Workspace") -> JobContext:
    """Parse job name, stem, and preset from a path within the workspace.

    Supported path patterns:
      jobs/<job>/source.wav                     → JobContext(job_name)
      jobs/<job>/stems/<preset>/<stem>.wav      → JobContext(job_name, stem, preset)
    """
    p = Path(path).resolve()
    root = workspace.root.resolve()
    try:
        rel = p.relative_to(root / "jobs")
    except ValueError:
        raise ValueError(f"Path is not inside the workspace jobs directory: {path}")

    parts = rel.parts
    if len(parts) < 2:
        raise ValueError(f"Path is not inside a job folder: {path}")

    job_name = parts[0]

    # jobs/<job>/stems/<preset>/<stem>.wav
    if len(parts) >= 4 and parts[1] == "stems":
        preset = parts[2]
        stem = Path(parts[3]).stem
        return JobContext(job_name=job_name, stem=stem, preset=preset)

    # jobs/<job>/source.wav or other direct children
    return JobContext(job_name=job_name)


class Workspace:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or DEFAULT_ROOT

    def _ensure(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- Old flat directories (backward compat) ---

    @property
    def imported(self) -> Path:
        return self._ensure("imported")

    @property
    def stems(self) -> Path:
        return self._ensure("stems")

    @property
    def spectrograms(self) -> Path:
        return self._ensure("spectrograms")

    @property
    def transcriptions(self) -> Path:
        return self._ensure("transcriptions")

    @property
    def isolated_notes(self) -> Path:
        return self._ensure("isolated_notes")

    @property
    def rendered(self) -> Path:
        return self._ensure("rendered")

    # --- Job-centric directories ---

    def job_dir(self, job_name: str) -> Path:
        return self._ensure(f"jobs/{job_name}")

    def job_stems_dir(self, job_name: str, preset: str) -> Path:
        return self._ensure(f"jobs/{job_name}/stems/{preset}")

    def job_transcriptions_dir(self, job_name: str, stem: str, preset: str) -> Path:
        return self._ensure(f"jobs/{job_name}/transcriptions/{stem}_{preset}")

    def job_triage_dir(self, job_name: str, stem: str, preset: str) -> Path:
        return self._ensure(f"jobs/{job_name}/triage/{stem}_{preset}")

    def job_isolated_notes_dir(self, job_name: str, stem: str, preset: str) -> Path:
        return self._ensure(f"jobs/{job_name}/isolated_notes/{stem}_{preset}")
