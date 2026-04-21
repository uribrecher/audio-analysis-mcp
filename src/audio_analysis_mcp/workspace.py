from pathlib import Path

DEFAULT_ROOT = Path.home() / ".audio-analysis-mcp" / "workspace"


class Workspace:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or DEFAULT_ROOT

    def _ensure(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

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
