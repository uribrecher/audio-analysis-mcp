"""Quick param count of ToneGenerationCNN for the Task 9 report."""

from __future__ import annotations

from audio_analysis_mcp.research.tone_generation.model import ToneGenerationCNN


def main() -> None:
    m = ToneGenerationCNN()
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"total params:     {total:,}")
    print(f"trainable params: {trainable:,}")
    # Rough breakdown
    for name, mod in m.named_children():
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name}: {n:,}")


if __name__ == "__main__":
    main()
