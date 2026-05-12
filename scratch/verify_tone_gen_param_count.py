"""Verify ToneGenerationCNN parameter count claimed by implementer (707,526)."""

from audio_analysis_mcp.research.tone_generation.model import ToneGenerationCNN


def main() -> None:
    model = ToneGenerationCNN()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params: {total}")
    print(f"trainable params: {trainable}")


if __name__ == "__main__":
    main()
