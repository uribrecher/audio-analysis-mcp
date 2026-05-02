import torch

from audio_analysis_mcp.research.tone_generation.model import ToneGenerationCNN


def test_forward_shapes_cpu():
    model = ToneGenerationCNN()
    mel = torch.randn(8, 1, 128, 30)
    pitch_mh = torch.zeros(8, 88)
    pitch_mh[:, 39] = 1.0  # MIDI 60
    out = model(mel, pitch_mh)
    assert out["shape_logits"].shape == (8, 4)
    assert out["cutoff_norm"].shape == (8,)
    assert out["resonance"].shape == (8,)


def test_forward_outputs_in_range():
    model = ToneGenerationCNN().eval()
    mel = torch.randn(2, 1, 128, 30)
    pitch_mh = torch.zeros(2, 88)
    pitch_mh[:, 39] = 1.0
    with torch.no_grad():
        out = model(mel, pitch_mh)
    assert torch.all(out["cutoff_norm"] >= 0.0) and torch.all(out["cutoff_norm"] <= 1.0)
    assert torch.all(out["resonance"] >= 0.0) and torch.all(out["resonance"] <= 1.0)


def test_backward_runs():
    model = ToneGenerationCNN()
    mel = torch.randn(4, 1, 128, 30, requires_grad=False)
    pitch_mh = torch.zeros(4, 88)
    pitch_mh[:, 40] = 1.0
    out = model(mel, pitch_mh)
    loss = (
        torch.nn.functional.cross_entropy(out["shape_logits"], torch.tensor([0, 1, 2, 3]))
        + torch.nn.functional.mse_loss(out["cutoff_norm"], torch.tensor([0.5, 0.5, 0.5, 0.5]))
        + torch.nn.functional.mse_loss(out["resonance"], torch.tensor([0.5, 0.5, 0.5, 0.5]))
    )
    loss.backward()
    # at least one param should have a non-None gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
