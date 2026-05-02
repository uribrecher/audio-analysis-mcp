"""Smoke test: verify PyTorch MPS backend on this machine (M3 Pro).

Task 2 of subtractive-tone-training-mvp plan. Runs a tiny CNN forward +
backward + 50-step training loop on the auto-detected device (mps > cuda > cpu)
and reports timing + loss decrease so we know whether MPS is healthy and fast
enough for the spec's 5-min GPU training target before we commit to a model
architecture in Task 9.

The script never fails on missing MPS — it falls back to CPU and reports
honestly. The whole point is to discover the truth on this machine.
"""

from __future__ import annotations

import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def pick_device() -> torch.device:
    """Auto-detect best device. Prefer MPS on Apple Silicon, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# TinyCNN — shape-compatible with the planned mel-spectrogram input
# ---------------------------------------------------------------------------
# Input shape (B, 1, 128, 30):
#   B  = batch size (8 in this smoke test)
#   1  = single channel (mono mel spectrogram)
#   128 = mel bins (matches the plan's typical n_mels)
#   30 = time frames (a sustain-region slice)
# Output: 4 logits — placeholder for the engine-classification softmax head.
class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2)
        # After conv (same padding) -> (B, 8, 128, 30)
        # After pool(2) -> (B, 8, 64, 15)
        self.fc = nn.Linear(8 * 64 * 15, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("PyTorch MPS smoke test")
    print("=" * 60)
    print(f"torch version:      {torch.__version__}")
    print(f"MPS available:      {torch.backends.mps.is_available()}")
    print(f"MPS built:          {torch.backends.mps.is_built()}")
    print(f"CUDA available:     {torch.cuda.is_available()}")

    device = pick_device()
    print(f"device:             {device}")

    if device.type == "cpu":
        print("\nNOTE: MPS unavailable. Falling back to CPU. Training will be slower.")

    # Try to set deterministic mode — known to be flaky on MPS depending on
    # PyTorch version. Don't crash the smoke test if it warns/errors.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:  # noqa: BLE001 — diagnostic only
        print(f"deterministic flag warn/err: {type(e).__name__}: {e}")

    torch.manual_seed(0)

    # ---------------- Forward + shape check ----------------
    model = TinyCNN(num_classes=4).to(device)
    x = torch.randn(8, 1, 128, 30, device=device)
    y_target = torch.randint(0, 4, (8,), device=device)

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        out = model(x)

    print(f"\nforward output shape: {tuple(out.shape)} (expected (8, 4))")
    print(f"forward output dtype: {out.dtype}")
    assert out.shape == (8, 4), f"unexpected output shape {out.shape}"

    # ---------------- Backward sanity ----------------
    loss_initial = F.cross_entropy(out, y_target)
    loss_initial.backward()
    grad_present = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    print(f"backward produced gradients: {grad_present}")
    assert grad_present, "no gradients flowed — backward is broken on this device"

    # ---------------- 50-step training loop ----------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Reseed and build a fresh fixed (x, y) so the loss curve is comparable
    # run-to-run. We're optimizing on a single batch — loss should decrease
    # quickly, which proves the backward + optimizer step path works.
    torch.manual_seed(0)
    x_train = torch.randn(8, 1, 128, 30, device=device)
    y_train = torch.randint(0, 4, (8,), device=device)

    model_train = TinyCNN(num_classes=4).to(device)
    optimizer = torch.optim.Adam(model_train.parameters(), lr=1e-3)

    # Warm-up step — first MPS kernel compile is one-time and would skew the
    # ms/step timing if included.
    out0 = model_train(x_train)
    loss0 = F.cross_entropy(out0, y_train)
    optimizer.zero_grad()
    loss0.backward()
    optimizer.step()

    initial_loss = loss0.item()

    # Sync before timing — MPS dispatches async, like CUDA.
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    final_loss = float("nan")
    for step in range(50):
        optimizer.zero_grad()
        out_t = model_train(x_train)
        loss_t = F.cross_entropy(out_t, y_train)
        loss_t.backward()
        optimizer.step()
        final_loss = loss_t.item()

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t_start
    ms_per_step = (elapsed / 50) * 1000

    print(f"\n50-step train loop:   {elapsed * 1000:.1f} ms total   ({ms_per_step:.2f} ms/step)")
    print(f"loss before / after:  {initial_loss:.4f}  →  {final_loss:.4f}")

    if final_loss >= initial_loss:
        print("\nFAIL: loss did not decrease — training is broken on this device.")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print(f"MPS smoke test passed (device={device}, {ms_per_step:.2f} ms/step)")
    print("=" * 60)


if __name__ == "__main__":
    main()
