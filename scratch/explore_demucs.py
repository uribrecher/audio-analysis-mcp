"""Explore the demucs v4.0.1 Python API for stem separation."""
import inspect
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs import audio as demucs_audio

# What's in demucs.audio?
print("=== demucs.audio exports ===")
for name in sorted(dir(demucs_audio)):
    if not name.startswith("_"):
        obj = getattr(demucs_audio, name)
        if callable(obj):
            try:
                print(f"  {name}{inspect.signature(obj)}")
            except (ValueError, TypeError):
                print(f"  {name} (no signature)")

# How to load audio
print("\n=== Loading audio ===")
from demucs.audio import AudioFile
af = AudioFile("../tests/__init__.py")  # just checking the class
print(f"AudioFile type: {type(af)}")
print(f"AudioFile members: {[m for m in dir(af) if not m.startswith('_')]}")

# Check save_audio
print("\n=== save_audio ===")
print(f"save_audio signature: {inspect.signature(demucs_audio.save_audio)}")

# Model info
print("\n=== Model info ===")
model = get_model("htdemucs")
print(f"sources: {model.sources}")
print(f"samplerate: {model.samplerate}")
print(f"audio_channels: {model.audio_channels}")

# apply_model info
print(f"\n=== apply_model ===")
print(f"signature: {inspect.signature(apply_model)}")
