from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.audio.capture import list_audio_devices, capture_audio
from audio_analysis_mcp.schemas import AudioDevice, AudioRenderResult, ListAudioDevicesResult


@mcp.tool()
def audio_list_devices() -> str:
    """List available audio input devices."""
    raw = list_audio_devices()
    result = ListAudioDevicesResult(
        devices=[
            AudioDevice(
                name=d.get("name", ""),
                index=d.get("index", 0),
                max_input_channels=d.get("max_input_channels", 0),
            )
            for d in raw
        ]
    )
    return result.model_dump_json(indent=2)


@mcp.tool()
def audio_render(
    duration: float = 5.0,
    device: str | int | None = None,
) -> str:
    """Capture audio from a system audio device (BlackHole, USB audio).

    macOS note: the host process must have Microphone permission in
    System Settings > Privacy & Security. This applies to virtual devices
    like BlackHole identically to physical microphones.
    """
    ws = get_workspace()
    path = capture_audio(duration=duration, output_dir=ws.rendered, device=device)

    return AudioRenderResult(
        audio_path=path,
        duration_seconds=duration,
        device="default" if device is None else str(device),
        sample_rate=44100,
    ).model_dump_json(indent=2)
