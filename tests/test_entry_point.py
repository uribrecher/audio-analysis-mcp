from audio_analysis_mcp.__main__ import main, _register_tools
from audio_analysis_mcp.server import mcp

EXPECTED_TOOLS = {
    "import_audio", "stem_separate", "audio_list_devices", "audio_render",
    "spectrum_analyze", "audio_compare", "note_transcribe", "note_triage",
    "note_isolate", "amplitude_analyze", "structure_analyze",
}


async def test_register_tools_registers_all() -> None:
    _register_tools()
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS <= names


def test_main_is_callable() -> None:
    assert callable(main)
