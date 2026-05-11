"""Smoke test for the /healthz route. Verifies the FastAPI app boots and
the route handler returns the expected shape."""
import pytest
from httpx import ASGITransport, AsyncClient

from audio_analysis_mcp.service.app import app


@pytest.mark.asyncio
async def test_healthz_ok() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
