"""Tiny helpers for formatting SSE events as ``{"event": "...", "data": ...}``
dicts that ``sse_starlette.EventSourceResponse`` knows how to serialize.

Centralized so endpoints emit a consistent shape and tests have one place to
inspect the wire format.
"""
from __future__ import annotations

import json
from typing import Any


def progress_event(stage: str, fraction: float, detail: str | None = None) -> dict[str, Any]:
    return {
        "event": "progress",
        "data": json.dumps({"stage": stage, "fraction": fraction, "detail": detail}),
    }


def result_event(payload: Any) -> dict[str, Any]:
    """``payload`` should be a Pydantic ``BaseModel`` or dict."""
    if hasattr(payload, "model_dump"):
        body = payload.model_dump()
    elif isinstance(payload, dict):
        body = payload
    else:
        body = {"value": payload}
    return {"event": "result", "data": json.dumps(body)}


def error_event(exc: BaseException) -> dict[str, Any]:
    return {
        "event": "error",
        "data": json.dumps({"type": type(exc).__name__, "message": str(exc)}),
    }
