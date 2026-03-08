"""Web UI routes."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import APIRouter
from fastapi.responses import HTMLResponse


def create_router(*, static_dir: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/web", response_class=HTMLResponse)
    async def web_ui():
        index = static_dir / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text())
        return HTMLResponse("<h1>Web UI not found</h1>", status_code=404)

    return router
