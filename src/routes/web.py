"""Web UI routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

# Browsers re-check the page and its scripts on every load (a cheap 304 when nothing changed), so
# after a deploy Safari never keeps running the previous reader.js or app.js from its cache.
REVALIDATE = {"Cache-Control": "no-cache"}


class RevalidatingStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers.update(REVALIDATE)
        return response


def create_router(*, static_dir: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/web", response_class=HTMLResponse)
    async def web_ui():
        index = static_dir / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text(encoding="utf-8"), headers=REVALIDATE)
        return HTMLResponse("<h1>Web UI not found</h1>", status_code=404)

    @router.get("/favicon.ico")
    async def favicon():
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
            '<rect width="32" height="32" rx="7" fill="#111827"/>'
            '<path d="M7 16c0-3 2-6 5-6 4 0 4 5 6 5s2-5 5-5c3 0 5 3 5 6s-2 6-5 6c-4 0-4-5-6-5s-2 5-5 5c-3 0-5-3-5-6z" '
            'fill="none" stroke="#8b5cf6" stroke-width="2" stroke-linecap="round"/>'
            "</svg>"
        )
        return Response(content=svg, media_type="image/svg+xml")

    return router
