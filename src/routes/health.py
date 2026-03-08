"""Health and metadata routes."""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter

from src.services import models as model_service


def create_router(*, get_runtime_version: Callable[[], str], get_backend_router: Callable) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health():
        return model_service.health_response(
            version=get_runtime_version(),
            backend_router=get_backend_router(),
        )

    return router
