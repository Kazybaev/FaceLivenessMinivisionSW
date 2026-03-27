"""FastAPI dependency injection wiring."""
from __future__ import annotations

from functools import lru_cache

from app.infrastructure.config import get_settings
from app.infrastructure.container import Container

_container_override: Container | None = None


def set_container(container: Container | None) -> None:
    global _container_override
    _container_override = container


@lru_cache(maxsize=1)
def _get_default_container() -> Container:
    return Container(get_settings())


def get_container() -> Container:
    if _container_override is not None:
        return _container_override
    return _get_default_container()
