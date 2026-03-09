"""FastAPI dependency injection wiring."""
from __future__ import annotations

from functools import lru_cache

from app.infrastructure.config import get_settings
from app.infrastructure.container import Container


@lru_cache(maxsize=1)
def get_container() -> Container:
    return Container(get_settings())
