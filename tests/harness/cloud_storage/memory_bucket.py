from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from obstore.store import MemoryStore

from pydantic_ai_harness.cloud_storage._filesystem import _ObjectStoreFilesystem  # pyright: ignore[reportPrivateUsage]


@dataclass(frozen=True)
class MemoryBucket(_ObjectStoreFilesystem):
    """obstore's in-memory store, through the code the three clouds share."""

    _schemes = ('memory',)
    _provider = 'memory'
    url: str = 'memory:///'

    @cached_property
    def _location(self) -> tuple[str, str]:
        return 'memory', ''

    @staticmethod
    def _open(*args: Any, **kwargs: Any) -> MemoryStore:
        return MemoryStore()
