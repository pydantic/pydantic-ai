from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai_harness.cloud_storage._filesystem import _ObjectStoreFilesystem  # pyright: ignore[reportPrivateUsage]


@dataclass(frozen=True)
class MemoryBucket(_ObjectStoreFilesystem):
    """obstore's in-memory store, through the code the three clouds share."""

    _schemes = ('memory',)
    _provider = 'memory'

    url: str = 'memory:///'
