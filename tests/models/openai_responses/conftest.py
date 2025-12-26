from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    pass


def pytest_configure(config: pytest.Config) -> None:
    """Add filterwarnings for deprecated builtin tool events."""
    config.addinivalue_line(
        'filterwarnings',
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning',
    )
    config.addinivalue_line(
        'filterwarnings',
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning',
    )


async def cleanup_openai_resources(file: Any, vector_store: Any, async_client: Any) -> None:  # pragma: lax no cover
    """Helper function to clean up OpenAI file search resources if they exist."""
    if file is not None:
        await async_client.files.delete(file.id)
    if vector_store is not None:
        await async_client.vector_stores.delete(vector_store.id)
    await async_client.close()
