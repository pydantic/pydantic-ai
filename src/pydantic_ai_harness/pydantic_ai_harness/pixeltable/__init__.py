"""Pixeltable integration: read-only catalog tools. The memory store is `pydantic_ai_harness.memory.PixeltableMemoryStore`."""

try:
    import pixeltable as _pixeltable  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:
    raise ImportError(
        'pixeltable is required for the Pixeltable integration. It needs Python 3.11 or later; install it with: pip install "pydantic-ai-harness[pixeltable]"'
    ) from _import_error

from pydantic_ai_harness.pixeltable._capability import Pixeltable
from pydantic_ai_harness.pixeltable._toolset import PixeltableToolset

__all__ = ['Pixeltable', 'PixeltableToolset']
