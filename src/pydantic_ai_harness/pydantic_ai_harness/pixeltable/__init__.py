"""Pixeltable integration: read-only catalog tools, and a `MemoryStore` backed by a Pixeltable table."""

try:
    import pixeltable as _pixeltable  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pixeltable is required for the Pixeltable integration. Install it with: pip install "pydantic-ai-harness[pixeltable]"'
    ) from _import_error

from pydantic_ai_harness.pixeltable._capability import Pixeltable
from pydantic_ai_harness.pixeltable._store import PixeltableMemoryStore
from pydantic_ai_harness.pixeltable._toolset import PixeltableToolset

__all__ = ['Pixeltable', 'PixeltableMemoryStore', 'PixeltableToolset']
