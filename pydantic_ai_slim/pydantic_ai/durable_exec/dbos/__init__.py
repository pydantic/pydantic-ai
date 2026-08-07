from __future__ import annotations

try:
    import dbos  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:
    raise ImportError(
        'Please install the `dbos` package to use the DBOS integration, '
        'you can use the `dbos` optional group — `pip install "pydantic-ai-slim[dbos]"`'
    ) from _import_error

from ._agent import DBOSAgent, DBOSParallelExecutionMode  # pyright: ignore[reportDeprecated]
from ._durability import DBOSDurability
from ._model import DBOSModel
from ._utils import StepConfig

__all__ = ['DBOSAgent', 'DBOSDurability', 'DBOSModel', 'DBOSParallelExecutionMode', 'StepConfig']
