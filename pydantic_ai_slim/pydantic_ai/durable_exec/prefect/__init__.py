try:
    import prefect  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:
    raise ImportError(
        'Please install the `prefect` package to use the Prefect integration, '
        'you can use the `prefect` optional group — `pip install "pydantic-ai-slim[prefect]"`'
    ) from _import_error

from ._agent import PrefectAgent  # pyright: ignore[reportDeprecated]
from ._cache_policies import DEFAULT_PYDANTIC_AI_CACHE_POLICY
from ._durability import PrefectDurability
from ._function_toolset import PrefectFunctionToolset  # pyright: ignore[reportDeprecated]
from ._mcp_toolset import PrefectMCPToolset  # pyright: ignore[reportDeprecated]
from ._model import PrefectModel
from ._types import TaskConfig

__all__ = [
    'PrefectAgent',
    'PrefectDurability',
    'PrefectModel',
    'PrefectMCPToolset',
    'PrefectFunctionToolset',
    'TaskConfig',
    'DEFAULT_PYDANTIC_AI_CACHE_POLICY',
]
