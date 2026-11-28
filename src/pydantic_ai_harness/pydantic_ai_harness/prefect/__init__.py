"""Prefect durable execution: the canonical import path for `PrefectDurability` and its supporting types.

The code lives in `pydantic_ai.durable_exec.prefect` until Pydantic AI v3, when it moves here. Both paths
give the same objects, and that path keeps working until then. The deprecated `PrefectAgent`,
`PrefectFunctionToolset` and `PrefectMCPToolset` are not re-exported: they stay importable from
`pydantic_ai.durable_exec.prefect` until v3.
"""

from pydantic_ai.durable_exec.prefect import (
    DEFAULT_PYDANTIC_AI_CACHE_POLICY,
    PrefectDurability,
    PrefectModel,
    PrefectOperationNamer,
    TaskConfig,
)

__all__ = [
    'DEFAULT_PYDANTIC_AI_CACHE_POLICY',
    'PrefectDurability',
    'PrefectModel',
    'PrefectOperationNamer',
    'TaskConfig',
]
