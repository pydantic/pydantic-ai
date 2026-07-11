"""Deprecated import location for `pydantic_ai_harness.context`.

This capability graduated out of `experimental`; importing from here still works but
emits a `DeprecationWarning`. Import from `pydantic_ai_harness.context` instead.
"""

from pydantic_ai_harness.context import (
    AgentContextInventory,
    AssetRoot,
    ContextFile,
    RepoContext,
    RepoContextToolset,
)
from pydantic_ai_harness.experimental._warn import warn_moved

warn_moved('context', 'context')

__all__ = [
    'AgentContextInventory',
    'AssetRoot',
    'ContextFile',
    'RepoContext',
    'RepoContextToolset',
]
