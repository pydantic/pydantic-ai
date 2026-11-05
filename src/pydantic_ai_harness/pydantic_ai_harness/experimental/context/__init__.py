"""Deprecated import location for `pydantic_ai_harness.repo_context`.

This capability graduated out of `experimental`; importing from here still works but
emits a `DeprecationWarning`. Import from `pydantic_ai_harness.repo_context` instead.
"""

from pydantic_ai_harness.experimental._warn import warn_moved
from pydantic_ai_harness.repo_context import (
    AgentContextInventory,
    AssetRoot,
    ContextFile,
    RepoContext,
    RepoContextToolset,
)

warn_moved('context', 'repo_context')

__all__ = [
    'AgentContextInventory',
    'AssetRoot',
    'ContextFile',
    'RepoContext',
    'RepoContextToolset',
]
