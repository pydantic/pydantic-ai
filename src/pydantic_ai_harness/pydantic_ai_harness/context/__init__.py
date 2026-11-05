"""Deprecated import location for `pydantic_ai_harness.repo_context`.

This module was renamed; importing from here still works but emits a
`DeprecationWarning`. Import from `pydantic_ai_harness.repo_context` instead.
"""

from pydantic_ai_harness._warn import warn_module_renamed
from pydantic_ai_harness.repo_context import (
    AgentContextInventory,
    AssetRoot,
    ContextFile,
    RepoContext,
    RepoContextToolset,
)

warn_module_renamed('context', 'repo_context')

__all__ = [
    'AgentContextInventory',
    'AssetRoot',
    'ContextFile',
    'RepoContext',
    'RepoContextToolset',
]
