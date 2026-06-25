"""Context capability: discover and load a repo's accumulated context engineering."""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.context._capability import RepoContext
from pydantic_ai_harness.experimental.context._inventory import AgentContextInventory, AssetRoot
from pydantic_ai_harness.experimental.context._loader import ContextFile
from pydantic_ai_harness.experimental.context._toolset import RepoContextToolset

warn_experimental('context')

__all__ = ['AgentContextInventory', 'AssetRoot', 'ContextFile', 'RepoContext', 'RepoContextToolset']
