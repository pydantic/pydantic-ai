"""Let the model keep notes that later sessions in the same project recall; adds /memory.

The built-in `memory` plugin: harness `Memory` over a `FileMemoryStore` in Pax's state folder, one folder per
project. A project is what folder trust is kept for: a repository's main checkout, so its worktrees share one
memory, or else the session's folder. Your home folder and the filesystem root are no project, so a session there
gets no memory.
"""

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import anyio
import anyio.to_thread

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.tools import RunContext
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.memory import FileMemoryStore, Memory
from pydantic_ai_pax.commands import Command
from pydantic_ai_pax.paths import state_dir
from pydantic_ai_pax.plugins import DepsT, NoSettings, Plugin
from pydantic_ai_pax.runtime.session import WorkspacelessCapability, supports_local_workspace
from pydantic_ai_pax.runtime.workspace_trust import trust_scope

AGENT_NAME = 'main'
"""`Memory`'s `agent_name`, the folder inside a project's memory folder that holds `MEMORY.md`."""

_UNSAFE = re.compile(r'[^a-z0-9_.-]+')


def memory_root() -> Path:
    """The folder that holds every project's memory folder."""
    return state_dir() / 'memory'


def project_folder(workspace: Path) -> str | None:
    """The name of the memory folder for a session working in `workspace`, or `None` when it is in no project.

    The project's folder name, for reading, then a hash of its full path, so two projects with one name stay apart.
    Lowercase, so a case-insensitive filesystem cannot merge two names that differ in case only.
    """
    scope = trust_scope(workspace)
    if not scope.remembered:
        # Home and `/`: memory kept there would collect notes from whatever unrelated work starts there.
        return None
    name = _UNSAFE.sub('-', scope.key.name.lower()).strip('.-') or 'project'
    digest = hashlib.sha256(str(scope.key).encode()).hexdigest()[:16]
    return f'{name}-{digest}'


class MemoryPlugin(Plugin[NoSettings, DepsT]):
    """Harness `Memory` with a `FileMemoryStore` in `memory_root()`, in the session's project's folder."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        if not supports_local_workspace():
            # `FileMemoryStore` replaces a file with `mv`, which a `WindowsWorkspaceBackend` doesn't have.
            return ()
        # The id qualifies the memory toolset's `directory` instruction block, which the wrapper's toolset carries.
        return (WorkspacelessCapability(self._for_run, id='memory'),)

    def get_commands(self) -> Sequence[Command]:
        return (Command(name='memory', description="Show where this project's memory is kept", handler=self._show),)

    async def _for_run(self, _ctx: RunContext[DepsT]) -> Memory[DepsT] | None:
        folder = await anyio.to_thread.run_sync(project_folder, self.host.workspace)
        if folder is None:
            return None
        root = memory_root()
        await anyio.Path(root).mkdir(mode=0o700, parents=True, exist_ok=True)
        return Memory[DepsT](
            store=FileMemoryStore(folder, workspace=LocalWorkspaceBackend(root)), agent_name=AGENT_NAME
        )

    async def _show(self, _args: list[str]) -> str:
        if not supports_local_workspace():
            return 'Memory is not available on Windows yet.'
        folder = await anyio.to_thread.run_sync(project_folder, self.host.workspace)
        if folder is None:
            return 'No memory here: your home folder and `/` are no project. Start Pax in a project folder.'
        return (
            f"This project's memory is in {memory_root() / folder / AGENT_NAME}: `MEMORY.md` and any other "
            'notes the model saved. Edit or delete them there; delete the folder to forget everything.'
        )
