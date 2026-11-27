"""Read Pydantic AI's docs on demand, from the repository's own docs/ when it is a Pydantic AI checkout.

The built-in `pydantic_ai_docs` plugin, off until enabled: harness `PydanticAIDocs`, a `read_pyai_docs` tool that
returns one Pydantic AI docs page (capabilities, hooks, tools, toolsets, the agent) as markdown. When the repository
the session works in has a `docs/` folder with every page the tool serves, as a Pydantic AI checkout does, the pages
are read from it through the run's workspace, so they match the code being worked on. Otherwise the capability's
defaults apply: the checkout `PYDANTIC_AI_HARNESS_DOCS_PATH` names, else GitHub at the installed `pydantic_ai`
release's tag.

Pages reach the model as tool output, like a file `read_file` returns, whichever folder trust the repository has.
Like the hosted integrations it is deferred, one line in the capability catalog until the model loads it.
"""

import os
from collections.abc import Sequence
from pathlib import Path

from anyio import to_thread

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.pydantic_ai_docs import PydanticAIDocs
from pydantic_ai_harness.pydantic_ai_docs._toolset import _PAGES  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_pax.config.project_settings import repository_directories
from pydantic_ai_pax.plugins import DepsT, NoSettings, Plugin
from pydantic_ai_pax.runtime.session import WorkspacelessCapability

DESCRIPTION = (
    'Read Pydantic AI documentation (capabilities, hooks, tools, toolsets, agents) before writing or changing '
    'Pydantic AI code.'
)
"""The catalog line the model reads before it loads the tool."""


def local_docs(workspace: Path) -> Path | None:
    """The `docs/` folder at the root of the repository holding `workspace`, when it has every page the tool serves.

    The root is the nearest folder at or above `workspace` with a `.git`, or `workspace` itself outside a repository.
    Requiring every page keeps a project's own `docs/agent.md` from being served as Pydantic AI's.
    """
    docs = repository_directories(workspace)[-1] / 'docs'
    # `os.path.isfile` rather than `Path.is_file`: before Python 3.12 an unreadable folder raises there.
    return docs if all(os.path.isfile(docs / page) for page in _PAGES.values()) else None


class PydanticAIDocsPlugin(Plugin[NoSettings, DepsT]):
    """Harness `PydanticAIDocs`, deferred, reading the repository's own docs when it is a Pydantic AI checkout."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        return (WorkspacelessCapability(self._for_run),)

    async def _for_run(self, _ctx: RunContext[DepsT]) -> PydanticAIDocs[DepsT]:
        # Looked up each run, as the session's folder can change (`/new` in another one under `pax --web`).
        local = await to_thread.run_sync(local_docs, self.host.workspace)
        return PydanticAIDocs[DepsT](local_docs_path=local, defer_loading=True, description=DESCRIPTION)
