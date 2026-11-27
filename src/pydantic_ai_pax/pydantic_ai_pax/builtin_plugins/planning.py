"""Let the model keep a todo plan for the conversation, drawn in the transcript as it changes.

The built-in `planning` plugin binds harness `Planning` with a `SqlitePlanStore` in `plan.db`, beside
`sessions.db`, holding one plan per saved conversation: `/resume` brings the plan back, while `/new` and a
fork start without one. It is off by default, since its tools and guidance go with every request.
"""

import os
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from anyio import to_thread
from rich.console import Group, RenderableType
from rich.text import Text

from pydantic_ai import AgentStreamEvent, FunctionToolResultEvent, RunContext
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.planning import (
    InMemoryPlanStore,
    PlanCreatedEvent,
    PlanDeletedEvent,
    PlanItem,
    Planning,
    PlanningToolset,
    PlanStore,
    PlanUpdatedEvent,
    SqlitePlanStore,
    TaskStatus,
)
from pydantic_ai_pax.plugins import DepsT, NoSettings, Plugin, PluginHost, SessionStart
from pydantic_ai_pax.ui.rendering import theme
from pydantic_ai_pax.ui.rendering.tool_output import terminal_text

PLAN_DATABASE = 'plan.db'
"""The plans' database, in the folder that holds the settings database and `sessions.db`."""

_CHANGES = (PlanCreatedEvent, PlanUpdatedEvent, PlanDeletedEvent)
"""Every change a plan tool makes emits one of these; a status change is an update first."""

_MARKS = {
    TaskStatus.pending: ('◻', ''),
    TaskStatus.in_progress: ('◼', theme.ACCENT),
    TaskStatus.completed: ('✓', theme.MUTED),
    TaskStatus.cancelled: ('✗', theme.MUTED),
    TaskStatus.blocked: ('!', theme.WARNING),
}
"""Each status's mark and the theme role its step is painted in; `''` keeps the terminal's colour."""


class _ShownPlanStore(SqlitePlanStore):
    """A conversation's plan in `plan.db`, keeping the steps it last read for the panel to draw."""

    shown: Sequence[PlanItem] = ()

    async def get_items(self) -> list[PlanItem]:
        # Every plan tool reads the plan back after its last change, before it reports the change.
        self.shown = await super().get_items()
        return self.shown


def render_plan_panel(items: Sequence[PlanItem]) -> Group:
    """The plan under the plan tool's header, a row per step: its mark, and the running one by its present-tense label."""
    rows: list[Text] = []
    for item in items:
        mark, role = _MARKS[item.status]
        label = item.active_form if item.status is TaskStatus.in_progress and item.active_form else item.content
        style = theme.color(role)
        row = Text(f'  {mark} ', style=style, overflow='ellipsis', no_wrap=True)
        if item.status in (TaskStatus.completed, TaskStatus.cancelled):
            style += ' strike'
        row.append(terminal_text(label, keep=''), style=style)
        rows.append(row)
    return Group(*rows)


def _create_private(path: Path) -> None:
    """Create `path` owner-only, like `sessions.db`, unless it exists: the plans are about the user's work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with suppress(FileExistsError):
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))


class PlanningPlugin(Plugin[NoSettings, DepsT]):
    """Harness `Planning` with a plan per saved conversation, and the plan drawn after each plan tool that changes it."""

    def __init__(self, host: PluginHost[DepsT], settings: NoSettings) -> None:
        super().__init__(host, settings)
        database = host.settings_database
        self._database = None if database is None else database.with_name(PLAN_DATABASE)
        self._planning = Planning[DepsT](store_resolver=self._store)
        toolset = self._planning.get_toolset()
        assert isinstance(toolset, PlanningToolset)
        self._tools = frozenset(toolset.tools)
        self._shown: _ShownPlanStore | None = None
        self._changed = False

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        return (self._planning,)

    async def on_session_start(self, event: SessionStart) -> None:
        del event
        if self._database is not None:
            await to_thread.run_sync(_create_private, self._database)

    def _store(self, ctx: RunContext[DepsT]) -> PlanStore:
        """The plan of the conversation on screen; any other run, such as a delegated task's, plans in memory."""
        conversation_id = self.host.conversation.conversation_id
        if self._database is None or ctx.conversation_id != conversation_id:
            return InMemoryPlanStore()
        self._shown = _ShownPlanStore(str(self._database), session=conversation_id)
        return self._shown

    def render(self, event: AgentStreamEvent) -> RenderableType | None:
        if isinstance(event, _CHANGES):
            self._changed = True
            return None
        if (
            isinstance(event, FunctionToolResultEvent)
            and event.part.tool_name in self._tools
            and self._changed
            and self._shown is not None
        ):
            self._changed = False
            return render_plan_panel(self._shown.shown)
        return None
