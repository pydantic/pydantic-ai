"""Ordering between capabilities that give a run a workspace only when no other capability does.

Each such fallback asks the run's capabilities for a workspace before supplying its own, so two of them
would each find the other's and both decline. A last-resort fallback, such as `Coder`'s on Windows,
declines while any fallback asks, so it never shadows one, such as CLAI's session directory, that knows better.
"""

from __future__ import annotations

from contextvars import ContextVar

from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

_asking: ContextVar[bool] = ContextVar('pydantic_ai_harness_workspace_fallback_asking', default=False)


def fallback_asking() -> bool:
    """Whether a fallback is asking the run's capabilities for a workspace; a last-resort fallback then declines."""
    return _asking.get()


def other_workspace(ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
    """The workspace the run's capabilities supply, with last-resort fallbacks declining meanwhile.

    The caller declines itself while it asks. Selection does no I/O, so asking more than once is harmless.
    """
    assert ctx.root_capability is not None, 'core sets the root capability before asking a capability function'
    token = _asking.set(True)
    try:
        return ctx.root_capability.get_workspace(ctx, ref=ref)
    finally:
        _asking.reset(token)
