"""One-shot CLI execution without terminal input or stream rendering."""

import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
from typing import TypeVar

from anyio import CancelScope
from rich.console import Console

from pydantic_ai.agent import AbstractAgent
from pydantic_ai.usage import UsageLimits
from pydantic_ai_pax.app import DEFAULT_PLUGINS, STOCK_PLUGINS, create_stock_agent as create_agent
from pydantic_ai_pax.config import Settings
from pydantic_ai_pax.config.project_settings import ProjectSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.errors import error_message
from pydantic_ai_pax.plugins import SessionEndReason, TurnEnd, TurnStart
from pydantic_ai_pax.plugins.loader import SessionPlugins
from pydantic_ai_pax.runtime.imported_sessions import ImportSource
from pydantic_ai_pax.runtime.launch import launch_session_id
from pydantic_ai_pax.runtime.session_runtime import create_shell
from pydantic_ai_pax.runtime.workspace_trust import undecided_notice

DepsT = TypeVar('DepsT')


@asynccontextmanager
async def no_screen() -> AsyncGenerator[None]:
    """Fail before a cooperating plugin can open a terminal widget."""
    raise RuntimeError('User interaction is unavailable in headless mode')
    yield  # pragma: no cover -- async context manager protocol.


def _report_pending(project: ProjectSettings) -> None:
    """Name the project settings left out on stderr: nobody is at a prompt to approve them, and stdout is the answer."""
    if project.pending_notice is not None:
        print(project.pending_notice, file=sys.stderr)


async def run_headless(
    *,
    text: str,
    settings: Settings,
    store: SettingsStore,
    project: ProjectSettings,
    resume: str | None = None,
    resume_from: ImportSource | None = None,
    agent: AbstractAgent[DepsT, object] | None = None,
    deps: DepsT = None,
    session_id: str | None = None,
    fork_session: bool = False,
) -> int:
    """Print only the final answer; preserve sessions and report failures on stderr.

    `resume_from` imports the `resume` session from Claude Code or Codex first.

    A supplied `agent` runs with `deps` and without any plugins, like `chat(..., load_plugins=False)`.
    `resume`, `session_id`, and `fork_session` work as for `chat`, except that `resume` needs an ID.
    """
    if resume == '':
        # The browser needs a terminal, which a headless run has none of.
        raise ValueError('A headless run needs an explicit `resume` (`--resume SESSION-ID`)')
    session_id = launch_session_id(resume=resume, session_id=session_id, fork=fork_session)
    _report_pending(project)
    load_plugins = agent is None
    stock = create_agent() if agent is None else None
    if settings.model is None and (stock.model if stock is not None else agent and agent.model) is None:
        raise ValueError('Choose a model with -m PROVIDER:NAME')
    reason: SessionEndReason = 'error'
    with open(os.devnull, 'w', encoding='utf-8') as sink:
        create = partial(
            create_shell,
            usage_limits=UsageLimits(request_limit=settings.request_limit),
            console=Console(file=sink, force_terminal=False),
            settings=settings,
            store=store,
            builtin_plugins=STOCK_PLUGINS if load_plugins else DEFAULT_PLUGINS,
            project=project,
            headless=True,
            load_plugins=load_plugins,
        )
        # Two calls, so each agent is typed with the deps it gets: the stock agent takes none.
        if agent is not None:
            shell = create(agent, deps=deps, plugins=())
        else:
            assert stock is not None
            shell = create(stock, deps=None, plugins=())
        # Nobody is asked in a headless run, so a folder you never decided about stays untrusted.
        if (notice := undecided_notice(shell.session.trust)) is not None:
            print(notice, file=sys.stderr)
        async with shell.agent:
            with shell.screen.bound(no_screen):  # pragma: no branch -- bound never suppresses exceptions.
                try:
                    # Restore first, so plugins start with the conversation the user asked for.
                    if resume is not None or session_id is not None:
                        await shell.sessions.start(
                            resume=resume, resume_from=resume_from, session_id=session_id, fork=fork_session
                        )
                    await _load_plugins(shell.loader)
                    if resume is not None:
                        # Now that observability has subscribed, under the restored conversation.
                        shell.session.record_resumed()
                    start = TurnStart(text=text)
                    ended = TurnEnd(text=text, outcome='cancelled')
                    try:
                        ended = await shell.run_turn(start, headless=True)
                    finally:
                        with CancelScope(shield=True):
                            await shell.loader.fire(ended)
                    if ended.outcome != 'completed':
                        if ended.error is not None:
                            raise ended.error
                        raise RuntimeError(start.cancel_reason or 'Turn cancelled by a plugin')
                    assert ended.result is not None
                    answer = str(ended.result.output)
                    reason = 'exit'
                except Exception as exc:  # noqa: BLE001 -- CLI boundary, stdout must remain answer-only.
                    Console(stderr=True).print(error_message(exc), markup=False, highlight=False)
                    return 1
                finally:
                    with CancelScope(shield=True):  # pragma: no branch -- this scope is never cancelled.
                        await shell.close(reason)
    print(answer)
    return 0


async def _load_plugins(loader: SessionPlugins[None] | SessionPlugins[DepsT]) -> None:
    """Load every enabled plugin but `ask_user`, skipping one that a plugin loaded before it includes."""
    # Skip before activation, even when a saved declaration overrides the built-in.
    for entry in loader.entries():
        if entry.declaration.enabled and entry.name != 'ask_user':
            # Read afresh: a plugin loaded earlier in this loop may include this one.
            current = next(other for other in loader.entries() if other.name == entry.name)
            if current.included_in is None:
                await loader.load(entry.name)
