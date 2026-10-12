"""One-shot CLI execution without terminal input or stream rendering."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, nullcontext

from anyio import CancelScope
from rich.console import Console

from pydantic_ai.agent import AbstractAgent
from pydantic_ai.usage import UsageLimits
from pydantic_clai2._app import DEFAULT_PLUGINS, STOCK_PLUGINS, create_shell, create_stock_agent as create_agent
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.errors import error_message
from pydantic_clai2.plugins import SessionEndReason, TurnEnd, TurnStart
from pydantic_clai2.runtime.imported_sessions import ImportSource


@asynccontextmanager
async def no_screen() -> AsyncGenerator[None]:
    """Fail before a cooperating plugin can open a terminal widget."""
    raise RuntimeError('User interaction is unavailable in headless mode')
    yield  # pragma: no cover -- async context manager protocol.


async def run_headless(
    *,
    text: str,
    settings: Settings,
    store: SettingsStore,
    project: ProjectSettings,
    resume: str | None = None,
    resume_from: ImportSource | None = None,
    agent: AbstractAgent[None, object] | None = None,
) -> int:
    """Print only the final answer; preserve sessions and report failures on stderr.

    `resume_from` imports the `resume` session from Claude Code or Codex first.

    A supplied `agent` runs without any plugins, like `chat(..., load_plugins=False)`.
    """
    load_plugins = agent is None
    agent = create_agent() if agent is None else agent
    if settings.model is None and agent.model is None:
        raise ValueError('Choose a model with -m PROVIDER:NAME')
    reason: SessionEndReason = 'error'
    with open(os.devnull, 'w', encoding='utf-8') as sink:
        shell = create_shell(
            agent,
            deps=None,
            plugins=(),
            usage_limits=UsageLimits(request_limit=settings.request_limit),
            console=Console(file=sink, force_terminal=False),
            settings=settings,
            store=store,
            builtin_plugins=STOCK_PLUGINS if load_plugins else DEFAULT_PLUGINS,
            project=project,
            headless=True,
            load_plugins=load_plugins,
        )
        async with agent:
            with shell.screen.bound(no_screen):  # pragma: no branch -- bound never suppresses exceptions.
                try:
                    with shell.defer_identity() if resume is not None else nullcontext():
                        # Skip before activation, even when a saved declaration overrides the built-in.
                        for entry in shell.loader.entries():
                            if entry.declaration.enabled and entry.name != 'ask_user':
                                # Read afresh: a plugin loaded earlier in this loop may include this one.
                                current = next(other for other in shell.loader.entries() if other.name == entry.name)
                                if current.included_in is None:
                                    await shell.loader.load(entry.name)
                        if resume is not None:
                            if resume_from is not None:
                                resume = await shell.sessions.import_session(resume_from, resume)
                            await shell.session.resume(resume)
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
                        await shell.loader.close(reason)
    print(answer)
    return 0
