"""One-shot CLI execution without terminal input or stream rendering."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from anyio import CancelScope
from pydantic_ai.usage import UsageLimits
from rich.console import Console

from ._app import DEFAULT_PLUGINS, create_agent, create_shell
from .config import Settings
from .errors import error_message
from .plugins import SessionEndReason, TurnEnd, TurnStart
from .project_settings import ProjectSettings
from .settings_store import SettingsStore


@asynccontextmanager
async def no_screen() -> AsyncGenerator[None]:
    """Fail before a cooperating plugin can open a terminal widget."""
    raise RuntimeError('User interaction is unavailable in headless mode')
    yield  # pragma: no cover -- async context manager protocol.


async def run_headless(
    *, text: str, settings: Settings, store: SettingsStore, project: ProjectSettings, resume: str | None = None
) -> int:
    """Print only the final answer; preserve sessions and report failures on stderr."""
    agent = create_agent()
    if settings.model is None:
        raise ValueError('Choose a model with -m PROVIDER:NAME')
    reason: SessionEndReason = 'error'
    with open(os.devnull, 'w') as sink:
        shell = create_shell(
            agent,
            deps=None,
            plugins=(),
            usage_limits=UsageLimits(request_limit=settings.request_limit),
            console=Console(file=sink, force_terminal=False),
            settings=settings,
            store=store,
            builtin_plugins=DEFAULT_PLUGINS,
            project=project,
            headless=True,
        )
        async with agent:
            with shell.screen.bound(no_screen):  # pragma: no branch -- bound never suppresses exceptions.
                try:
                    # Skip before activation, even when a saved declaration overrides the built-in.
                    for entry in shell.loader.entries():
                        if entry.declaration.enabled and entry.name != 'ask_user':
                            await shell.loader.load(entry.name)
                    if resume is not None:
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
