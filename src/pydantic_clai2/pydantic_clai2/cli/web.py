"""`clai2 --web`: CLAI with no terminal prompt, driven only from the web UI, logging the conversation as it goes."""

import os
from collections.abc import Callable
from pathlib import Path

from anyio import CancelScope, create_task_group, get_cancelled_exc_class
from rich.console import Console

from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2._app import STOCK_PLUGINS, create_shell, create_stock_agent
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings, load_project_settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import SessionEndReason
from pydantic_clai2.runtime.imported_sessions import ImportSource
from pydantic_clai2.runtime.session_workspace import SessionWorkspace, workspace_directory

WEB = 'web'
"""The built-in plugin `--web` loads for this session, whether or not it is enabled."""


async def run_web(
    *,
    settings_for: Callable[[ProjectSettings], Settings],
    store: SettingsStore,
    project: ProjectSettings,
    console: Console,
    resume: str | None = None,
    resume_from: ImportSource | None = None,
    session_id: str | None = None,
    fork_session: bool = False,
) -> None:
    """Load the stock plugins and `web`, start the web UI, and run turns the browser enters until Ctrl-C.

    `console` gets the transcript as plain text. Nothing saved changes: `web` loads for this launch only.
    `resume`, `session_id`, and `fork_session` work as for `chat`, except that `resume` needs an ID.

    `project` is the launch directory's project file and `settings_for` the settings CLAI starts with
    beside a project file. Each session runs in its own directory: `resume` starts there, `/resume ID`
    moves there, and `/workspace DIRECTORY` starts a new session in another one (see `SessionWorkspace`).
    """
    if resume == '':
        raise ValueError('--web needs an explicit --resume SESSION-ID; the sidebar lists the others')
    if resume and resume_from is None:
        project = await _start_in(store=store, conversation_id=resume) or project
    settings = settings_for(project)
    agent = create_stock_agent()
    shell = create_shell(
        agent,
        deps=None,
        plugins=(),
        usage_limits=UsageLimits(request_limit=settings.request_limit),
        console=console,
        settings=settings,
        store=store,
        builtin_plugins=STOCK_PLUGINS,
        project=project,
        headless=True,
    )
    SessionWorkspace(
        session=shell.session,
        context=shell.context,
        loader=shell.loader,
        status=shell.status,
        settings=settings_for,
        keep=WEB,
        busy=shell.background_running,
    ).attach(sessions=shell.sessions, commands=shell.commands)
    reason: SessionEndReason = 'error'
    async with agent, create_task_group() as workers:
        workers.start_soon(shell.sessions.namer.run)
        try:
            notice = ''
            if resume is not None or session_id is not None:
                notice = await shell.sessions.start(
                    resume=resume, resume_from=resume_from, session_id=session_id, fork=fork_session
                )
            await shell.loader.load_all()
            if not any(entry.name == WEB and entry.loaded is not None for entry in shell.loader.entries()):
                await shell.loader.load(WEB)
            if resume is not None:
                shell.session.record_resumed()
            if notice:
                console.print(notice, markup=False)
            console.print(await shell.commands.execute_async(f'/{WEB}'), markup=False)
            console.print('Prompts entered in the browser run here. Press Ctrl-C to quit.', markup=False)
            reason = await shell.run()
        except get_cancelled_exc_class():
            reason = 'exit'
            raise
        finally:
            workers.cancel_scope.cancel()
            with CancelScope(shield=True):
                await shell.loader.close(reason)


async def _start_in(*, store: SettingsStore, conversation_id: str) -> ProjectSettings | None:
    """Change into the directory a resumed session was saved in and read its project file; `None` when it is here."""
    conversations = SqliteConversationStore(database=store.path.with_name('sessions.db'))
    try:
        workspace = (await conversations.get(conversation_id=conversation_id)).summary.workspace
    except LookupError:
        return None  # Restoring it reports the unknown ID.
    if workspace == str(Path.cwd().resolve()):
        return None
    try:
        directory = workspace_directory(workspace)
    except ValueError as exc:
        raise ValueError(
            f'{exc} Session {conversation_id} was saved there; start without --resume to read it in the sidebar.'
        ) from exc
    project = load_project_settings(directory)
    os.chdir(directory)
    return project
