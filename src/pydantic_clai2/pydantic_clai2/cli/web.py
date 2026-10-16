"""`clai2 --web`: CLAI with no terminal prompt, driven only from the web UI, logging the conversation as it goes."""

import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from anyio import CancelScope, create_task_group, get_cancelled_exc_class
from rich.console import Console

from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2._app import STOCK_PLUGINS, create_shell, create_stock_agent
from pydantic_clai2.builtin_plugins.web import WebPlugin
from pydantic_clai2.cli.self_update import WebHandoff
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings, load_project_settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import DepsT, SessionEndReason
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.runtime.imported_sessions import ImportSource
from pydantic_clai2.runtime.session_workspace import SessionWorkspace, workspace_directory

WEB = 'web'
LAUNCHER_HOST = '127.0.0.1'
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
    port: int | None = None,
    token: str | None = None,
    restart: WebHandoff | None = None,
    desktop: Callable[[str], Awaitable[int]] | None = None,
) -> None:
    """Load the stock plugins and `web`, start the web UI, and run turns the browser enters until Ctrl-C.

    `console` gets the transcript as plain text. Nothing saved changes: `web` loads for this launch only.
    `resume`, `session_id`, and `fork_session` work as for `chat`, except that `resume` needs an ID.

    `project` is the launch directory's project file and `settings_for` the settings CLAI starts with
    beside a project file. Each session runs in its own directory: `resume` starts there, `/resume ID`
    moves there, and `/workspace DIRECTORY` starts a new session in another one (see `SessionWorkspace`).

    `port` replaces the saved port for this launch. `token` replaces the token `web` makes, for a launcher
    that keeps it, such as the `workspaces` plugin: CLAI then listens on 127.0.0.1 whatever the saved
    address, prints the address without the token as `Web UI running at http://127.0.0.1:PORT/`, which
    is how a launcher that passed port 0 learns the port, and opens no browser.

    `restart` is the web UI the build before `/update` served: CLAI serves it again on that port with that
    token, opening no browser, so the pages it had reconnect. After `/update`, this raises `Relaunch`.

    `desktop` shows the web UI in a window instead of the browser, as `clai2 --desktop` does with
    `open_window`: CLAI calls it with the address once serving, and quits when it returns.
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
    served: WebHandoff | None = None
    async with agent, create_task_group() as workers:
        workers.start_soon(shell.sessions.namer.run)
        try:
            notice = ''
            if resume is not None or session_id is not None:
                notice = await shell.sessions.start(
                    resume=resume, resume_from=resume_from, session_id=session_id, fork=fork_session
                )
            await shell.loader.load_all()
            if _loaded_web(shell.loader) is None:
                await shell.loader.load(WEB)
            web = _loaded_web(shell.loader)
            assert web is not None, 'a failed load raises'
            if resume is not None:
                shell.session.record_resumed()
            if notice:
                console.print(notice, markup=False)
            web.port = port
            if restart is not None:
                web.port, web.token = restart.port, restart.token
            if token is not None:
                # A launcher talks to it on this machine alone, whatever address the saved settings name.
                web.token, web.listen = token, LAUNCHER_HOST
            if desktop is not None:
                url = await web.start()
                console.print(f'Web UI running at {url}, in its own window', markup=False)
                workers.start_soon(_until_closed, desktop, url, console, workers.cancel_scope)
            elif token is None and restart is None:
                console.print(await shell.commands.execute_async(f'/{WEB}'), markup=False)
            else:
                url = await web.start()
                # A launcher keeps its token, so the address it reads leaves the token out.
                console.print(f'Web UI running at {url if token is None else url.partition("?")[0]}', markup=False)
            console.print('Prompts entered in the browser run here. Press Ctrl-C to quit.', markup=False)
            reason = await shell.run()
        except get_cancelled_exc_class():
            reason = 'exit'
            raise
        finally:
            workers.cancel_scope.cancel()
            # Before closing the plugins stops the web UI.
            served = serving(shell.loader)
            with CancelScope(shield=True):
                await shell.loader.close(reason)
    if (relaunch := shell.relaunch(web=served)) is not None:
        raise relaunch


async def _until_closed(
    window: Callable[[str], Awaitable[int]], url: str, console: Console, scope: CancelScope
) -> None:
    """Show `url` in the desktop window, then quit CLAI once the user closes it."""
    if status := await window(url):
        console.print(f'The CLAI window exited with status {status}.', markup=False)
    scope.cancel()


def _loaded_web(loader: PluginLoader[DepsT]) -> WebPlugin | None:
    loaded = next((entry.loaded for entry in loader.entries() if entry.name == WEB), None)
    return loaded.plugin if loaded is not None and isinstance(loaded.plugin, WebPlugin) else None


def serving(loader: PluginLoader[DepsT]) -> WebHandoff | None:
    """The port and token of the web UI `loader`'s `web` plugin serves now; `None` while it serves none."""
    web = _loaded_web(loader)
    if web is None or web.server is None:
        return None
    return WebHandoff(port=web.server.port, token=web.token)


async def serve_again(loader: PluginLoader[DepsT], handoff: WebHandoff) -> str:
    """Serve the web UI on the port and with the token CLAI used before `/update`, opening no browser."""
    web = _loaded_web(loader)
    if web is None:
        return 'The web UI did not start again: the web plugin is off. Turn it on with /plugins enable web.'
    web.port, web.token = handoff.port, handoff.token
    try:
        return f'Web UI running at {await web.start()}'
    except OSError as exc:
        return f'The web UI did not start again on port {handoff.port}: {exc}. Start it on another with /web.'


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
