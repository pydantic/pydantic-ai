"""`clai2 --web`: CLAI with no terminal prompt, driven only from the web UI, logging the conversation as it goes."""

import os
from collections.abc import Callable
from pathlib import Path

from anyio import CancelScope, create_task_group, get_cancelled_exc_class
from rich.console import Console

from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.step_persistence.conversations import SqliteSessionStore
from pydantic_clai2.app import STOCK_PLUGINS, create_stock_agent
from pydantic_clai2.builtin_plugins.web import WebPlugin
from pydantic_clai2.cli.self_update import WebHandoff
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings, load_project_settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import DepsT, SessionEndReason
from pydantic_clai2.plugins.loader import SessionPlugins
from pydantic_clai2.runtime.imported_sessions import ImportSource
from pydantic_clai2.runtime.process import ClaiProcess
from pydantic_clai2.runtime.session_runtime import SessionRuntime, create_shell
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
    """
    if resume == '':
        raise ValueError('--web needs an explicit --resume SESSION-ID; the sidebar lists the others')
    launched_in, launch_project = Path.cwd().resolve(), project
    workspace = None
    if resume and resume_from is None and (saved := await _start_in(store=store, conversation_id=resume)):
        workspace, project = saved
    shell = _web_session(
        settings_for=settings_for,
        project=project,
        console=console,
        workspace=workspace,
        store=store,
    )
    with open(os.devnull, 'w', encoding='utf-8') as sink:
        # Sessions opened from the browser log nothing: their output would interleave with this one's.
        shell.process.live.build = lambda directory: _web_session(
            settings_for=settings_for,
            project=launch_project if directory in (None, launched_in) else load_project_settings(directory),
            console=Console(file=sink, force_terminal=False),
            workspace=directory or launched_in,
            process=shell.process,
        )
        await _serve_web(
            shell,
            console=console,
            resume=resume,
            resume_from=resume_from,
            session_id=session_id,
            fork_session=fork_session,
            port=port,
            token=token,
            restart=restart,
        )


def _web_session(
    *,
    settings_for: Callable[[ProjectSettings], Settings],
    project: ProjectSettings,
    console: Console,
    workspace: Path | None,
    store: SettingsStore | None = None,
    process: ClaiProcess[None] | None = None,
) -> SessionRuntime[None, str]:
    """A session of `clai2 --web` in `workspace`: over a new process with `store`, or joining `process`."""
    settings = settings_for(project)
    shell = create_shell(
        create_stock_agent(),
        deps=None,
        plugins=(),
        usage_limits=UsageLimits(request_limit=settings.request_limit),
        console=console,
        settings=settings,
        store=store,
        builtin_plugins=STOCK_PLUGINS if process is None else (),
        project=project,
        headless=True,
        workspace=workspace,
        process=process,
        # Ctrl-C reaches the session CLAI started with, which ends CLAI; the others are interrupted from the browser.
        signals=process is None,
    )
    SessionWorkspace(
        session=shell.session,
        context=shell.context,
        loader=shell.loader,
        status=shell.status,
        settings=settings_for,
        busy=shell.background_running,
    ).attach(sessions=shell.sessions, commands=shell.commands)
    return shell


async def _serve_web(
    shell: SessionRuntime[None, str],
    *,
    console: Console,
    resume: str | None,
    resume_from: ImportSource | None,
    session_id: str | None,
    fork_session: bool,
    port: int | None,
    token: str | None,
    restart: WebHandoff | None,
) -> None:
    """Run `shell`, the session CLAI started with, serving the web UI until it ends; other live sessions end first."""
    reason: SessionEndReason = 'error'
    served: WebHandoff | None = None
    async with shell.agent, create_task_group() as workers:
        workers.start_soon(shell.process.namer.run)
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
            # Before the page is reachable, so a browser opening a session at once finds CLAI serving them.
            async with shell.process.live.serving():
                web.port = port
                if restart is not None:
                    web.port, web.token = restart.port, restart.token
                if token is not None:
                    # A launcher talks to it on this machine alone, whatever address the saved settings name.
                    web.token, web.listen = token, LAUNCHER_HOST
                if token is None and restart is None:
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
                await shell.close(reason)
    if (relaunch := shell.relaunch(web=served)) is not None:
        raise relaunch


def _loaded_web(loader: SessionPlugins[DepsT]) -> WebPlugin | None:
    loaded = next((entry.loaded for entry in loader.entries() if entry.name == WEB), None)
    return loaded.plugin if loaded is not None and isinstance(loaded.plugin, WebPlugin) else None


def serving(loader: SessionPlugins[DepsT]) -> WebHandoff | None:
    """The port and token of the web UI `loader`'s `web` plugin serves now; `None` while it serves none."""
    web = _loaded_web(loader)
    if web is None or web.server is None:
        return None
    return WebHandoff(port=web.server.port, token=web.token)


async def serve_again(loader: SessionPlugins[DepsT], handoff: WebHandoff) -> str:
    """Serve the web UI on the port and with the token CLAI used before `/update`, opening no browser."""
    web = _loaded_web(loader)
    if web is None:
        return 'The web UI did not start again: the web plugin is off. Turn it on with /plugins enable web.'
    web.port, web.token = handoff.port, handoff.token
    try:
        return f'Web UI running at {await web.start()}'
    except OSError as exc:
        return f'The web UI did not start again on port {handoff.port}: {exc}. Start it on another with /web.'


async def _start_in(*, store: SettingsStore, conversation_id: str) -> tuple[Path, ProjectSettings] | None:
    """The directory a resumed session was saved in, with its project file; `None` when it is here."""
    conversations = SqliteSessionStore(database=store.path.with_name('sessions.db'))
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
    return directory, load_project_settings(directory)
