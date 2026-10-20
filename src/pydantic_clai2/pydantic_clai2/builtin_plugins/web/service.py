"""The web UI of one CLAI process: its live sessions, each a `SessionChannel`, and everything the page shares.

The browser names a session by the conversation it holds. Sessions open and close while CLAI runs,
and each one's channel is made when it is first seen and dropped once it is gone.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from anyio import Lock, fail_after, get_cancelled_exc_class, to_thread
from pydantic import JsonValue
from termflow.themes import PALETTES

from pydantic_ai import AgentRunResult, AgentStreamEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai_harness.ask_user import Answerer
from pydantic_ai_harness.subagents import DelegationTasks
from pydantic_clai2.builtin_plugins.web.channel import SessionChannel
from pydantic_clai2.builtin_plugins.web.dialogs import BrowserDialogs
from pydantic_clai2.builtin_plugins.web.framing import FrameCheck
from pydantic_clai2.builtin_plugins.web.host_session import HostSessions
from pydantic_clai2.builtin_plugins.web.live import BrowserAnswerer, Client, Hub
from pydantic_clai2.builtin_plugins.web.menu_forms import menu_info, plugin_form, run_action, set_row
from pydantic_clai2.builtin_plugins.web.orchestrator import Orchestrator
from pydantic_clai2.builtin_plugins.web.protocol import (
    LiveChanged,
    LiveEntry,
    LiveState,
    MenuEdited,
    MenuInfo,
    Notice,
    Palette,
    PanelInfo,
    PluginRow,
    PluginsChanged,
    Project,
    QuestionAsked,
    QuestionClosed,
    SessionEntry,
    SessionsChanged,
    SettingsChanged,
    TaskTranscript,
    ThemeChanged,
    ThemeInfo,
    Transcript,
    UpdateChanged,
    UpdateInfo,
    dump,
    history_items,
)
from pydantic_clai2.builtin_plugins.web.sidebar import busy_elsewhere, session_entry, worktrees
from pydantic_clai2.builtin_plugins.web.workers import task_entry
from pydantic_clai2.cli.self_update import installed
from pydantic_clai2.config.theme_names import names as theme_names
from pydantic_clai2.plugins import (
    CommandEnd,
    ConversationChanged,
    LiveSession,
    LiveSessions,
    PanelView,
    PluginCatalog,
    PluginHost,
    PluginInfo,
    SessionCatalog,
    SidePanel,
    TaskCatalog,
    TaskEvent,
    UpdateProgress,
)
from pydantic_clai2.plugins.loader import PluginError
from pydantic_clai2.plugins.menus import SettingsMenu
from pydantic_clai2.runtime.monty_cells import is_monty_cell
from pydantic_clai2.runtime.project_identity import MISSING, ProjectIdentity, placements, project_identity
from pydantic_clai2.runtime.session_workspace import workspace_directory

THEME = 'display.theme'
PluginAction = Literal['enable', 'disable', 'reload']
TaskAction = Literal['stop', 'background', 'resume']
SERVING = 'This page runs on the web plugin, so change it from the terminal; any change restarts the web UI.'

VIEW_TIMEOUT = 10.0
"""Seconds a side panel's view may take before the panel says it did not answer, so one cannot hold up the rest."""


class WebService:
    """What every browser of this CLAI shares: the live sessions, plugins, settings menus, sidebar, and orchestrator.

    The legacy routes and a browser that never sends `watch` use the default session: the one the
    page opened or started last while it is live, else the newest live one.
    """

    def __init__(self, *, host: PluginHost[None], terminal: Answerer | None = None) -> None:
        """`terminal` races the browser to answer the model's questions, where CLAI has a terminal."""
        self.host = host
        self.live: LiveSessions = host.live if host.live is not None else HostSessions(host)
        """The process's live sessions; a host built outside CLAI's loader is its own one."""
        self.hub = Hub(default=self._default_id)
        self.answerer = BrowserAnswerer(publish=self._question, terminal=terminal, session=self._asking)
        """Asks the model's questions in the browsers watching the session asking."""
        self.dialogs = BrowserDialogs(publish=self.hub.publish)
        """Where the settings menus opened in this page ask their questions."""
        self.orchestrator = Orchestrator(host=host, publish=self.hub.publish)
        """The orchestrator chat, a separate agent with its own conversation, beside the sessions."""
        self.frames = FrameCheck()
        """Which side panel pages this page can frame; each page's headers are read once."""
        self.version = installed().label
        self.update: UpdateInfo | None = None
        """`/update` installing a newer CLAI, until the install ends; a browser that connects meanwhile sees it."""
        self._uv_output: list[str] = []
        self._channels: dict[str, SessionChannel] = {}
        """By `LiveSession.id`."""
        self._chosen: str | None = None
        """The `LiveSession.id` the page opened or started last, which legacy routes follow while it is live."""
        self._identities: dict[str, ProjectIdentity] = {}
        self._theme = self._theme_name()
        """The theme browsers were last sent, so a command that changes it sends it again."""
        self._command_edits = Lock()
        """One page edit of a command's menu at a time; `/set`'s source also serializes its saves with the terminal's."""
        self._following = False
        """Set while `follow` counts watchers, which tells the live sessions again."""
        self._live_sent = self.live_message()
        """The `live` message browsers were last sent; each connecting one gets it as it is then."""
        self._unsubscribe = self.live.subscribe(self.live_changed)

    def close(self) -> None:
        """Stop following the live sessions and stop counting browsers as their watchers."""
        self._unsubscribe()
        for client in self.hub.connected():
            self.unwatch(client)

    # Live sessions and their channels.

    def channel(self, session: LiveSession) -> SessionChannel:
        found = self._channels.get(session.id)
        if found is None:
            found = self._channels[session.id] = SessionChannel(session=session, service=self)
        return found

    def channels(self) -> list[SessionChannel]:
        """A channel per live session, oldest first."""
        return [self.channel(session) for session in self.live.listing()]

    def session(self, conversation_id: str) -> SessionChannel:
        """The channel of the live session holding `conversation_id`; `LookupError` when none does."""
        for channel in self.channels():
            if channel.id == conversation_id:
                return channel
        raise LookupError(f'Session {conversation_id} is not open; open it first.')

    def default(self) -> SessionChannel:
        """The session the legacy routes drive: the one the page chose last while live, else the newest."""
        if (found := self._default()) is None:
            raise LookupError('No session is open.')
        return found

    def _default(self) -> SessionChannel | None:
        sessions = self.live.listing()
        if not sessions:
            return None
        return self.channel(next((session for session in sessions if session.id == self._chosen), sessions[-1]))

    def _default_id(self) -> str | None:
        return None if (found := self._default()) is None else found.id

    def current(self) -> SessionChannel:
        """The channel of the session whose moment or run is handled now, else the default one."""
        session = self.live.current()
        return self.default() if session is None else self.channel(session)

    def live_message(self) -> LiveChanged:
        return LiveChanged(
            sessions=[
                LiveEntry(
                    id=channel.id,
                    state=self._state(channel),
                    title=channel.session.conversation.title,
                    workspace=str(channel.session.workspace),
                )
                for channel in self.channels()
            ]
        )

    def _state(self, channel: SessionChannel) -> LiveState:
        session = channel.session
        if self.answerer.asked(channel.id):
            return 'asking'
        if session.queued:
            return 'queued'
        return 'running' if session.running else 'idle'

    def live_changed(self) -> None:
        """A session opened, closed, changed state or queued input, or switched conversation: tell browsers what changed.

        Called synchronously by the live sessions, so it never waits and never raises.
        """
        live = {session.id for session in self.live.listing()}
        for gone in self._channels.keys() - live:
            del self._channels[gone]
        message = self.live_message()
        if message != self._live_sent:
            self._live_sent = message
            self.hub.publish(message)
        for channel in self.channels():
            channel.publish_queue()
        for client in self.hub.connected():
            self.follow(client)

    def watch(self, client: Client, conversation_ids: Sequence[str]) -> None:
        """Replace what `client` watches; it gets a snapshot of each live session it did not watch before."""
        if client.watching is None:
            # It showed the default session until now; each session it watches gets a snapshot.
            self.unwatch(client)
        client.watching = set(conversation_ids)
        self.follow(client)

    def follow(self, client: Client, *, snapshots: bool = True) -> None:
        """Count `client` as a watcher of exactly the live sessions it shows, sending each new one's snapshot.

        It shows those it watches, or the default session until it sends `watch`. One it watches that is
        not live yet is followed once it opens. A session switching conversation keeps its watchers; it
        sends them its new snapshot itself.
        """
        if self._following:
            return  # Counting a watcher tells the live sessions, which call `live_changed` again.
        self._following = True
        try:
            if client.watching is None:
                wanted = {} if (default := self._default()) is None else {default.session.id: default}
            else:
                # By the conversation browsers were last shown: one a session switched to but this plugin
                # has not heard of yet is followed after `conversation_changed`, which sends its snapshot.
                held = {channel.shown: channel for channel in self.channels()}
                wanted = {held[name].session.id: held[name] for name in client.watching if name in held}
            for live_id in client.unwatch.keys() - wanted.keys():
                client.unwatch.pop(live_id)()
            for live_id, channel in wanted.items():
                if live_id not in client.unwatch:
                    client.unwatch[live_id] = channel.session.watch()
                    if snapshots:
                        self.hub.send([client], dump(channel.snapshot()))
        finally:
            self._following = False

    def unwatch(self, client: Client) -> None:
        """`client` left: it no longer keeps any session from closing for being idle."""
        for unwatch in client.unwatch.values():
            unwatch()
        client.unwatch.clear()

    def _question(self, message: QuestionAsked | QuestionClosed, session: str | None) -> None:
        assert session is not None, 'every question here is asked by a live session, from `_asking`'
        self.hub.publish_session(message, session=session)
        self.live_changed()

    def _asking(self) -> str:
        session = self.live.current()
        return (session or self.default().session).conversation.conversation_id

    async def open(self, conversation_id: str) -> str:
        """Make a saved session live, resumed in a session of its own unless one holds it; its conversation ID.

        A host that cannot open other sessions (its live sessions have no `build`) resumes it in the default
        session instead: the ID is then the default session's, whose watchers get the resumed one next.
        """
        try:
            session = await self.live.open(conversation_id)
        except RuntimeError:
            default = self.default()
            default.session.send_input(f'/resume {conversation_id}')
            return default.id
        return self._choose(session)

    async def new(self, workspace: str | None = None) -> str:
        """Start a new live session in `workspace`, by default the default session's directory; its conversation ID.

        A CLAI that opens no other sessions starts it in the default session, as `open` does.
        """
        if workspace is not None:
            directory = workspace_directory(workspace)
        else:
            directory = None if (shown := self._default()) is None else shown.session.workspace
        try:
            session = await self.live.new(workspace=directory)
        except RuntimeError:
            # With no session live there is no default to start it in: `LookupError`.
            default = self.default()
            if directory != default.session.workspace:
                if not self.catalog.follows_workspace:
                    raise ValueError(f'This CLAI works in {default.session.workspace}. Start CLAI in {directory}.')
                default.session.send_input(f'/workspace {directory}')
            else:
                default.session.send_input('/new')
            return default.id
        return self._choose(session)

    def _choose(self, session: LiveSession) -> str:
        self._chosen = session.id
        self.live_changed()
        self.hub.publish(SessionsChanged())
        return self.channel(session).id

    # Theme and settings, which every session shares.

    def theme(self) -> ThemeInfo:
        """The `/theme` choice with its palette, as the terminal paints it."""
        name = self._theme_name()
        found = PALETTES.get(name)
        palette = None if found is None else Palette(bg=found.bg, fg=found.fg, ansi=list(found.ansi))
        return ThemeInfo(name=name, names=list(theme_names()), palette=palette)

    def _theme_name(self) -> str:
        try:
            return str(self.host.setting(THEME))
        except KeyError:
            return 'default'

    def settings_changed(self, channel: SessionChannel | None = None) -> None:
        """Tell browsers CLAI's settings may have changed: the theme if it did, the status rows, and open menus."""
        if (name := self._theme_name()) != self._theme:
            self._theme = name
            self.hub.publish(ThemeChanged(theme=self.theme()))
        for each in self.channels() if channel is None else [channel]:
            each.publish_status()
        self.hub.publish(SettingsChanged())

    # The plugin's moments, routed to the session they happened in.

    def turn_ended(self) -> None:
        """A turn, or a background `/fork`, ended and saved its outcome and usage: the figures and sidebar show it."""
        self.current().publish_status()
        self.hub.publish(SessionsChanged())

    def conversation_changed(self, event: ConversationChanged) -> None:
        # By the conversation, not `current()`: a session that is still opening is not listed yet.
        for channel in self.channels():
            if channel.id == event.conversation_id:
                channel.conversation_changed(title=event.title)
        self.live_changed()
        self.hub.publish(SessionsChanged())

    def command_ended(self, event: CommandEnd) -> None:
        channel = self.current()
        channel.command_ended(event)
        self.settings_changed(channel)
        # `/plugins`, or a plugin's own command such as `/grain`, may have changed a plugin or its settings.
        self.hub.publish(PluginsChanged())

    def task_event(self, event: TaskEvent) -> None:
        self.current().workers.observe(event)

    def update_progress(self, event: UpdateProgress) -> None:
        """Show `/update` going: uv's last line while it installs, then all it printed, as the terminal shows it."""
        if event.stage == 'installing':
            self._uv_output = []
            self.update = UpdateInfo(target=event.target)
        elif event.stage == 'output':
            self._uv_output.append(event.text)
            self.update = UpdateInfo(target=event.target, line=event.text)
        else:
            if self._uv_output:
                level = 'error' if event.stage == 'failed' else 'info'
                self.current().publish(Notice(text='\n'.join(self._uv_output), level=level, command='uv'))
            self.update = UpdateInfo(target=event.target, restarting=True) if event.stage == 'restarting' else None
        self.hub.publish(UpdateChanged(update=self.update))

    async def models(self) -> list[str]:
        """The saved models and chains `/model` offers, as `PROVIDER:NAME`."""
        return [completion.text for completion in await self.host.complete('/model ') if ':' in completion.text]

    # The plugins panel.

    @property
    def plugin_catalog(self) -> PluginCatalog:
        if self.host.plugins is None:
            raise LookupError('This CLAI has no plugins to manage.')
        return self.host.plugins

    def plugins(self) -> list[PluginRow]:
        """Every plugin, as `/plugins` lists it, with a form for the settings of each one that is on."""
        return [self._plugin_row(info) for info in self.plugin_catalog.listing()]

    def _plugin(self, name: str) -> PluginInfo:
        found = next((info for info in self.plugin_catalog.listing() if info.name == name), None)
        if found is None:
            raise LookupError(f'Unknown plugin: {name}')
        return found

    def _plugin_row(self, info: PluginInfo) -> PluginRow:
        return PluginRow(
            name=info.name,
            description=info.description,
            origin=info.origin,
            source=info.source,
            status=info.status,
            included_in=info.included_in,
            error=info.error,
            notice=info.notice,
            provides=info.provides,
            has_menu=info.has_menu,
            settings_menu=info.settings_menu,
            terminal=self.host.console.is_terminal,
            form=None if info.settings_menu else plugin_form(info),
            serving=info.name == self.host.name,
        )

    async def plugin_action(self, name: str, action: PluginAction) -> None:
        """Turn a plugin on or off, or reload it, as `/plugins` does; a plugin's own menu stays closed."""
        if name == self.host.name and action != 'enable':
            raise ValueError(SERVING)
        catalog = self.plugin_catalog
        run = {'enable': catalog.enable, 'disable': catalog.disable, 'reload': catalog.reload}[action]
        try:
            await run(name)
        except PluginError as error:
            raise ValueError(str(error)) from error
        finally:
            self.hub.publish(PluginsChanged())

    async def configure_plugin(self, name: str, changes: Mapping[str, JsonValue], reset: Sequence[str] = ()) -> None:
        """Change some settings of a plugin that is on and reset others to their defaults, then load it again.

        The rest keep their saved values. Raises `ValidationError` naming each setting its model rejects.
        """
        if name == self.host.name:
            # Loading it again stops this server and starts one with a new token, which this page lacks.
            raise ValueError(SERVING)
        info = self._plugin(name)
        if info.settings_model is None or info.settings is None:
            if info.status != 'on':
                raise ValueError(f'Turn {name} on to change its settings.')
            raise ValueError(f'{name} has no settings form; replace its declaration with /plugins add.')
        values = info.settings.model_dump(by_alias=True)
        for key in reset:
            values.pop(key, None)
        settings = info.settings_model.model_validate({**values, **changes})
        try:
            await self.plugin_catalog.save_settings(name, settings)
        except PluginError as error:
            raise ValueError(str(error)) from error
        except TypeError as error:
            # Its model changed since the listing above, as a reload that re-imports it does.
            raise ValueError(f'{name} changed; reload the panel and try again.') from error
        finally:
            self.hub.publish(PluginsChanged())

    def open_plugin_menu(self, name: str) -> None:
        """Open a plugin's own settings menu in the terminal, as `/plugins configure NAME` typed there does."""
        self.default().session.send_input(f'/plugins configure {self._plugin(name).name}')

    async def plugin_menu(self, name: str, *, path: Sequence[str] = ()) -> MenuInfo:
        """A plugin's declared settings menu, or the one `path`'s rows open from it, with every row's current value."""
        menu = self.plugin_catalog.settings_menu(self._plugin(name).name)
        if menu is None:
            raise LookupError(f'{name} declares no settings menu here; open its own in the terminal.')
        # Its source may read the keyring or files, so it runs off the loop, as the terminal's menu does.
        return await to_thread.run_sync(menu_info, menu, path)

    async def edit_menu(self, name: str, key: str, value: str | None, *, path: Sequence[str] = ()) -> MenuEdited:
        """Set a row of a plugin's declared menu to `value`, or reset it with `None` or empty text, as its menu does.

        Raises `ValueError` with the row's own message when the menu rejects the value; nothing is saved then.
        """
        return await self._edit(name, partial(set_row, path=path, key=key, value=value), path)

    async def run_menu_action(self, name: str, key: str, *, path: Sequence[str] = ()) -> MenuEdited:
        """Run what a row of a plugin's declared menu does when chosen, asking through this page's dialogs."""
        return await self._edit(name, partial(run_action, path=path, key=key, dialogs=self.dialogs, owner=name), path)

    async def _edit(
        self, name: str, edit: Callable[[SettingsMenu], Awaitable[list[str]]], path: Sequence[str]
    ) -> MenuEdited:
        """One edit, after which the plugin loads again if it saved new settings; shown here and in the terminal."""
        if name == self.host.name:
            raise ValueError(SERVING)
        catalog = self.plugin_catalog
        try:
            messages = await catalog.edit_settings(name, edit)
        except PluginError as error:
            raise ValueError(str(error)) from error
        finally:
            self.hub.publish(PluginsChanged())
        menu = catalog.settings_menu(name)
        return MenuEdited(
            messages=messages, menu=None if menu is None else await to_thread.run_sync(menu_info, menu, path)
        )

    def answer_dialog(self, dialog_id: str, value: str | None) -> None:
        """Answer a settings menu's open dialog, or cancel it with `None`."""
        self.dialogs.answer(dialog_id, value)

    # Commands' own menus, such as `/set`'s, which the page shows instead of the terminal. Each is built
    # from one live session's commands, so `/model` or `/effort` read and change that session: `session`
    # is its channel, the default session's for the unprefixed routes.

    def _command_menu(self, session: SessionChannel, name: str) -> SettingsMenu:
        """Build the menu bare `/NAME` declares; building may read files, as `/model`'s does, so call it off the loop."""
        menu = session.session.command_menu(name)
        if menu is None:
            raise LookupError(f'/{name} has no menu to show here.')
        return menu

    def _command_info(self, session: SessionChannel, name: str, path: Sequence[str]) -> MenuInfo:
        return menu_info(self._command_menu(session, name), path)

    async def command_menu(self, session: SessionChannel, name: str, *, path: Sequence[str] = ()) -> MenuInfo:
        """The menu bare `/NAME` opens, or the one `path`'s rows open from it, with every row's current value."""
        return await to_thread.run_sync(self._command_info, session, name, path)

    async def edit_command_menu(
        self, session: SessionChannel, name: str, key: str, value: str | None, *, path: Sequence[str] = ()
    ) -> MenuEdited:
        """Set a row of a command's menu to `value`, or reset it with `None` or empty text, as the terminal's menu does."""
        return await self._edit_command(session, name, partial(set_row, path=path, key=key, value=value), path)

    async def run_command_menu_action(
        self, session: SessionChannel, name: str, key: str, *, path: Sequence[str] = ()
    ) -> MenuEdited:
        """Run what a row of a command's menu does when chosen, asking through this page's dialogs."""
        action = partial(run_action, path=path, key=key, dialogs=self.dialogs, owner=f'/{name}')
        return await self._edit_command(session, name, action, path)

    async def _edit_command(
        self,
        session: SessionChannel,
        name: str,
        edit: Callable[[SettingsMenu], Awaitable[list[str]]],
        path: Sequence[str],
    ) -> MenuEdited:
        """One edit, printed in the terminal too, after which every page reads the settings again."""
        async with self._command_edits:
            menu = await to_thread.run_sync(self._command_menu, session, name)
            try:
                messages = await edit(menu)
            finally:
                # Every session's status: a `display.*` setting changes them all.
                self.settings_changed()
        for message in messages:
            self.host.console.print(message, markup=False, highlight=False)
        # Built again: the edit may have changed what it lists, as adding or removing a model does.
        return MenuEdited(messages=messages, menu=await to_thread.run_sync(self._command_info, session, name, path))

    # The Workers tree: the delegated tasks of every live session, by task ID.

    def _task(self, task_id: str) -> tuple[SessionChannel, TaskCatalog]:
        owned = [(channel, channel.session.tasks) for channel in self.channels()]
        catalogs = [(channel, catalog) for channel, catalog in owned if catalog is not None]
        if not catalogs:
            raise LookupError('This CLAI runs no delegated tasks.')
        for channel, catalog in catalogs:
            if any(task.id == task_id for task in catalog.listing()):
                return channel, catalog
        raise LookupError(f'No task {task_id!r} in an open session.')

    def task_transcript(self, task_id: str) -> TaskTranscript:
        _, catalog = self._task(task_id)
        [task] = [task for task in catalog.listing() if task.id == task_id]
        return TaskTranscript(task=task_entry(task), items=history_items(catalog.history(task_id)))

    async def task_action(self, task_id: str, action: TaskAction) -> None:
        """Stop, background, or resume a task, as `/tasks` does."""
        channel, catalog = self._task(task_id)
        if action == 'stop':
            await catalog.stop(task_id)
        elif action == 'background':
            catalog.background(task_id)
        else:
            await catalog.resume(task_id)
        # Backgrounding changes the row without an event from the task, which may be waiting on its model.
        for task in catalog.listing():
            if task.id == task_id:
                channel.workers.observe(TaskEvent(task=task))

    def steer(self, task_id: str, text: str) -> bool:
        """Send a running task a message, shown in its transcript; `False` once it has finished."""
        channel, catalog = self._task(task_id)
        text = text.strip()
        steered = catalog.steer(task_id, text)
        if steered:
            channel.workers.steered(task_id, text)
        return steered

    # Side panels.

    async def panels(self, *, origin: str) -> list[PanelInfo]:
        """Every loaded plugin's side panels as they are now, for a page served from `origin`."""
        found: list[PanelInfo] = []
        declared = {} if self.host.plugins is None else self.host.plugins.side_panels()
        for plugin, panels in declared.items():
            for panel in panels:
                view = await _view(panel)
                found.append(
                    PanelInfo(
                        plugin=plugin,
                        id=panel.id,
                        title=panel.title,
                        icon=panel.icon,
                        url=view.url,
                        note=view.note,
                        embed=view.url is not None and await self.frames(view.url, ancestor=origin),
                    )
                )
        return found

    # The sidebar's sessions.

    @property
    def catalog(self) -> SessionCatalog:
        """The saved sessions, one store for every live session."""
        if self.host.sessions is None:
            raise LookupError('This CLAI does not save sessions.')
        return self.host.sessions

    async def projects(self, query: str = '', *, include_archived: bool = False) -> list[Project]:
        """Saved sessions grouped by repository, then checkout, the current ones first and removed folders last."""
        entries = await self.catalog.listing(query=query, archived=None if include_archived else False)
        workspace = str(self.default().session.workspace)
        workspaces = {entry.workspace for entry in entries} | {workspace}
        known = dict(self._identities)

        def resolve() -> dict[str, ProjectIdentity]:
            # A folder removed or recreated since it was last listed resolves again: one stat each, Git only for those.
            return {
                workspace: project_identity(workspace)
                for workspace in workspaces
                if (identity := known.get(workspace)) is None or Path(workspace).exists() == identity.missing
            }

        self._identities.update(await to_thread.run_sync(resolve))
        live = {channel.id: channel.session for channel in self.channels()}
        # Checking whether another process runs a session can spawn a program (`tasklist` on Windows).
        running = [entry for entry in entries if entry.outcome == 'running' and entry.id not in live]
        busy: set[str] = (
            await to_thread.run_sync(lambda: {entry.id for entry in running if busy_elsewhere(entry)})
            if running
            else set()
        )
        # Every workspace seen so far, not only this search's, tells where a removed folder belonged.
        places = placements(entries, identities=self._identities)
        here = places[workspace]
        default = self.default().id
        checkouts: dict[str, dict[str, tuple[ProjectIdentity, list[SessionEntry]]]] = {}
        for entry in entries:
            identity = places[entry.workspace]
            _, sessions = checkouts.setdefault(identity.key, {}).setdefault(identity.root, (identity, []))
            sessions.append(
                session_entry(entry, live=live.get(entry.id), busy=entry.id in busy, current=entry.id == default)
            )
        projects = [
            Project(
                key=key,
                name=next(iter(found.values()))[0].name,
                current=key == here.key,
                worktrees=worktrees(found, here=here),
            )
            for key, found in checkouts.items()
        ]
        return sorted(projects, key=lambda project: (not project.current, project.key == MISSING))

    async def transcript(self, conversation_id: str) -> Transcript:
        saved = await self.catalog.saved(conversation_id)
        openable = self._continuable(saved.summary.workspace)
        return Transcript(
            id=saved.summary.id,
            title=saved.summary.title,
            workspace=saved.summary.workspace,
            items=history_items(saved.messages),
            openable=openable,
            missing=self.catalog.follows_workspace and not openable,
        )

    def _continuable(self, workspace: str) -> bool:
        """Whether a session saved in `workspace` can be opened: here, or there while CLAI follows sessions."""
        if workspace == str(self.default().session.workspace):
            return True
        if not self.catalog.follows_workspace:
            return False
        try:
            workspace_directory(workspace)
        except ValueError:
            return False
        return True

    async def rename(self, conversation_id: str, title: str) -> None:
        await self.catalog.rename(conversation_id, title)
        self.hub.publish(SessionsChanged())

    async def fork(self, conversation_id: str) -> str:
        """Copy a session and open the copy."""
        copy = await self.catalog.fork(conversation_id)
        self.hub.publish(SessionsChanged())
        if self._continuable(copy.workspace):
            await self.open(copy.id)
        return copy.id

    async def delete(self, conversation_id: str) -> None:
        await self.catalog.delete(conversation_id)
        self.hub.publish(SessionsChanged())

    async def archive(self, conversation_id: str, *, archived: bool) -> None:
        """Archive a session, hiding it from the sidebar, or unarchive it."""
        catalog = self.catalog
        await (catalog.archive(conversation_id) if archived else catalog.unarchive(conversation_id))
        self.hub.publish(SessionsChanged())


async def _view(panel: SidePanel) -> PanelView:
    """What `panel` shows now; a failing view, or a page that is not on the web, shows why instead."""
    try:
        with fail_after(VIEW_TIMEOUT):
            view = await panel.view()
    except TimeoutError:
        return PanelView(note=f'{panel.title} did not answer in time.')
    # Any plugin's bug: one broken panel must not take the others, or the page, down.
    except Exception as error:  # noqa: BLE001
        return PanelView(note=f'{panel.title} failed: {str(error) or type(error).__name__}')
    if view.url is not None and not _web_page(view.url):
        return PanelView(note=f'{panel.title} offered a page that is not on the web; only http and https open here.')
    return view


def _web_page(url: str) -> bool:
    """Whether `url` is a well-formed `http` or `https` address, which the browser can open."""
    try:
        parts = urlsplit(url)
    except ValueError:  # Such as an unclosed `[` in an IPv6 host.
        return False
    return parts.scheme in ('http', 'https') and bool(parts.hostname) and not any(char.isspace() for char in url)


@dataclass
class Observer(AbstractCapability[None]):
    """Hands each live session's own runs and their events to its channel.

    Turns follow these runs rather than the turn moments, which a background `/fork` fires too.
    """

    service: WebService

    def _channel(self, ctx: RunContext[None]) -> SessionChannel | None:
        """The channel of the session whose held conversation this run continues; `None` for a fork or a task."""
        if DelegationTasks.child_id() is not None:
            return None
        try:
            channel = self.service.current()
        except LookupError:
            return None
        return channel if ctx.conversation_id == channel.id else None

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        # A `/monty` cell is not a turn: its result comes as the command's, and its events as any run's.
        channel = self._channel(ctx)
        if channel is None or is_monty_cell(ctx):
            return await handler()
        channel.run_started(ctx.prompt)
        try:
            result = await handler()
        except get_cancelled_exc_class():
            channel.run_ended('cancelled')
            raise
        except Exception as error:
            channel.run_ended('failed', error=error)
            raise
        channel.run_ended('completed', usage=result.usage)
        return result

    async def on_event(self, ctx: RunContext[None], *, event: AgentStreamEvent) -> None:
        if (channel := self._channel(ctx)) is not None:
            channel.observe(event, usage=ctx.usage, by_user=is_monty_cell(ctx))
