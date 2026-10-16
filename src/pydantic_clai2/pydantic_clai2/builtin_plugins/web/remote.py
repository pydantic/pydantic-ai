"""The session as the browser drives it: what it sees on connect, what each action does, and the sidebar's sessions."""

import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path, PurePath
from typing import Literal
from urllib.parse import urlsplit

from anyio import Lock, fail_after, get_cancelled_exc_class, to_thread
from pydantic import JsonValue
from pydantic.errors import PydanticInvalidForJsonSchema
from termflow.themes import PALETTES

from pydantic_ai import AgentRunResult, AgentStreamEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.messages import UserContent, UserPromptPart
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.ask_user import AskUserAnswer
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationConflict,
    ConversationSummary,
    ensure_inactive,
)
from pydantic_ai_harness.subagents import DelegationTasks
from pydantic_clai2.builtin_plugins.web.dialogs import BrowserDialogs
from pydantic_clai2.builtin_plugins.web.framing import FrameCheck
from pydantic_clai2.builtin_plugins.web.live import BrowserAnswerer, Hub
from pydantic_clai2.builtin_plugins.web.orchestrator import Orchestrator
from pydantic_clai2.builtin_plugins.web.protocol import (
    Choice,
    CompletionInfo,
    EventTranslator,
    Item,
    ItemAdded,
    ItemLog,
    MenuEdited,
    MenuField,
    MenuInfo,
    Notice,
    Palette,
    PanelInfo,
    PluginForm,
    PluginRow,
    PluginsChanged,
    Project,
    SessionEntry,
    SessionInfo,
    SessionsChanged,
    SessionState,
    SettingsChanged,
    Snapshot,
    StatusChanged,
    StatusInfo,
    TaskEntry,
    TaskTranscript,
    TextDelta,
    TextDone,
    ThemeChanged,
    ThemeInfo,
    TitleChanged,
    Transcript,
    TurnEnded,
    TurnStarted,
    UpdateChanged,
    UpdateInfo,
    Worktree,
    history_items,
    usage_info,
    user_text,
)
from pydantic_clai2.builtin_plugins.web.workers import Workers, task_entry
from pydantic_clai2.cli.self_update import installed
from pydantic_clai2.commands import is_command_input
from pydantic_clai2.config.theme_names import names as theme_names
from pydantic_clai2.plugins import (
    CommandEnd,
    ConversationChanged,
    PanelView,
    PluginCatalog,
    PluginHost,
    PluginInfo,
    SessionCatalog,
    SidePanel,
    TaskCatalog,
    TaskEvent,
    TurnOutcome,
    UpdateProgress,
)
from pydantic_clai2.plugins.loader import PluginError
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, SettingsMenu
from pydantic_clai2.runtime.monty_cells import is_monty_cell
from pydantic_clai2.runtime.project_identity import MISSING, ProjectIdentity, placements, project_identity
from pydantic_clai2.runtime.session_workspace import workspace_directory

THEME = 'display.theme'
CODE_MODE = 'run.code_mode'
SPECULATION = 'run.speculative_code_mode'
Switch = Literal['run.code_mode', 'run.speculative_code_mode']
"""The settings the browser's toggles change; nothing else is set from it but through the prompt."""
PluginAction = Literal['enable', 'disable', 'reload']
TaskAction = Literal['stop', 'background', 'resume']
SERVING = 'This page runs on the web plugin, so change it from the terminal; any change restarts the web UI.'

VIEW_TIMEOUT = 10.0
"""Seconds a side panel's view may take before the panel says it did not answer, so one cannot hold up the rest."""
STATUS_INTERVAL = 0.2
"""Seconds between status updates sent while a turn streams, so token counts tick; turn and command ends always send one."""


class Remote:
    """One shell session, as every connected browser shares it.

    Everything a browser does goes through the prompt, as typing there would, so the terminal and
    the browser always drive the same session. What happens comes back through the plugin's moments
    and the run's events.
    """

    def __init__(self, *, host: PluginHost[None], hub: Hub, answerer: BrowserAnswerer) -> None:
        self.host = host
        self.hub = hub
        self.answerer = answerer
        self.workers = Workers(publish=hub.publish, history=lambda task_id: self.task_catalog.history(task_id))
        """The delegated tasks, which the plugin's `on_task_event` keeps up to date."""
        self.dialogs = BrowserDialogs(publish=hub.publish)
        """Where the settings menus opened in this page ask their questions."""
        self.orchestrator = Orchestrator(host=host, publish=hub.publish)
        """The orchestrator chat, a separate agent with its own conversation, beside the session."""
        self.running = False
        self._translator = EventTranslator()
        """One for the whole load, so item IDs stay unique across turns in a browser that keeps them all."""
        self._turn = ItemLog()
        self._status_sent = 0.0
        self._usage: RunUsage | None = None
        """The running turn's usage, as its run context reports it; `None` between turns."""
        self._identities: dict[str, ProjectIdentity] = {}
        self._shown = host.conversation.conversation_id
        """The conversation browsers were last sent, so a retitle sends only the title."""
        self._theme = self._theme_name()
        """The theme browsers were last sent, so a command that changes it sends it again."""
        self.frames = FrameCheck()
        """Which side panel pages this page can frame; each page's headers are read once."""
        self._command_edits = Lock()
        """One page edit of a command's menu at a time; `/set`'s source also serializes its saves with the terminal's."""
        self.version = installed().label
        self.update: UpdateInfo | None = None
        """`/update` installing a newer CLAI, until the install ends; a browser that connects meanwhile sees it."""
        self._uv_output: list[str] = []

    # What the browser sees.

    def snapshot(self) -> Snapshot:
        conversation = self.host.conversation
        items = history_items(conversation.messages)
        if self.running:
            items += self._turn.items.values()
        return Snapshot(
            session=SessionInfo(
                id=conversation.conversation_id, title=conversation.title, workspace=self.host.status.workspace
            ),
            items=items,
            running=self.running,
            status=self.status(),
            questions=self.answerer.questions,
            dialogs=self.dialogs.dialogs,
            theme=self.theme(),
            tasks=self.tasks(),
            task_items=self.workers.items(),
            follows_workspace=self.host.sessions is not None and self.host.sessions.follows_workspace,
            version=self.version,
            update=self.update,
        )

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

    def status(self) -> StatusInfo:
        status = self.host.status
        return StatusInfo(
            model=status.model,
            activity=status.activity if self.running else 'ready',
            context_tokens=status.context_tokens,
            context_window=status.context_window,
            cost=None if status.cost is None else str(status.cost),
            streamed_tokens=math.ceil(status.streamed_chars / 4),
            output_tokens=status.output_tokens,
            run=None if self._usage is None else usage_info(self._usage),
            code_mode=self._switch(CODE_MODE),
            speculation=self._switch(SPECULATION),
        )

    def _switch(self, key: str) -> bool | None:
        try:
            return self.host.setting(key) is True
        except KeyError:
            return None

    def switch(self, key: Switch, on: bool) -> None:
        """Flip a run switch as `/set` typed at the prompt does; a running turn keeps the tools it started with."""
        self.host.send_input(f'/set {key} {"true" if on else "false"}')

    def publish_status(self) -> None:
        self._status_sent = time.monotonic()
        self.hub.publish(StatusChanged(status=self.status()))

    # What happens in the session, from the plugin's moments and capability.

    def foreground(self, ctx: RunContext[None]) -> bool:
        """A run of the conversation the shell holds, rather than a background `/fork` or a delegated task."""
        return ctx.conversation_id == self.host.conversation.conversation_id and DelegationTasks.child_id() is None

    def observe(self, event: AgentStreamEvent, *, usage: RunUsage, by_user: bool = False) -> None:
        """Send `event` to browsers; `by_user` marks a `/monty` cell's run, which is not a turn and has no usage to show."""
        if not by_user:
            self._usage = usage
        for message in self._translator.translate(event):
            if by_user:
                # Only the cell itself: the scripted model's closing text is not the user's.
                if isinstance(message, TextDelta | TextDone) or (
                    isinstance(message, ItemAdded) and message.item.kind != 'tool'
                ):
                    continue
                if isinstance(message, ItemAdded):
                    message = ItemAdded(item=replace(message.item, by_user=True))
            self._turn.apply(message)
            self.hub.publish(message)
        if time.monotonic() - self._status_sent >= STATUS_INTERVAL:
            self.publish_status()

    def run_started(self, prompt: str | Sequence[UserContent] | None) -> None:
        """A run of the held conversation started; a run woken by a finished task has no prompt to show."""
        self.running = True
        self._usage = None
        self._turn.clear()
        item = None
        if prompt is not None:
            item = Item(id=f'prompt{time.monotonic_ns()}', kind='user', text=user_text(UserPromptPart(prompt)))
        self.hub.publish(TurnStarted(item=item))
        self.publish_status()

    def run_ended(
        self, outcome: TurnOutcome, *, error: BaseException | None = None, usage: RunUsage | None = None
    ) -> None:
        self.running = False
        self._usage = None
        self._turn.clear()
        self.hub.publish(
            TurnEnded(
                outcome=outcome,
                error=None if error is None else str(error) or type(error).__name__,
                usage=None if usage is None else usage_info(usage),
            )
        )
        self.publish_status()

    def turn_ended(self) -> None:
        """A turn, or a background `/fork`, ended and saved its outcome and usage: the figures and sidebar show it."""
        self.publish_status()
        self.hub.publish(SessionsChanged())

    def conversation_changed(self, event: ConversationChanged) -> None:
        if event.conversation_id == self._shown:
            self.hub.publish(TitleChanged(title=event.title))
        else:
            self._shown = event.conversation_id
            self.workers.reset()
            self.hub.publish(self.snapshot())
        self.hub.publish(SessionsChanged())

    def command_ended(self, event: CommandEnd) -> None:
        if event.error is not None:
            self.hub.publish(Notice(text=str(event.error), level='error', command=event.text))
        elif event.output:
            self.hub.publish(Notice(text=event.output, command=event.text))
        self._settings_changed()
        # `/plugins`, or a plugin's own command such as `/grain`, may have changed a plugin or its settings.
        self.hub.publish(PluginsChanged())

    def _settings_changed(self) -> None:
        """Tell browsers CLAI's settings may have changed: the theme if it did, the status row, and open menus."""
        if (name := self._theme_name()) != self._theme:
            self._theme = name
            self.hub.publish(ThemeChanged(theme=self.theme()))
        self.publish_status()
        self.hub.publish(SettingsChanged())

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
                self.hub.publish(Notice(text='\n'.join(self._uv_output), level=level, command='uv'))
            self.update = UpdateInfo(target=event.target, restarting=True) if event.stage == 'restarting' else None
        self.hub.publish(UpdateChanged(update=self.update))

    # What the browser does.

    def send(self, text: str) -> str | None:
        """Enter a prompt or `/command` as if typed at the prompt, unless it is a bare command that declares a menu.

        That menu opens in the page instead of the terminal: the name of its command is returned, and
        nothing is entered. Otherwise the result is `None`.
        """
        text = text.strip()
        if not text:
            raise ValueError('Type a prompt or a /command.')
        if text.startswith('!'):
            raise ValueError('Shell lines (!) run only in the terminal.')
        if is_command_input(text) and len(text.split()) == 1 and self.host.command_menu(text[1:]) is not None:
            return text[1:]
        self.host.send_input(text)
        return None

    async def complete(self, text: str) -> list[CompletionInfo]:
        return [
            CompletionInfo(
                text=completion.text,
                start=-completion.start_position,
                display=completion.display or completion.text,
                meta=completion.display_meta,
            )
            for completion in await self.host.complete(text)
        ]

    async def models(self) -> list[str]:
        """The saved models and chains `/model` offers, as `PROVIDER:NAME`."""
        return [completion.text for completion in await self.host.complete('/model ') if ':' in completion.text]

    def answer(self, request_id: str, answers: Sequence[AskUserAnswer] | None) -> None:
        """Answer an open question, or decline it with `None`; `ValueError` when the answers do not fit it."""
        self.answerer.reply(request_id, answers)

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
            form=None if info.settings_menu else _form(info),
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
        self.host.send_input(f'/plugins configure {self._plugin(name).name}')

    async def plugin_menu(self, name: str, *, path: Sequence[str] = ()) -> MenuInfo:
        """A plugin's declared settings menu, or the one `path`'s rows open from it, with every row's current value."""
        menu = self.plugin_catalog.settings_menu(self._plugin(name).name)
        if menu is None:
            raise LookupError(f'{name} declares no settings menu here; open its own in the terminal.')
        # Its source may read the keyring or files, so it runs off the loop, as the terminal's menu does.
        return await to_thread.run_sync(_menu_info, menu, path)

    async def edit_menu(self, name: str, key: str, value: str | None, *, path: Sequence[str] = ()) -> MenuEdited:
        """Set a row of a plugin's declared menu to `value`, or reset it with `None` or empty text, as its menu does.

        Raises `ValueError` with the row's own message when the menu rejects the value; nothing is saved then.
        """
        return await self._edit(name, partial(_set_row, path=path, key=key, value=value), path)

    async def run_menu_action(self, name: str, key: str, *, path: Sequence[str] = ()) -> MenuEdited:
        """Run what a row of a plugin's declared menu does when chosen, asking through this page's dialogs."""
        return await self._edit(name, partial(_run_action, path=path, key=key, dialogs=self.dialogs, owner=name), path)

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
            messages=messages, menu=None if menu is None else await to_thread.run_sync(_menu_info, menu, path)
        )

    def answer_dialog(self, dialog_id: str, value: str | None) -> None:
        """Answer a settings menu's open dialog, or cancel it with `None`."""
        self.dialogs.answer(dialog_id, value)

    # Commands' own menus, such as `/set`'s, which the page shows instead of the terminal.

    def _command_menu(self, name: str) -> SettingsMenu:
        """Build the menu bare `/NAME` declares; building may read files, as `/model`'s does, so call it off the loop."""
        menu = self.host.command_menu(name)
        if menu is None:
            raise LookupError(f'/{name} has no menu to show here.')
        return menu

    def _command_info(self, name: str, path: Sequence[str]) -> MenuInfo:
        return _menu_info(self._command_menu(name), path)

    async def command_menu(self, name: str, *, path: Sequence[str] = ()) -> MenuInfo:
        """The menu bare `/NAME` opens, or the one `path`'s rows open from it, with every row's current value."""
        return await to_thread.run_sync(self._command_info, name, path)

    async def edit_command_menu(
        self, name: str, key: str, value: str | None, *, path: Sequence[str] = ()
    ) -> MenuEdited:
        """Set a row of a command's menu to `value`, or reset it with `None` or empty text, as the terminal's menu does."""
        return await self._edit_command(name, partial(_set_row, path=path, key=key, value=value), path)

    async def run_command_menu_action(self, name: str, key: str, *, path: Sequence[str] = ()) -> MenuEdited:
        """Run what a row of a command's menu does when chosen, asking through this page's dialogs."""
        return await self._edit_command(
            name, partial(_run_action, path=path, key=key, dialogs=self.dialogs, owner=f'/{name}'), path
        )

    async def _edit_command(
        self, name: str, edit: Callable[[SettingsMenu], Awaitable[list[str]]], path: Sequence[str]
    ) -> MenuEdited:
        """One edit, printed in the terminal too, after which every page reads the settings again."""
        menu = await to_thread.run_sync(self._command_menu, name)
        async with self._command_edits:
            try:
                messages = await edit(menu)
            finally:
                self._settings_changed()
        for message in messages:
            self.host.console.print(message, markup=False, highlight=False)
        # Built again: the edit may have changed what it lists, as adding or removing a model does.
        return MenuEdited(messages=messages, menu=await to_thread.run_sync(self._command_info, name, path))

    # The sidebar's workers: the delegated tasks.

    @property
    def task_catalog(self) -> TaskCatalog:
        if self.host.tasks is None:
            raise LookupError('This CLAI runs no delegated tasks.')
        return self.host.tasks

    def tasks(self) -> list[TaskEntry]:
        """The conversation's delegated tasks, each parent before its children; none without a task owner."""
        return [] if self.host.tasks is None else [task_entry(task) for task in self.host.tasks.listing()]

    def task_transcript(self, task_id: str) -> TaskTranscript:
        catalog = self.task_catalog
        task = next((task for task in catalog.listing() if task.id == task_id), None)
        if task is None:
            raise LookupError(f'No task {task_id!r} in this conversation.')
        return TaskTranscript(task=task_entry(task), items=history_items(catalog.history(task_id)))

    async def task_action(self, task_id: str, action: TaskAction) -> None:
        """Stop, background, or resume a task, as `/tasks` does."""
        catalog = self.task_catalog
        if action == 'stop':
            await catalog.stop(task_id)
        elif action == 'background':
            catalog.background(task_id)
        else:
            await catalog.resume(task_id)
        # Backgrounding changes the row without an event from the task, which may be waiting on its model.
        for task in catalog.listing():
            if task.id == task_id:
                self.workers.observe(TaskEvent(task=task))

    def steer(self, task_id: str, text: str) -> bool:
        """Send a running task a message, shown in its transcript; `False` once it has finished."""
        text = text.strip()
        steered = self.task_catalog.steer(task_id, text)
        if steered:
            self.workers.steered(task_id, text)
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
        if self.host.sessions is None:
            raise LookupError('This CLAI does not save sessions.')
        return self.host.sessions

    async def projects(self, query: str = '', *, include_archived: bool = False) -> list[Project]:
        """Saved sessions grouped by repository, then checkout, the current ones first and removed folders last."""
        entries = await self.catalog.listing(query=query, archived=None if include_archived else False)
        workspaces = {entry.workspace for entry in entries} | {self.host.status.workspace}
        known = dict(self._identities)

        def resolve() -> dict[str, ProjectIdentity]:
            # A folder removed or recreated since it was last listed resolves again: one stat each, Git only for those.
            return {
                workspace: project_identity(workspace)
                for workspace in workspaces
                if (identity := known.get(workspace)) is None or Path(workspace).exists() == identity.missing
            }

        self._identities.update(await to_thread.run_sync(resolve))
        # Checking whether another process runs a session can spawn a program (`tasklist` on Windows).
        running = [entry for entry in entries if entry.outcome == 'running']
        busy: set[str] = (
            await to_thread.run_sync(lambda: {entry.id for entry in running if _busy(entry)}) if running else set()
        )
        # Every workspace seen so far, not only this search's, tells where a removed folder belonged.
        places = placements(entries, identities=self._identities)
        here = places[self.host.status.workspace]
        checkouts: dict[str, dict[str, tuple[ProjectIdentity, list[SessionEntry]]]] = {}
        for entry in entries:
            identity = places[entry.workspace]
            _, sessions = checkouts.setdefault(identity.key, {}).setdefault(identity.root, (identity, []))
            sessions.append(self._entry(entry, busy=entry.id in busy))
        projects = [
            Project(
                key=key,
                name=next(iter(found.values()))[0].name,
                current=key == here.key,
                worktrees=_worktrees(found, here=here),
            )
            for key, found in checkouts.items()
        ]
        return sorted(projects, key=lambda project: (not project.current, project.key == MISSING))

    def _entry(self, summary: ConversationSummary, *, busy: bool) -> SessionEntry:
        current = summary.id == self.host.conversation.conversation_id
        state: SessionState = summary.outcome
        if current and self.running:
            state = 'running'
        elif summary.outcome == 'running':
            state = 'busy' if busy else 'cancelled'
        return SessionEntry(
            id=summary.id,
            title=summary.title,
            workspace=summary.workspace,
            updated_at=summary.updated_at,
            state=state,
            messages=summary.message_count,
            model=summary.model,
            current=current,
            archived=summary.archived_at is not None,
        )

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
        """Whether `/resume` continues a session saved in `workspace`: here, or there while CLAI follows sessions."""
        if workspace == self.host.status.workspace:
            return True
        if not self.catalog.follows_workspace:
            return False
        try:
            workspace_directory(workspace)
        except ValueError:
            return False
        return True

    def open(self, conversation_id: str) -> None:
        self.host.send_input(f'/resume {conversation_id}')

    def new(self, workspace: str | None = None) -> None:
        """Start a new session, in the directory `workspace` names when CLAI follows sessions' directories."""
        if workspace is not None:
            directory = str(workspace_directory(workspace))
            if directory != self.host.status.workspace:
                if not self.catalog.follows_workspace:
                    raise ValueError(f'This CLAI works in {self.host.status.workspace}. Start CLAI in {directory}.')
                self.host.send_input(f'/workspace {directory}')
                return
        self.host.send_input('/new')

    async def rename(self, conversation_id: str, title: str) -> None:
        await self.catalog.rename(conversation_id, title)
        self.hub.publish(SessionsChanged())

    async def fork(self, conversation_id: str) -> str:
        """Copy a session and switch to the copy."""
        copy = await self.catalog.fork(conversation_id)
        self.hub.publish(SessionsChanged())
        if self._continuable(copy.workspace):
            self.open(copy.id)
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


def _worktrees(
    found: Mapping[str, tuple[ProjectIdentity, list[SessionEntry]]], *, here: ProjectIdentity
) -> list[Worktree]:
    """A project's checkouts, the current one first, each labelled by its branch or else its folder.

    A label two of them share, such as two removed folders of one name, shows the folder's path instead.
    """
    labels = {root: identity.checkout or PurePath(root).name or root for root, (identity, _) in found.items()}
    worktrees = [
        Worktree(
            root=root,
            label=labels[root] if list(labels.values()).count(labels[root]) == 1 else root,
            current=(identity.key, root) == (here.key, here.root),
            missing=identity.missing,
            sessions=sessions,
        )
        for root, (identity, sessions) in found.items()
    ]
    return sorted(worktrees, key=lambda worktree: not worktree.current)


def _form(info: PluginInfo) -> PluginForm | None:
    """A settings form for a plugin that is on and takes settings, unless its model has no JSON Schema."""
    if info.settings_model is None or info.settings is None:
        return None
    try:
        schema = info.settings_model.model_json_schema(by_alias=True)
    except PydanticInvalidForJsonSchema:  # A type JSON Schema cannot express; its own menu or `/plugins add` still can.
        return None
    return PluginForm(json_schema=schema, values=info.settings.model_dump(mode='json', by_alias=True))


def _submenu(menu: SettingsMenu, path: Sequence[str]) -> SettingsMenu:
    """The menu reached from `menu` by choosing each row of `path` in turn; builds them, so call it off the loop."""
    for key in path:
        build = menu.menus.get(key)
        if build is None:
            raise LookupError(f'{menu.source.title} has no menu {key!r}.')
        menu = build()
    return menu


def _menu_info(menu: SettingsMenu, path: Sequence[str]) -> MenuInfo:
    """Every row of the menu at `path` with its current value; reads the menu's source, so call it off the loop."""
    menu = _submenu(menu, path)
    source = menu.source
    return MenuInfo(
        title=source.title,
        fields=[
            MenuField(
                key=row.key,
                label=row.label or row.key,
                description=row.description,
                value=source.current(row),
                default=row.default,
                choices=[Choice(value=choice, label=row.display(choice)) for choice in row.choices],
                allow_custom=row.allow_custom,
                secret=row.secret,
                note=row.note,
                action=row.key in menu.actions,
                menu=row.key in menu.menus,
            )
            for row in source.rows()
        ],
    )


def _row(menu: SettingsMenu, key: str) -> FieldRow:
    row = next((row for row in menu.source.rows() if row.key == key), None)
    if row is None:
        raise LookupError(f'{menu.source.title} has no setting {key!r}.')
    return row


async def _set_row(menu: SettingsMenu, *, path: Sequence[str], key: str, value: str | None) -> list[str]:
    """Set a row as its menu does: empty text resets it, and a value the row rejects raises `ValueError`."""
    menu = await to_thread.run_sync(_submenu, menu, path)
    if key in menu.menus:
        raise LookupError(f'{menu.source.title}: {key!r} opens a menu; it takes no value.')
    row = await to_thread.run_sync(_row, menu, key)
    text = (value or '').strip()
    if not text:
        return [await to_thread.run_sync(menu.source.reset, row)]
    if (problem := await to_thread.run_sync(menu.source.problem, row, text)) is not None:
        raise ValueError(problem)
    return [await to_thread.run_sync(menu.source.apply, row, text)]


async def _run_action(menu: SettingsMenu, *, path: Sequence[str], key: str, dialogs: Dialogs, owner: str) -> list[str]:
    """Run a row's action, asking through `dialogs`; `owner`, the plugin or `/command`, names the menu in errors."""
    menu = await to_thread.run_sync(_submenu, menu, path)
    action = menu.actions.get(key)
    if action is None:
        raise LookupError(f'{owner} has no action {key!r}.')
    return list(await action(dialogs))


def _busy(summary: ConversationSummary) -> bool:
    """Whether another live CLAI is running this saved session now, rather than having died mid-turn."""
    try:
        ensure_inactive(summary)
    except ConversationConflict:
        return True
    return False


@dataclass
class Observer(AbstractCapability[None]):
    """Hands the held conversation's runs and their events to the browser.

    Turns follow these runs rather than the turn moments, which a background `/fork` fires too.
    """

    remote: Remote

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        # A `/monty` cell is not a turn: its result comes as the command's, and its events as any run's.
        if not self.remote.foreground(ctx) or is_monty_cell(ctx):
            return await handler()
        self.remote.run_started(ctx.prompt)
        try:
            result = await handler()
        except get_cancelled_exc_class():
            self.remote.run_ended('cancelled')
            raise
        except Exception as error:
            self.remote.run_ended('failed', error=error)
            raise
        self.remote.run_ended('completed', usage=result.usage)
        return result

    async def on_event(self, ctx: RunContext[None], *, event: AgentStreamEvent) -> None:
        if self.remote.foreground(ctx):
            self.remote.observe(event, usage=ctx.usage, by_user=is_monty_cell(ctx))
