"""One live session's plugins: load, unload, and reload them, between turns or during one.

What every session shares, the declarations, the change lock, and process-scoped plugins, is in
`process_plugins.py`. Discarding a host unloads its plugin.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal
from uuid import uuid4

from anyio import CancelScope, fail_after
from anyio.lowlevel import checkpoint
from pydantic import BaseModel, JsonValue, ValidationError
from rich.console import Console

from pydantic_ai.capabilities import AbstractCapability, AgentCapability, Hooks, WrapperCapability
from pydantic_ai.models import Model
from pydantic_clai2.commands import Commands, added_plugin
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.features import CAPABILITY_REQUIREMENTS
from pydantic_clai2.config.plugin_requirements import (
    Requirements,
    apply_requirements,
    ignored_notice,
    stored_requirements,
    withheld,
)
from pydantic_clai2.config.settings_store import SettingsStore, canonical_plugin_declarations, canonical_plugin_id
from pydantic_clai2.models.profiles import provider_of
from pydantic_clai2.plugins import (
    Conversation,
    DepsT,
    FullScreen,
    HostEvent,
    LoadedPlugin,
    ModelProvider,
    NoSettings,
    Plugin,
    PluginHost,
    PluginInfo,
    PluginLoadFailed,
    PluginLogin,
    PluginOrigin,
    PluginStatus,
    Renderer,
    ResultT,
    SendInput,
    SessionCatalog,
    SessionEnd,
    SessionEndReason,
    SessionStart,
    SidePanel,
    TaskCatalog,
    TurnStart,
    bare_screen,
    collect,
)
from pydantic_clai2.plugins._factories import build, scope, settings_capability
from pydantic_clai2.plugins._git import ADD_USAGE, CHECKOUTS_DIR, checkout_dir, install_git_plugin
from pydantic_clai2.plugins.compatibility import INCLUDED, Binds, included_by
from pydantic_clai2.plugins.describe import describe
from pydantic_clai2.plugins.menus import SettingsMenu
from pydantic_clai2.plugins.process_plugins import LiveRuntime, ProcessPlugins, PromptInputs, handling, same_plugin
from pydantic_clai2.runtime.capability_guard import CapabilitySetupError, PluginGuard
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.spinners import Spinner
from pydantic_clai2.ui.rendering.status import Status, StatusSegment

_LOGGER = logging.getLogger(__name__)

FAILED_SESSION_END_TIMEOUT = 5.0
"""Seconds a plugin that failed to load may spend in `on_session_end` before loading continues without it."""
TURN_NOTICE = 'The running turn keeps the plugins it started with; the agent sees the change on the next prompt.'


class PluginError(Exception):
    """A plugin failed while loading or while handling an event."""

    def __init__(self, plugin: str, error: BaseException) -> None:
        """Name the plugin so the user knows which one to fix."""
        super().__init__(f'Plugin {plugin!r}: {type(error).__name__}: {error}')
        self.plugin = plugin
        self.error = error


class PluginSettingsError(PluginError, ValueError):
    """A plugin rejected its settings while activating, so `/plugins add` does not save them.

    A usage error, like other `ValueError`s from a command, so its message, which can quote a pasted secret, is
    not recorded as a handled error.
    """


@dataclass(kw_only=True)
class PluginEntry(Generic[DepsT]):
    """One `/plugins` row: the saved declaration plus what is loaded right now."""

    declaration: PluginSettings
    path: Path | None
    builtin: bool = False
    project: bool = False
    loaded: LoadedPlugin[DepsT] | None = None
    error: str | None = None
    ignored: str | None = None
    """The notice for saved settings this build ignored at the last load, if any."""
    included_in: str | None = None
    """The loaded plugin that already includes this one, which keeps it off; see `compatibility.INCLUDED`."""

    @property
    def name(self) -> str:
        """The plugin id used by `/plugins` commands."""
        return self.declaration.id

    @property
    def shipped(self) -> bool:
        """Declared by CLAI or the project file, so `add` replaces it and `remove` restores it."""
        return self.builtin or self.project

    @property
    def source(self) -> str:
        """The file path for drop-in plugins, otherwise the import string."""
        if self.path is not None:
            return str(self.path)
        if self.project:
            return f'{self.declaration.factory} (project)'
        return f'{self.declaration.factory} (built-in)' if self.builtin else self.declaration.factory

    @property
    def origin(self) -> PluginOrigin:
        """Where the declaration came from, short enough for a list."""
        if self.project:
            return 'project'
        if self.builtin:
            return 'built-in'
        return 'drop-in' if self.path is not None else 'installed'

    @property
    def own_status(self) -> Literal['on', 'off', 'idle', 'failed']:
        """Loaded, turned off, on but not loaded, or on but failed to load, whether or not another plugin includes it."""
        if self.loaded is not None:
            return 'on'
        if not self.declaration.enabled:
            return 'off'
        return 'failed' if self.error else 'idle'

    @property
    def status(self) -> PluginStatus:
        """`own_status`, or `included` while another loaded plugin includes it."""
        return 'included' if self.included_in is not None else self.own_status

    @property
    def state(self) -> str:
        """Human-readable enabled/loaded/failed state."""
        if self.included_in is not None:
            return f'included in {self.included_in}'
        if not self.declaration.enabled:
            return 'disabled'
        if self.loaded is not None:
            return 'enabled, loaded'
        return f'enabled, failed: {self.error}' if self.error else 'enabled, not loaded'


class SessionPlugins(Generic[DepsT]):
    """One live session's loaded plugins; its shell asks it for capabilities and renderers each turn.

    It is the session's `PluginCatalog` too: a change made through it applies here at once, and every
    other live session of the process follows it (see `ProcessPlugins`).
    """

    def __init__(
        self,
        *,
        store: SettingsStore | None = None,
        process: ProcessPlugins[DepsT] | None = None,
        console: Console,
        commands: Commands,
        session_start: Callable[[], SessionStart],
        builtin: Sequence[PluginSettings] = (),
        project: Sequence[PluginSettings] = (),
        conversation: Conversation | None = None,
        session_id: Callable[[], str | None] = lambda: None,
        status: Status | None = None,
        full_screen: FullScreen = bare_screen,
        enabled: bool = True,
        submit_prompt: Callable[[str], None] | None = None,
        send_input: SendInput | None = None,
        interrupt: Callable[[], bool] = lambda: False,
        sessions: SessionCatalog | None = None,
        setting: Callable[[str], JsonValue] | None = None,
        tasks: Callable[[], TaskCatalog | None] = lambda: None,
        workspace: Callable[[], Path] = Path.cwd,
        running: Callable[[], bool] = lambda: False,
        queued: Callable[[], int] = lambda: 0,
        inputs: Callable[[], PromptInputs | None] = lambda: None,
        live_id: str | None = None,
        default_model: Callable[[], Awaitable[Model | str | None]] | None = None,
        default_model_name: Callable[[], Awaitable[str | None]] | None = None,
    ) -> None:
        """Join `process`, the plugins every live session of this CLAI shares.

        Without one, the session gets a process of its own over `store`, with `builtin` as the
        declarations shipped with CLAI, as a host with one session does. `project` comes from the
        session's `.clai/settings.json`; the store overrides it and the built-ins.

        `enabled=False` lists and loads no plugins at all, without changing anything saved.

        `full_screen` is handed to every host; the shell binds it to the live renderer per prompt.
        `conversation`, `status`, `submit_prompt`, `send_input`, `interrupt`, `sessions`, `setting`, `tasks`,
        `workspace`, `default_model`, `default_model_name`, and `commands` are handed to every host; see `PluginHost` for the defaults. `running`
        says whether a turn runs, `inputs` the prompt holding the input that waits for it, `queued` how much waits
        when there is no prompt, and `live_id` names the session in `host.live`, by default a new ID.
        """
        if process is None:
            if store is None:
                raise TypeError('SessionPlugins needs the process it joins, or a settings store for one of its own.')
            process = ProcessPlugins[DepsT](store=store, builtin=builtin)
        elif store is not None or builtin:
            raise TypeError('A joined process holds the settings store and built-in plugins; pass them to it.')
        self._process = process
        self._console = console
        self._commands = commands
        self._session_start = session_start
        self._full_screen = full_screen
        self._conversation = conversation
        self._session_id = session_id
        self._status = status
        self._submit_prompt = submit_prompt
        self._send_input = send_input
        self._interrupt = interrupt
        self._sessions = sessions
        self._setting = setting
        self._tasks = tasks
        self._workspace = workspace
        self._default_model = default_model
        self._default_model_name = default_model_name
        self._project = canonical_plugin_declarations(project)
        self._entries: dict[str, PluginEntry[DepsT]] = {}
        self._loaded: dict[str, LoadedPlugin[DepsT]] = {}
        # Capabilities that rejected their configuration while a run was set up, by plugin, until it loads again.
        self._suspended: dict[str, list[object]] = {}
        # The capability CLAI built from a `module:Class` declaration's settings, by plugin.
        self._from_settings: dict[str, AbstractCapability[DepsT]] = {}
        self._guards: dict[str, PluginGuard[DepsT]] = {}
        # Plugins unloaded while a turn runs, with their `session_end` reasons; `None` between turns.
        self._retired: list[tuple[str, LoadedPlugin[DepsT], SessionEndReason]] | None = None
        self.enabled = enabled
        self._base = PluginHost[None](
            name='',
            console=console,
            settings={},
            full_screen=full_screen,
            conversation=conversation,
            session_id=session_id,
            status=status,
            submit_prompt=submit_prompt,
            send_input=send_input,
            interrupt=interrupt,
            commands=commands,
            sessions=sessions,
            setting=setting,
            plugins=self,
            tasks=tasks,
            workspace=workspace,
        )
        self.live_session = LiveRuntime(
            live_id=live_id or uuid4().hex,
            host=self._base,
            running=running,
            queued=queued,
            changed=lambda: self._process.live.changed(self.live_session),
            inputs=inputs,
        )
        """This session as `host.live` lists it to process-scoped plugins; it joins the process as it loads."""

    @property
    def process(self) -> ProcessPlugins[DepsT]:
        """The plugins every live session of this CLAI shares."""
        return self._process

    @property
    def plugins_dir(self) -> Path:
        """Folder scanned for drop-in plugin files."""
        return self._process.plugins_dir

    def entries(self) -> list[PluginEntry[DepsT]]:
        """Saved declarations plus drop-in files, keeping the loaded state of each."""
        if not self.enabled:
            return []
        declared = self._process.declared(self._project, console=self._console)
        builtin = self._process.builtin
        refreshed: dict[str, PluginEntry[DepsT]] = {}
        for name in sorted(declared):
            previous = self._entries.get(name)
            path = declared[name].path
            refreshed[name] = PluginEntry(
                declaration=declared[name],
                path=Path(path) if path is not None else None,
                builtin=same_plugin(declared[name], builtin.get(name)),
                project=same_plugin(declared[name], self._project.get(name)),
                loaded=previous.loaded if previous else None,
                error=previous.error if previous else None,
                ignored=previous.ignored if previous else None,
            )
        for name, previous in self._entries.items():
            if name not in refreshed and previous.loaded is not None:
                refreshed[name] = previous
        includes = {
            name: _includes(entry.declaration.factory, entry.loaded)
            for name, entry in refreshed.items()
            if entry.loaded is not None
        }
        for entry in refreshed.values():
            entry.included_in = included_by(entry.declaration.factory, includes)
        self._entries = refreshed
        return list(refreshed.values())

    def _saved(self, entry: PluginEntry[DepsT]) -> PluginSettings:
        """The stored declaration as it is now, without refreshing `entries()` mid-load."""
        return self._process.stored(entry.name) or entry.declaration

    def _registration_order(self) -> list[PluginEntry[DepsT]]:
        """Shipped plugins first, in declaration order, then everything else by name.

        `entries()` sorts by name so the menu and `/plugins list` are easy to scan, but that sort
        must not decide which instructions, renderer, or status segment comes first: alphabetical
        order put `ask_user`'s guidance ahead of `coder`'s. Claiming a built-in's id still counts
        as shipped, so a replaced declaration keeps its position.
        """
        entries = self.entries()
        order = {name: index for index, name in enumerate(self._process.builtin)}
        shipped = sorted((entry for entry in entries if entry.name in order), key=lambda entry: order[entry.name])
        return [*shipped, *(entry for entry in entries if entry.name not in order)]

    def _entry(self, name: str) -> PluginEntry[DepsT]:
        entry = {entry.name: entry for entry in self.entries()}.get(name)
        if entry is None:
            raise ValueError(f'Unknown plugin: {name}')
        return entry

    def capabilities(self) -> list[AgentCapability[DepsT]]:
        """Bound on every run, in load order, minus capabilities left out by `suspend`."""
        return [capability for _, capability in self._active_capabilities()]

    def run_capabilities(self) -> list[AgentCapability[DepsT]]:
        """`capabilities()` as a run binds them, with settings-built capabilities guarded.

        Only a capability CLAI built itself from a `module:Class` declaration's saved settings is
        guarded, and only when nothing in it is a `Hooks`: there a setup `UserError` can only mean
        those settings are wrong. Capabilities a plugin's `get_capabilities` returns, and any policy hook, stay
        unguarded, so whatever they raise still fails closed every turn.
        """
        return [self._guarded(name, capability) for name, capability in self._active_capabilities()]

    def _guarded(self, name: str, capability: AgentCapability[DepsT]) -> AgentCapability[DepsT]:
        built = self._from_settings.get(name)
        if built is None or capability is not built or _binds(built, Hooks):
            return capability
        if name not in self._guards:
            self._guards[name] = PluginGuard[DepsT](built, plugin=name)
        return self._guards[name]

    def _active_capabilities(self) -> list[tuple[str, AgentCapability[DepsT]]]:
        return [
            (name, capability)
            for name, host in self._loaded.items()
            for capability in host.capabilities
            if not any(capability is suspended for suspended in self._suspended.get(name, ()))
        ]

    def suspend(self, error: CapabilitySetupError) -> str:
        """Leave the failed capability out of later runs until its plugin loads again; return the notice."""
        self._suspended.setdefault(error.plugin, []).append(error.capability)
        return (
            f'Leaving the failing {error.plugin} capability out of later turns; '
            f'fix its settings, then run /plugins reload {error.plugin}.'
        )

    def renderers(self) -> list[Renderer]:
        """Consulted before the default display, in load order."""
        return [loaded.plugin.render for loaded in self._loaded.values() if loaded.plugin.has_render]

    def status_segments(self) -> list[StatusSegment]:
        """Appended to the status row, in load order."""
        return [segment for loaded in self._loaded.values() for segment in loaded.status_segments]

    def spinners(self) -> list[Spinner]:
        """Plugin spinners, in load order, so a later plugin wins a name collision."""
        return [spinner for loaded in self._loaded.values() for spinner in loaded.spinners]

    def model_providers(self) -> dict[str, ModelProvider]:
        """Plugin model prefixes; a later plugin wins a prefix collision, as with spinners."""
        return {provider.prefix: provider for loaded in self._loaded.values() for provider in loaded.model_providers}

    def logins(self) -> dict[str, PluginLogin]:
        """Plugin sign-ins for `/login NAME`; a later plugin wins a name collision, as with model prefixes."""
        return {login.name: login for loaded in self._loaded.values() for login in loaded.logins}

    def model_names(self) -> list[str]:
        """Every plugin-offered model, with its prefix, for menus and completions."""
        return [name for provider in self.model_providers().values() for name in provider.names]

    def settings_model(self, model: str) -> str:
        """The model whose `/model settings` controls `model` takes: `PREFIX[@PROFILE]:NAME` as `settings_from:NAME`."""
        _, separator, name = model.partition(':')
        provider = self.model_providers().get(provider_of(model)) if separator else None
        if provider is None or provider.settings_from is None:
            return model
        return f'{provider.settings_from}:{name}'

    async def load_all(self, *, fresh: bool = False) -> None:
        """Load enabled plugins, re-importing after a shell reload so host event types match.

        A declaration whose own module is not installed, such as a built-in saved by another CLAI
        version, is skipped quietly: `/plugins list` still shows the failure. Loading it explicitly
        with `/plugins enable`, `add`, or `reload` still raises.
        """
        if self.enabled:
            self._process.ensure_plugins_dir(self._console)
        self._process.join(self)
        failures: list[PluginLoadFailed] = []
        for entry in self._registration_order():
            # Read afresh: a plugin loaded earlier in this loop may include this one.
            if entry.declaration.enabled and entry.loaded is None and self._entry(entry.name).included_in is None:
                try:
                    await self.load(entry.name, fresh=fresh)
                except PluginError as exc:
                    if not _module_absent(entry, exc.error):
                        failures.append(PluginLoadFailed(plugin=entry.name, error=exc.error))
                        self._console.print(str(exc), style=theme.color(theme.ERROR), markup=False)
                # Settings another build saved are expected in a shared database, not a problem to
                # announce on every start: `/plugins` shows the notice. `load` refreshed the entries.
                ignored = self._entries[entry.name].ignored
                if ignored is not None:
                    _LOGGER.debug('%s', ignored)
        # A process-scoped plugin loaded for this launch alone, as `--web` loads `web`, joins every session.
        for name, loaded in self._process.shared_plugins():
            # After it has started: one that fails to start is not joined.
            async with self._process.loading(name):
                if self._process.shared(name) is loaded:
                    await self.attach(name, loaded)
        # Observers can load after a failing plugin, so report only once startup loading finishes.
        for failure in failures:
            await self.fire(failure)

    async def load(self, name: str, *, fresh: bool = False) -> None:
        """Import, build the plugin, collect its contributions, and fire `session_start`.

        A failure leaves nothing registered. Loading a plugin unloads the plugins it includes. A
        process-scoped plugin is built once: it joins every live session, and loading it where it
        was already built only joins it there.
        """
        self._process.join(self)
        # One load of a name at a time across sessions, so a process-scoped plugin is built once.
        async with self._process.loading(name):
            await self._load_one(name, fresh=fresh)

    async def _load_one(self, name: str, *, fresh: bool) -> None:
        entry = self._entry(name)
        if entry.loaded is not None:
            return
        _check_not_included(entry)
        if (shared := self._process.shared(name)) is not None:
            await self.attach(name, shared)
            return
        declaration, ignored = self._applied(entry)
        entry.ignored = ignored
        activating = process = False
        plugin: Plugin[BaseModel, DepsT] | None = None
        loaded: LoadedPlugin[DepsT] | None = None
        try:
            module = self._process.import_module(name, entry.declaration, fresh=fresh)
            activating = True
            process = scope(module, declaration) == 'process'
            plugin = build(module, declaration, self._host(entry, declaration, process=process))
            built = settings_capability(plugin)
            if built is not None:
                self._from_settings[name] = built
            activating = False
            await plugin.prepare()
            loaded = collect(plugin)
            self._register(loaded)
            entry.loaded = loaded
            self._loaded[name] = loaded
            if process:
                self._process.register_shared(name, loaded=loaded)
            with handling(self.live_session):
                await loaded.dispatch(self._session_start())
        except asyncio.CancelledError:
            await self._failed_load(entry, plugin, loaded)
            raise
        except Exception as exc:
            await self._failed_load(entry, plugin, loaded)
            entry.error = f'{type(exc).__name__}: {exc}'
            if activating and isinstance(exc, ValidationError):
                raise PluginSettingsError(name, exc) from exc
            raise PluginError(name, exc) from exc
        entry.error = None
        await self._unload_included(entry, loaded)
        if process:
            await self._process.share(name, loaded, origin=self)

    def _host(self, entry: PluginEntry[DepsT], declaration: PluginSettings, *, process: bool) -> PluginHost[DepsT]:
        """The plugin's host. A process-scoped plugin's follows the session being handled, else this one."""
        host: PluginHost[DepsT] | None = None

        def save(settings: dict[str, JsonValue]) -> None:
            assert host is not None, 'settings are saved once the plugin is built'
            self._save(entry, host, settings)

        def here() -> PluginHost[None]:
            return self._process.here(self)._base

        submit_prompt, send_input, setting = self._submit_prompt, self._send_input, self._setting
        host = PluginHost[DepsT](
            name=entry.name,
            console=self._console,
            settings=declaration.settings,
            full_screen=self._full_screen,
            conversation=self._conversation,
            session_id=(lambda: here().session_id) if process else self._session_id,
            status=self._status,
            save_settings=save,
            requirements=CAPABILITY_REQUIREMENTS.get(declaration.factory),
            submit_prompt=(lambda text: here().submit_prompt(text)) if process and submit_prompt else submit_prompt,
            send_input=(lambda text, images: here().send_input(text, images=images))
            if process and send_input
            else send_input,
            interrupt=(lambda: here().interrupt()) if process else self._interrupt,
            commands=self._commands,
            sessions=self._sessions,
            setting=(lambda key: here().setting(key)) if process and setting else setting,
            plugins=_HerePlugins(lambda: self._process.here(self)) if process else self,
            tasks=(lambda: here().tasks) if process else self._tasks,
            workspace=(lambda: here().workspace) if process else self._workspace,
            live=self._process.live if process else None,
            default_model=self._default_model,
            default_model_name=self._default_model_name,
        )
        return host

    def _register(self, loaded: LoadedPlugin[DepsT]) -> None:
        """Add the plugin's commands to this session's, warning about any an earlier command hides."""
        for hidden in self._commands.register_many(loaded.commands):
            self._console.print(
                f'/{hidden.name} ({hidden.description}) is hidden by another /{hidden.name} command.',
                style=theme.color(theme.WARNING),
                markup=False,
            )

    async def _unload_included(self, entry: PluginEntry[DepsT], loaded: LoadedPlugin[DepsT]) -> None:
        included = _includes(entry.declaration.factory, loaded)
        for other in self.entries():
            if other.loaded is not None and other.declaration.factory in included:
                await self.unload(other.name)

    async def attach(self, name: str, loaded: LoadedPlugin[DepsT]) -> None:
        """Join a process-scoped plugin that is already running: no `session_start`, it had its own.

        Skipped where the plugin is not declared, as when plugins are off, or already joined.
        """
        entry = next((entry for entry in self.entries() if entry.name == name), None)
        if entry is None or entry.loaded is not None:
            return
        self._register(loaded)
        entry.loaded = loaded
        entry.error = None
        self._loaded[name] = loaded
        await self._unload_included(entry, loaded)

    def holds(self, name: str, loaded: LoadedPlugin[DepsT]) -> bool:
        """Whether this session has `loaded` as its plugin `name`."""
        return self._loaded.get(name) is loaded

    def detach(self, name: str, loaded: LoadedPlugin[DepsT]) -> None:
        """Take a process-scoped plugin out of this session without ending it."""
        if self.holds(name, loaded):
            self._drop(self._entries[name])

    def _applied(self, entry: PluginEntry[DepsT]) -> tuple[PluginSettings, str | None]:
        """The declaration to load, without saved settings that need features this build lacks, and the notice.

        A dropped setting takes the shipped declaration's value, else the plugin's own default. Shipped
        and staged declarations are this build's own or the user's just now, so they are taken as they are.
        """
        saved = entry.declaration
        if entry.name in self._process.staged or entry.shipped or not saved.settings:
            return saved, None
        shipped = self._project.get(entry.name) or self._process.builtin.get(entry.name)
        defaults = shipped.settings if shipped is not None and shipped.factory == saved.factory else {}
        requirements = stored_requirements(self._process.store.plugin_requirements(entry.name), saved.settings)
        applied = apply_requirements(saved.settings, requirements, defaults=defaults)
        if not applied.ignored:
            return saved, None
        # Validated again so the declaration's own defaults, such as `Coder`'s `sub_agents`, apply.
        declaration = PluginSettings.model_validate({**saved.model_dump(), 'settings': applied.settings})
        return declaration, ignored_notice(entry.name, applied.ignored)

    def _save(self, entry: PluginEntry[DepsT], host: PluginHost[DepsT], settings: dict[str, JsonValue]) -> None:
        """Persist a host's saved settings, or, while `/plugins add` is trying them, update what it will save.

        Saved values this build ignored are written back unchanged, with their tags: it cannot judge them.
        """
        update: dict[str, object] = {'settings': settings, 'enabled': True}
        store = self._process.store
        staged = self._process.staged.get(entry.name)
        if staged is not None:
            self._process.staged[entry.name] = staged.model_copy(update=update)
            return
        # The declaration saved now, not at load: another CLAI process may have replaced it since.
        stored = self._process.stored(entry.name)
        if stored is None:
            store.save_plugin(entry.declaration.model_copy(update=update), requires=host.requirements)
            return
        tags = stored_requirements(store.plugin_requirements(entry.name), stored.settings)
        update['settings'] = {**settings, **withheld(stored.settings, tags)}
        store.save_plugin(stored.model_copy(update=update), requires=host.requirements)

    def _requirements(self, entry: PluginEntry[DepsT]) -> Requirements:
        """What the plugin declares for its settings: its host's, or the capability table's while it is not loaded."""
        if entry.loaded is not None:
            return entry.loaded.plugin.host.requirements
        return CAPABILITY_REQUIREMENTS.get(entry.declaration.factory, {})

    async def _failed_load(
        self, entry: PluginEntry[DepsT], plugin: Plugin[BaseModel, DepsT] | None, loaded: LoadedPlugin[DepsT] | None
    ) -> None:
        """Give a plugin that was built the `session_end` it would get on unload, then drop it."""
        try:
            if plugin is not None:
                # Its own task keeps the handler's `CancelledError` apart from ours: the former is
                # reported, the latter propagates. The shield defers scope cancellation until cleanup
                # finishes; the `checkpoint()` below delivers it.
                with handling(self.live_session):  # The task copies the context: this session's end.
                    cleanup = asyncio.create_task(_end_failed_session(plugin))
                try:
                    with CancelScope(shield=True):
                        await asyncio.wait({cleanup})
                except asyncio.CancelledError:
                    cleanup.cancel()
                    await asyncio.wait({cleanup})
                    raise
                if (error := cleanup.result()) is not None:
                    self._report(entry.name, error, SessionEnd)
        finally:
            self._drop(entry)
            self._process.forget_shared(entry.name, loaded=loaded)
        await checkpoint()

    async def unload(self, name: str, *, reason: SessionEndReason = 'exit') -> None:
        """Fire `session_end`, then drop everything the plugin registered.

        During a turn the plugin is dropped now, but `session_end` waits for the turn to end:
        the running agent still uses the capabilities it bound when the turn started. A
        process-scoped plugin leaves every live session and ends after every running turn.
        """
        entry = self._entry(name)
        loaded = entry.loaded
        if loaded is None:
            return
        if loaded is self._process.shared(name):
            await self._process.unload_shared(name, reason=reason, origin=self)
            return
        # Dropped first, so a concurrent unload finds nothing left to end.
        self._drop(entry)
        if not self.retire(name, loaded, reason=reason):
            await self.end(name, loaded, reason=reason)

    def retire(self, name: str, loaded: LoadedPlugin[DepsT], *, reason: SessionEndReason) -> bool:
        """Hold back the end of an unloaded plugin until the running turn is over; `False` between turns."""
        if self._retired is None:
            return False
        self._retired.append((name, loaded, reason))
        return True

    async def end(self, name: str, loaded: LoadedPlugin[DepsT], *, reason: SessionEndReason) -> None:
        """Fire `session_end` on an unloaded plugin, reporting a failure, unless another session's turn still holds it."""
        if not self._process.last_end(loaded):
            return
        try:
            with handling(self.live_session):
                await loaded.dispatch(SessionEnd(reason=reason))
        except Exception as exc:  # noqa: BLE001 -- unloading must finish even if the plugin misbehaves.
            self._report(name, exc, SessionEnd)

    @property
    def in_turn(self) -> bool:
        """Whether a turn is running, so plugin changes reach the agent only on the next prompt."""
        return self._retired is not None

    @asynccontextmanager
    async def turn(self) -> AsyncGenerator[None]:
        """Hold back `session_end` for plugins unloaded while a turn runs, then end them in order.

        Capabilities are bound once per agent run, so a plugin unloaded mid-turn may still have
        tools, hooks, or transports in use. Its teardown runs once the run is over, even if it was
        cancelled. Everything else the plugin declared is gone at once.
        """
        self._retired = []
        try:
            yield
        finally:
            retired, self._retired = self._retired, None
            with CancelScope(shield=True):
                for name, loaded, reason in retired:
                    await self.end(name, loaded, reason=reason)

    def _drop(self, entry: PluginEntry[DepsT]) -> None:
        if entry.loaded is not None:
            self._commands.unregister(entry.loaded.commands)
        entry.loaded = None
        self._loaded.pop(entry.name, None)
        self._suspended.pop(entry.name, None)
        self._from_settings.pop(entry.name, None)
        self._guards.pop(entry.name, None)

    async def restart(self, *, project: Sequence[PluginSettings]) -> None:
        """Load every session-scoped plugin again, with `project` as the project file's declarations.

        As CLAI starting in another directory would: what a plugin read from `host.workspace`
        is read again. Process-scoped plugins, which no session's directory decides, stay loaded
        (so `web` keeps its page, and `observability` what it read at launch, such as the Git email),
        and their capabilities come after the others'.
        """
        kept = [name for name in self._loaded if self._is_shared(name)]
        for name in reversed(list(self._loaded)):
            if name not in kept:
                await self.unload(name)
        self._project = canonical_plugin_declarations(project)
        await self.load_all()
        for name in [name for name in self._loaded if name in kept]:
            self._loaded[name] = self._loaded.pop(name)

    def _is_shared(self, name: str) -> bool:
        """Whether this session's `name` is a process-scoped plugin."""
        loaded = self._loaded.get(name)
        return loaded is not None and loaded is self._process.shared(name)

    async def close(self, reason: SessionEndReason) -> None:
        """Unload every plugin, last loaded first, as the session ends; process-scoped ones only leave it.

        Once no live session remains, the process-scoped plugins end too.
        """
        for name in reversed(list(self._loaded)):
            if self._is_shared(name):
                self.detach(name, self._loaded[name])
            else:
                await self.unload(name, reason=reason)
        self._process.leave(self)
        if not self._process.sessions:
            await self._process.close(reason, origin=self)

    def handles(self, event: type[HostEvent]) -> bool:
        """Whether a loaded plugin handles `event`, so a caller can skip building one that nothing reads."""
        return any(loaded.handles(event) for loaded in self._loaded.values())

    async def fire(self, event: HostEvent) -> None:
        """Dispatch to every loaded plugin. `turn_start` fails closed; the rest report and continue.

        Process-scoped plugins get it as this session's, with `host.live.current()` naming it.
        """
        with handling(self.live_session):
            for name, loaded in list(self._loaded.items()):
                try:
                    await loaded.dispatch(event)
                except Exception as exc:
                    if isinstance(event, TurnStart):
                        raise PluginError(name, exc) from exc
                    self._report(name, exc, type(event))

    async def follow(self, name: str, *, again: bool) -> None:
        """Catch up with a change another session made to `name`: load, unload, or load it again here.

        `again` loads it again if it stays loaded. A failure is reported here, as at startup. A running
        turn keeps what it bound, as for a change made in this session.
        """
        entry = next((entry for entry in self.entries() if entry.name == name), None)
        if entry is None or self._is_shared(name):
            return
        try:
            if entry.loaded is not None and (again or not entry.declaration.enabled):
                await self.unload(name)
            now = next((entry for entry in self.entries() if entry.name == name), None)
            if now is not None and now.declaration.enabled and now.loaded is None and now.included_in is None:
                await self.load(name)
        except PluginError as exc:
            self._console.print(str(exc), style=theme.color(theme.ERROR), markup=False)
        await self._load_released(entry)

    def _report(self, name: str, error: BaseException, event: type[HostEvent]) -> None:
        """Show a plugin handler's failure, which CLAI carries on from, and record it as a handled error."""
        self._console.print(str(PluginError(name, error)), style=theme.color(theme.ERROR), markup=False)
        telemetry.handled_error('Plugin {plugin!r} failed handling {event}', error, plugin=name, event=event.__name__)

    async def enable(self, name: str) -> None:
        """Remember the plugin as enabled and load it now, in every live session."""
        await self._process.change(name, lambda: self._enable(name), origin=self, again=False)

    async def _enable(self, name: str) -> None:
        entry = self._entry(name)
        _check_not_included(entry)
        _requested('enable', name)
        declaration = entry.declaration.model_copy(update={'enabled': True})
        self._process.store.save_plugin(declaration, requires=self._requirements(entry))
        await self._load_enabled(entry)

    async def _load_enabled(self, entry: PluginEntry[DepsT]) -> None:
        """Load an enabled declaration and remember requirements learned from its plugin."""
        await self.load(entry.name)
        # `load` refreshed the entries; only a loaded plugin can say what its settings need.
        loaded = self._entries[entry.name].loaded
        host = loaded.plugin.host if loaded is not None else None
        if host is not None and host.requirements:
            self._process.store.save_plugin(self._saved(entry), requires=host.requirements)

    async def disable(self, name: str) -> None:
        """Unload the plugin now, in every live session, and remember it as disabled."""
        await self._process.change(name, lambda: self._disable(name), origin=self, again=False)

    async def _disable(self, name: str) -> None:
        entry = self._entry(name)
        _requested('disable', name)
        requires = self._requirements(entry)
        await self.unload(name)
        self._process.store.save_plugin(entry.declaration.model_copy(update={'enabled': False}), requires=requires)
        await self._load_released(entry)

    async def remove(self, name: str) -> str:
        """Unload the plugin and forget its saved declaration; a shipped declaration comes back as declared."""
        return await self._process.change(name, lambda: self._remove(name), origin=self, again=True)

    async def _remove(self, name: str) -> str:
        entry = self._entry(name)
        _requested('remove', name)
        requires = self._requirements(entry)
        await self.unload(name)
        checkout_notice = ''
        # A relative path never names a drop-in, only an approval saved before paths were anchored: forget it.
        if entry.path is not None and entry.path.is_absolute() and not entry.shipped:
            # Unrelated file plugins remain removable even when the drop-in directory cannot be resolved.
            if entry.path.parent.parent.name == CHECKOUTS_DIR and entry.path.parent == checkout_dir(
                self.plugins_dir, name
            ):
                checkout_notice = (
                    f' Checkout kept at {entry.path.parent}. Delete that directory before reinstalling from Git.'
                )
            else:
                self._process.store.save_plugin(
                    entry.declaration.model_copy(update={'enabled': False}), requires=requires
                )
                await self._load_released(entry)
                return f'Disabled {name}. Delete {entry.path} to remove the plugin itself.'
        self._process.store.delete_plugin(name)
        shipped = self._project.get(name) or self._process.builtin.get(name)
        if shipped is not None and shipped.enabled and self._entry(name).included_in is None:
            await self.load(name)
        else:
            await self._load_released(entry)
        if shipped is None:
            return f'Removed {name}.{checkout_notice}'
        origin = 'declared by the project' if name in self._project else 'built in'
        return (
            f'{name} is {origin}; restored its defaults. Use /plugins disable {name} to turn it off.{checkout_notice}'
        )

    async def _load_released(self, owner: PluginEntry[DepsT]) -> None:
        """Load the enabled plugins `owner` could include that it no longer keeps off."""
        included = INCLUDED.get(owner.declaration.factory, {})
        for entry in self._registration_order():
            if entry.declaration.factory in included and entry.declaration.enabled and entry.included_in is None:
                # Like `load_all`: one that fails is reported and keeps its error, and the rest still load.
                try:
                    await self.load(entry.name)
                except PluginError as exc:
                    self._console.print(str(exc), style=theme.color(theme.ERROR), markup=False)

    async def reload(self, name: str) -> None:
        """Unload, re-import the module, and load again, in every live session."""
        await self._process.change(name, lambda: self._reload(name), origin=self, again=True)

    async def _reload(self, name: str) -> None:
        if not self._entry(name).declaration.enabled:
            raise ValueError(f'Plugin {name} is disabled; enable it before reloading.')
        _requested('reload', name)
        await self._load_again(name, fresh=True)

    async def _load_again(self, name: str, *, fresh: bool = False) -> None:
        """Unload and load again, then load what it no longer includes, also when loading fails."""
        await self.unload(name)
        try:
            await self.load(name, fresh=fresh)
        finally:
            # New settings may change what it includes, such as `coder` with `sub_agents` off.
            await self._load_released(self._entry(name))

    async def configure(self, name: str) -> str:
        """Open the plugin's settings menu, then load it again if its saved settings changed."""
        _requested('configure', name)
        async with self._settings_open(name) as loaded:
            if not loaded.plugin.has_configure:
                raise ValueError(f'Plugin {name} has no settings menu; replace its declaration with /plugins add.')
            return await loaded.plugin.configure()

    @asynccontextmanager
    async def _settings_open(self, name: str) -> AsyncGenerator[LoadedPlugin[DepsT]]:
        """The loaded plugin, its menu held open for the block; afterwards it loads again if its saved settings changed."""
        entry = self._entry(name)
        _check_not_included(entry)
        loaded = entry.loaded
        if loaded is None:
            raise ValueError(f'Plugin {name} is not loaded; enable it before configuring.')
        editing = self._process.editing
        if name in editing:
            raise ValueError(f'The settings menu of {name} is already open; finish there first.')
        editing.add(name)
        before = entry.declaration.settings
        try:
            yield loaded
        finally:
            editing.discard(name)
            # Also when the menu fails after saving, so the running plugin matches what is saved.
            if self._entry(name).declaration.settings != before:
                await self._load_again(name)
                await self._process.follow(name, origin=self, again=True)

    def listing(self) -> list[PluginInfo]:
        """Every plugin as `/plugins` shows it, for `host.plugins`; see `PluginCatalog`."""
        return [_info(entry) for entry in self.entries()]

    async def save_settings(self, name: str, settings: BaseModel) -> None:
        """Save a loaded plugin's settings through its host, then load it again if they changed, as `configure` does."""
        async with self._process.changing, self._settings_open(name) as loaded:
            model = loaded.plugin.settings_type
            if model is NoSettings:
                raise ValueError(f'Plugin {name} takes no settings.')
            if not isinstance(settings, model):
                raise TypeError(f'Plugin {name} takes {model.__name__}, not {type(settings).__name__}.')
            _requested('configure', name)
            loaded.plugin.host.save_settings(settings)

    def settings_menu(self, name: str) -> SettingsMenu | None:
        """A loaded plugin's declared settings menu, for `host.plugins`; see `PluginCatalog`."""
        loaded = self._entry(name).loaded
        return None if loaded is None else loaded.plugin.settings_menu()

    def side_panels(self) -> dict[str, tuple[SidePanel, ...]]:
        """Loaded plugins' side panels, for `host.plugins`; see `PluginCatalog`."""
        return {name: loaded.side_panels for name, loaded in self._loaded.items() if loaded.side_panels}

    async def edit_settings(self, name: str, edit: Callable[[SettingsMenu], Awaitable[ResultT]]) -> ResultT:
        """Edit through a loaded plugin's declared menu, then load it again if its settings changed; see `PluginCatalog`."""
        async with self._settings_open(name) as loaded:
            menu = loaded.plugin.settings_menu()
            if menu is None:
                raise ValueError(f'Plugin {name} declares no settings menu; /plugins configure {name} opens its own.')
            _requested('configure', name)
            return await edit(menu)

    def configurable(self, name: str) -> bool:
        """Whether the plugin is loaded and offers a settings menu by overriding `configure`."""
        loaded = self._entry(name).loaded
        return loaded is not None and loaded.plugin.has_configure

    async def _configure_new(self, name: str, message: str) -> str:
        """After enable or add, open a newly loaded plugin's settings menu, if it has one."""
        if not self.configurable(name):
            return message
        return f'{message}\n{await self.configure(name)}'

    async def _add(self, rest: list[str]) -> str:
        """Back `/plugins add`, replacing a shipped declaration with the same id."""
        name = rest[0] if rest else ''  # No id matches '', so `added_plugin` reports the usage error.
        existing = next((entry for entry in self.entries() if entry.name == name), None)
        if existing is not None and not existing.shipped:
            raise ValueError(f'Plugin {name} already exists; remove its declaration before replacing it.')
        declaration = added_plugin(['add', *rest])
        loaded = existing is not None and existing.loaded is not None
        # Settings are plaintext, so they are saved only once the plugin accepts them: settings it
        # rejects, perhaps a pasted secret, never reach the store.
        staged, store = self._process.staged, self._process.store
        staged[name] = declaration
        try:
            # Like `enable`, refuse an included plugin before unloading anything, and save nothing.
            _check_not_included(self._entry(name))
        except ValueError:
            del staged[name]
            raise
        if existing is not None:
            await self.unload(name)
        try:
            await self.load(name)
        except PluginSettingsError:
            del staged[name]
            if loaded:
                await self.load(name)
            raise
        except BaseException:
            requires = self._requirements(self._entry(name))
            store.save_plugin(staged.pop(name), requires=requires)
            raise
        requires = self._requirements(self._entry(name))
        store.save_plugin(staged.pop(name), requires=requires)
        _requested('add', name)
        await self._process.follow(name, origin=self, again=True)
        if existing is None:
            return await self._configure_new(name, f'Added and loaded {name}.')
        kind = 'project' if existing.project else 'built-in'
        return await self._configure_new(name, f'Replaced {kind} {name}.')

    async def command(self, args: list[str]) -> str:
        """Back `/plugins` with arguments; changes apply now and are saved.

        During a turn, a change to what the agent gets says that it waits for the next prompt.
        """
        before = self.capabilities()
        message = await self._command(args)
        after = self.capabilities()
        if self.in_turn and (len(after) != len(before) or any(new is not old for new, old in zip(after, before))):
            return f'{message}\n{TURN_NOTICE}'
        return message

    async def _command(self, args: list[str]) -> str:
        if not args or args == ['list']:
            return (
                '\n'.join(f'{entry.name}: {entry.source} ({entry.state})' for entry in self.entries()) or 'No plugins.'
            )
        action, *rest = args
        if action == 'add' and not rest:
            raise ValueError(ADD_USAGE)
        if action == 'add' and len(rest) == 1:
            async with install_git_plugin(
                rest[0], plugins_dir=self.plugins_dir, names=[entry.name for entry in self.entries()]
            ) as declaration:
                if any(entry.name == declaration.id for entry in self.entries()):
                    raise ValueError(f'Plugin {declaration.id} already exists; it has not been changed.')
                self._process.store.save_plugin(declaration, overwrite=False)
            _requested('add', declaration.id)
            await self._load_enabled(self._entry(declaration.id))
            await self._process.follow(declaration.id, origin=self, again=False)
            return await self._configure_new(declaration.id, f'Added and loaded {declaration.id}.')
        if rest:
            rest[0] = canonical_plugin_id(rest[0])
        if action == 'add':
            return await self._add(rest)
        if len(rest) != 1:
            raise ValueError(
                'Usage: /plugins [list|add GIT_URL|add ID MODULE[:ATTR] [JSON]|enable ID|disable ID|remove ID|reload ID'
                '|configure ID]'
            )
        name = rest[0]
        if action == 'remove':
            return await self.remove(name)
        if action == 'configure':
            return await self.configure(name)
        if action == 'enable' and self._entry(name).loaded is None:
            await self.enable(name)
            return await self._configure_new(name, self._with_notice(name, f'Enabled {name}.'))
        actions = {
            'enable': (self.enable, 'Enabled'),
            'disable': (self.disable, 'Disabled'),
            'reload': (self.reload, 'Reloaded'),
        }
        if action not in actions:
            raise ValueError(f'Unknown plugins action: {action}')
        run, past = actions[action]
        await run(name)
        return self._with_notice(name, f'{past} {name}.')

    def _with_notice(self, name: str, message: str) -> str:
        """`message`, then the plugin's ignored-settings notice when its last load had one."""
        ignored = self._entry(name).ignored
        return f'{message}\n{ignored}' if ignored else message


PluginLoader = SessionPlugins
"""The former name of `SessionPlugins`, which tests still use."""


class _HerePlugins(Generic[DepsT]):
    """A process-scoped plugin's `host.plugins`: the `PluginCatalog` of the session being handled."""

    def __init__(self, here: Callable[[], SessionPlugins[DepsT]]) -> None:
        self._here = here

    def listing(self) -> list[PluginInfo]:
        return self._here().listing()

    async def enable(self, name: str) -> None:
        await self._here().enable(name)

    async def disable(self, name: str) -> None:
        await self._here().disable(name)

    async def reload(self, name: str) -> None:
        await self._here().reload(name)

    async def save_settings(self, name: str, settings: BaseModel) -> None:
        await self._here().save_settings(name, settings)

    def settings_menu(self, name: str) -> SettingsMenu | None:
        return self._here().settings_menu(name)

    def side_panels(self) -> dict[str, tuple[SidePanel, ...]]:
        return self._here().side_panels()

    async def edit_settings(self, name: str, edit: Callable[[SettingsMenu], Awaitable[ResultT]]) -> ResultT:
        return await self._here().edit_settings(name, edit)


def _binds(capability: AbstractCapability[DepsT], kind: Binds) -> bool:
    """Whether any part of `capability` is a `kind`."""
    if isinstance(capability, kind):
        return True
    # `apply` does not descend into a wrapper's lone leaf, so look behind every wrapper too.
    if isinstance(capability, WrapperCapability) and _binds(capability.wrapped, kind):
        return True
    parts: list[AbstractCapability[DepsT]] = []
    capability.apply(parts.append)
    return any(part is not capability and _binds(part, kind) for part in parts)


def _includes(factory: str, loaded: LoadedPlugin[DepsT]) -> frozenset[str]:
    """The factories a loaded plugin includes, given the capabilities it actually binds."""
    bound: list[AbstractCapability[DepsT]] = [
        capability for capability in loaded.capabilities if isinstance(capability, AbstractCapability)
    ]
    return frozenset(
        included
        for included, kind in INCLUDED.get(factory, {}).items()
        if any(_binds(capability, kind) for capability in bound)
    )


def _check_not_included(entry: PluginEntry[DepsT]) -> None:
    """Refuse to turn on a plugin that a loaded plugin already includes."""
    if entry.included_in is not None:
        raise ValueError(f'{entry.name} is included in {entry.included_in}; disable {entry.included_in} to use it.')


def _info(entry: PluginEntry[DepsT]) -> PluginInfo:
    """A `/plugins` row for `host.plugins`, with the settings model and saved settings of a loaded plugin."""
    loaded = entry.loaded
    model = loaded.plugin.settings_type if loaded is not None else NoSettings
    settings = loaded.plugin.host.settings(model) if loaded is not None and model is not NoSettings else None
    return PluginInfo(
        name=entry.name,
        description=describe(entry.declaration.factory, entry.path),
        origin=entry.origin,
        source=entry.declaration.factory if entry.path is None else str(entry.path),
        status=entry.status,
        included_in=entry.included_in,
        error=entry.error,
        notice=entry.ignored,
        provides=loaded.summary() if loaded is not None else '',
        has_menu=loaded is not None and loaded.plugin.has_configure,
        settings_menu=loaded is not None and loaded.plugin.settings_menu() is not None,
        settings_model=None if settings is None else model,
        settings=settings,
    )


def _requested(action: str, name: str) -> None:
    """UI telemetry for an action on a plugin known to exist (never a mistyped name), before disabling `observability`."""
    telemetry.record('plugin {plugin} {action}', plugin=name, action=action)


def _module_absent(entry: PluginEntry[DepsT], error: BaseException) -> bool:
    """Whether the plugin's own module (or a parent package) is missing, not one of its imports."""
    if entry.path is not None or not isinstance(error, ModuleNotFoundError) or error.name is None:
        return False
    module_name = entry.declaration.factory.partition(':')[0]
    return module_name == error.name or module_name.startswith(f'{error.name}.')


async def _end_failed_session(plugin: Plugin[BaseModel, DepsT]) -> BaseException | None:
    """Return the handler's failure rather than raising it."""
    try:
        with fail_after(FAILED_SESSION_END_TIMEOUT):
            await plugin.on_session_end(SessionEnd(reason='error'))
    except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- reported by the caller.
        return exc
    return None
