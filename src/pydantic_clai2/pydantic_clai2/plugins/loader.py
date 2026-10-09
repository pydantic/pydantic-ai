"""Load, unload, and reload plugins, between turns or during one. Discarding a host unloads its plugin."""

import asyncio
import importlib
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Generic

from anyio import CancelScope, fail_after
from anyio.lowlevel import checkpoint
from pydantic import BaseModel, JsonValue, ValidationError
from rich.console import Console

from pydantic_ai.capabilities import AbstractCapability, AgentCapability, Hooks, WrapperCapability
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
    Plugin,
    PluginHost,
    PluginLoadFailed,
    PluginLogin,
    Renderer,
    SessionEnd,
    SessionEndReason,
    SessionStart,
    TurnStart,
    bare_screen,
    collect,
)
from pydantic_clai2.plugins._factories import build, import_file, settings_capability
from pydantic_clai2.plugins._git import ADD_USAGE, CHECKOUTS_DIR, checkout_dir, install_git_plugin
from pydantic_clai2.plugins.compatibility import INCLUDED, Binds, included_by
from pydantic_clai2.runtime.capability_guard import CapabilitySetupError, PluginGuard
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.spinners import Spinner
from pydantic_clai2.ui.rendering.status import Status, StatusSegment

# Local settings databases from before the package move, or from before plugins were declared as
# `Plugin` classes, can still name the old factories.
_MOVED_FACTORIES = {
    'pydantic_ai_harness.coder:Coder': 'pydantic_clai2.builtin_plugins.coder',
    'pydantic_ai_harness:Coder': 'pydantic_clai2.builtin_plugins.coder',
    'pydantic_clai2.sessions': 'pydantic_clai2.runtime.sessions',
    'pydantic_clai2.ask_user_menu:activate': 'pydantic_clai2.builtin_plugins.ask_user_menu',
    'pydantic_clai2.builtin_plugins.ask_user_menu:activate': 'pydantic_clai2.builtin_plugins.ask_user_menu',
    **{
        f'pydantic_clai2.{name}': f'pydantic_clai2.builtin_plugins.{name}'
        for name in (
            'repo_context',
            'compaction',
            'logfire',
            'notifications',
            'github',
            'pylon',
            'google_workspace',
            'day_ai',
            'ordinal',
            'notion',
            'slack',
            'logfire_mcp',
            'posthog',
            'grain',
            'linear',
        )
    },
}
_RETIRED_BUILTINS: dict[str, PluginSettings] = {
    'google_workspace': PluginSettings(
        id='google_workspace', factory='pydantic_ai_harness.google_workspace:GoogleWorkspace', enabled=False
    ),
    'ordinal': PluginSettings(id='ordinal', factory='pydantic_ai_harness.ordinal:Ordinal', enabled=False),
    'slack': PluginSettings(id='slack', factory='pydantic_ai_harness.slack:Slack', enabled=False),
    'grain': PluginSettings(id='grain', factory='pydantic_ai_harness.grain:Grain', enabled=False),
}
"""Former built-in declarations. A stored copy of one loads the built-in now declared under its id."""


TURN_NOTICE = 'The running turn keeps the plugins it started with; the agent sees the change on the next prompt.'


class PluginError(Exception):
    """A plugin failed while loading or while handling an event."""

    def __init__(self, plugin: str, error: BaseException) -> None:
        """Name the plugin so the user knows which one to fix."""
        super().__init__(f'Plugin {plugin!r}: {type(error).__name__}: {error}')
        self.plugin = plugin
        self.error = error


class PluginSettingsError(PluginError):
    """A plugin rejected its settings while activating, so `/plugins add` does not save them."""


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
    def state(self) -> str:
        """Human-readable enabled/loaded/failed state."""
        if self.included_in is not None:
            return f'included in {self.included_in}'
        if not self.declaration.enabled:
            return 'disabled'
        if self.loaded is not None:
            return 'enabled, loaded'
        return f'enabled, failed: {self.error}' if self.error else 'enabled, not loaded'


class PluginLoader(Generic[DepsT]):
    """Own every loaded plugin; the shell asks it for capabilities and renderers each turn."""

    def __init__(
        self,
        *,
        store: SettingsStore,
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
    ) -> None:
        """`builtin` ships with CLAI, `project` comes from `.clai/settings.json`; the store overrides both.

        `enabled=False` lists and loads no plugins at all, without changing anything saved.

        `full_screen` is handed to every host; the shell binds it to the live renderer per prompt.
        `conversation` and `status` are handed to every host; see `PluginHost` for the defaults.
        """
        self._store = store
        self._console = console
        self._commands = commands
        self._session_start = session_start
        self._full_screen = full_screen
        self._conversation = conversation
        self._session_id = session_id
        self._status = status
        self._builtin = canonical_plugin_declarations(builtin)
        self._project = canonical_plugin_declarations(project)
        self._entries: dict[str, PluginEntry[DepsT]] = {}
        self._loaded: dict[str, LoadedPlugin[DepsT]] = {}
        # A `/plugins add` declaration being tried before it is saved, so rejected settings never reach the store.
        self._staged: dict[str, PluginSettings] = {}
        # Capabilities that rejected their configuration while a run was set up, by plugin, until it loads again.
        self._suspended: dict[str, list[object]] = {}
        # The capability CLAI built from a `module:Class` declaration's settings, by plugin.
        self._from_settings: dict[str, AbstractCapability[DepsT]] = {}
        self._guards: dict[str, PluginGuard[DepsT]] = {}
        # Plugins unloaded while a turn runs, with their `session_end` reasons; `None` between turns.
        self._retired: list[tuple[str, LoadedPlugin[DepsT], SessionEndReason]] | None = None
        self.enabled = enabled

    @property
    def plugins_dir(self) -> Path:
        """Folder scanned for drop-in plugin files."""
        return self._store.plugins_dir

    def entries(self) -> list[PluginEntry[DepsT]]:
        """Saved declarations plus drop-in files, keeping the loaded state of each."""
        if not self.enabled:
            return []
        folder = self._discover()
        declared = {declaration.id: self._upgrade(declaration) for declaration in self._store.plugins()}
        declared.update(self._staged)
        for name in folder.keys() - declared.keys():
            declared[name] = PluginSettings(id=name, factory=name, path=str(folder[name]))
        for shipped in (self._project, self._builtin):
            for name in shipped.keys() - declared.keys():
                declared[name] = shipped[name]
        refreshed: dict[str, PluginEntry[DepsT]] = {}
        for name in sorted(declared):
            previous = self._entries.get(name)
            path = declared[name].path
            refreshed[name] = PluginEntry(
                declaration=declared[name],
                path=Path(path) if path is not None else None,
                builtin=_same_plugin(declared[name], self._builtin.get(name)),
                project=_same_plugin(declared[name], self._project.get(name)),
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

    def _upgrade(self, saved: PluginSettings) -> PluginSettings:
        """Resolve moved plugin paths and retired harness built-ins in stored declarations.

        Preserve custom settings when a local database uses an old CLAI module path.
        """
        old = saved
        factory = _MOVED_FACTORIES.get(saved.factory)
        if factory is not None:
            saved = saved.model_copy(update={'factory': factory})
        current = self._builtin.get(saved.id)
        if current is not None and _same_plugin(old, _RETIRED_BUILTINS.get(saved.id)):
            return current.model_copy(update={'enabled': saved.enabled})
        return saved

    def _saved(self, entry: PluginEntry[DepsT]) -> PluginSettings:
        """The stored declaration as it is now, without refreshing `entries()` mid-load."""
        return next((saved for saved in self._store.plugins() if saved.id == entry.name), entry.declaration)

    def _registration_order(self) -> list[PluginEntry[DepsT]]:
        """Shipped plugins first, in declaration order, then everything else by name.

        `entries()` sorts by name so the menu and `/plugins list` are easy to scan, but that sort
        must not decide which instructions, renderer, or status segment comes first: alphabetical
        order put `ask_user`'s guidance ahead of `coder`'s. Claiming a built-in's id still counts
        as shipped, so a replaced declaration keeps its position.
        """
        entries = self.entries()
        order = {name: index for index, name in enumerate(self._builtin)}
        shipped = sorted((entry for entry in entries if entry.name in order), key=lambda entry: order[entry.name])
        return [*shipped, *(entry for entry in entries if entry.name not in order)]

    def _ensure_plugins_dir(self) -> None:
        """Create the drop-in folder, so installing a plugin is one copy into a folder that already exists."""
        try:
            self._store.plugins_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._console.print(
                f'Cannot create the plugins folder: {exc}', style=theme.color(theme.ERROR), markup=False
            )

    def _discover(self) -> dict[str, Path]:
        folder = self._store.plugins_dir
        if not folder.is_dir():
            return {}
        found: dict[str, Path] = {}
        try:
            children = sorted(folder.iterdir())
        except OSError as exc:
            self._console.print(f'Cannot discover plugins: {exc}', style=theme.color(theme.ERROR), markup=False)
            return {}
        for child in children:
            name = child.stem if child.suffix == '.py' else child.name
            if not name.isidentifier() or name.startswith('_'):
                continue
            name = canonical_plugin_id(name)
            if child.is_file() and child.suffix == '.py':
                found[name] = child
            elif (child / '__init__.py').is_file():
                found[name] = child / '__init__.py'
        return found

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
            self._ensure_plugins_dir()
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
                # `load` refreshed the entries, so read the notice from the current one.
                ignored = self._entries[entry.name].ignored
                if ignored is not None:
                    self._console.print(ignored, style=theme.color(theme.WARNING), markup=False)
        # Observers can load after a failing plugin, so report only once startup loading finishes.
        for failure in failures:
            await self.fire(failure)

    async def load(self, name: str, *, fresh: bool = False) -> None:
        """Import, build the plugin, collect its contributions, and fire `session_start`.

        A failure leaves nothing registered. Loading a plugin unloads the plugins it includes.
        """
        entry = self._entry(name)
        if entry.loaded is not None:
            return
        _check_not_included(entry)
        declaration, ignored = self._applied(entry)
        entry.ignored = ignored

        def save(settings: dict[str, JsonValue]) -> None:
            self._save(entry, host, settings)

        host = PluginHost[DepsT](
            name=name,
            console=self._console,
            settings=declaration.settings,
            full_screen=self._full_screen,
            conversation=self._conversation,
            session_id=self._session_id,
            status=self._status,
            save_settings=save,
            requirements=CAPABILITY_REQUIREMENTS.get(declaration.factory),
        )
        activating = False
        plugin: Plugin[BaseModel, DepsT] | None = None
        try:
            module = self._import(entry, fresh=fresh)
            activating = True
            plugin = build(module, declaration, host)
            built = settings_capability(plugin)
            if built is not None:
                self._from_settings[name] = built
            loaded = collect(plugin)
            activating = False
            self._commands.register_many(loaded.commands)
            entry.loaded = loaded
            self._loaded[name] = loaded
            await loaded.dispatch(self._session_start())
        except asyncio.CancelledError:
            await self._failed_load(entry, plugin)
            raise
        except Exception as exc:
            await self._failed_load(entry, plugin)
            entry.error = f'{type(exc).__name__}: {exc}'
            if activating and isinstance(exc, ValidationError):
                raise PluginSettingsError(name, exc) from exc
            raise PluginError(name, exc) from exc
        entry.error = None
        included = _includes(entry.declaration.factory, loaded)
        for other in self.entries():
            if other.loaded is not None and other.declaration.factory in included:
                await self.unload(other.name)

    def _applied(self, entry: PluginEntry[DepsT]) -> tuple[PluginSettings, str | None]:
        """The declaration to load, without saved settings that need features this build lacks, and the notice.

        A dropped setting takes the shipped declaration's value, else the plugin's own default. Shipped
        and staged declarations are this build's own or the user's just now, so they are taken as they are.
        """
        saved = entry.declaration
        if entry.name in self._staged or entry.shipped or not saved.settings:
            return saved, None
        shipped = self._project.get(entry.name) or self._builtin.get(entry.name)
        defaults = shipped.settings if shipped is not None and shipped.factory == saved.factory else {}
        requirements = stored_requirements(self._store.plugin_requirements(entry.name), saved.settings)
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
        staged = self._staged.get(entry.name)
        if staged is not None:
            self._staged[entry.name] = staged.model_copy(update=update)
            return
        # The declaration saved now, not at load: another CLAI process may have replaced it since.
        stored = next((saved for saved in self._store.plugins() if saved.id == entry.name), None)
        if stored is None:
            self._store.save_plugin(entry.declaration.model_copy(update=update), requires=host.requirements)
            return
        tags = stored_requirements(self._store.plugin_requirements(entry.name), stored.settings)
        update['settings'] = {**settings, **withheld(stored.settings, tags)}
        self._store.save_plugin(stored.model_copy(update=update), requires=host.requirements)

    def _requirements(self, entry: PluginEntry[DepsT]) -> Requirements:
        """What the plugin declares for its settings: its host's, or the capability table's while it is not loaded."""
        if entry.loaded is not None:
            return entry.loaded.plugin.host.requirements
        return CAPABILITY_REQUIREMENTS.get(entry.declaration.factory, {})

    async def _failed_load(self, entry: PluginEntry[DepsT], plugin: Plugin[BaseModel, DepsT] | None) -> None:
        """Give a plugin that was built the `session_end` it would get on unload, then drop it."""
        try:
            if plugin is not None:
                # Its own task keeps the handler's `CancelledError` apart from ours: the former is
                # reported, the latter propagates. The shield defers scope cancellation until cleanup
                # finishes; the `checkpoint()` below delivers it.
                cleanup = asyncio.create_task(_end_failed_session(plugin))
                try:
                    with CancelScope(shield=True):
                        await asyncio.wait({cleanup})
                except asyncio.CancelledError:
                    cleanup.cancel()
                    await asyncio.wait({cleanup})
                    raise
                if (error := cleanup.result()) is not None:
                    self._console.print(
                        str(PluginError(entry.name, error)), style=theme.color(theme.ERROR), markup=False
                    )
        finally:
            self._drop(entry)
        await checkpoint()

    async def unload(self, name: str, *, reason: SessionEndReason = 'exit') -> None:
        """Fire `session_end`, then drop everything the plugin registered.

        During a turn the plugin is dropped now, but `session_end` waits for the turn to end:
        the running agent still uses the capabilities it bound when the turn started.
        """
        entry = self._entry(name)
        loaded = entry.loaded
        if loaded is None:
            return
        if self._retired is not None:
            self._drop(entry)
            self._retired.append((name, loaded, reason))
            return
        try:
            await self._end(name, loaded, reason=reason)
        finally:
            self._drop(entry)

    async def _end(self, name: str, loaded: LoadedPlugin[DepsT], *, reason: SessionEndReason) -> None:
        try:
            await loaded.dispatch(SessionEnd(reason=reason))
        except Exception as exc:  # noqa: BLE001 -- unloading must finish even if the plugin misbehaves.
            self._console.print(str(PluginError(name, exc)), style=theme.color(theme.ERROR), markup=False)

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
                    await self._end(name, loaded, reason=reason)

    def _drop(self, entry: PluginEntry[DepsT]) -> None:
        if entry.loaded is not None:
            self._commands.unregister(command.name for command in entry.loaded.commands)
        entry.loaded = None
        self._loaded.pop(entry.name, None)
        self._suspended.pop(entry.name, None)
        self._from_settings.pop(entry.name, None)
        self._guards.pop(entry.name, None)

    async def close(self, reason: SessionEndReason) -> None:
        """Unload every plugin, last loaded first."""
        for name in reversed(list(self._loaded)):
            await self.unload(name, reason=reason)

    async def fire(self, event: HostEvent) -> None:
        """Dispatch to every loaded plugin. `turn_start` fails closed; the rest report and continue."""
        for name, loaded in list(self._loaded.items()):
            try:
                await loaded.dispatch(event)
            except Exception as exc:
                if isinstance(event, TurnStart):
                    raise PluginError(name, exc) from exc
                self._console.print(str(PluginError(name, exc)), style=theme.color(theme.ERROR), markup=False)

    async def enable(self, name: str) -> None:
        """Remember the plugin as enabled and load it now."""
        entry = self._entry(name)
        _check_not_included(entry)
        _requested('enable', name)
        declaration = entry.declaration.model_copy(update={'enabled': True})
        self._store.save_plugin(declaration, requires=self._requirements(entry))
        await self._load_enabled(entry)

    async def _load_enabled(self, entry: PluginEntry[DepsT]) -> None:
        """Load an enabled declaration and remember requirements learned from its plugin."""
        await self.load(entry.name)
        # `load` refreshed the entries; only a loaded plugin can say what its settings need.
        loaded = self._entries[entry.name].loaded
        host = loaded.plugin.host if loaded is not None else None
        if host is not None and host.requirements:
            self._store.save_plugin(self._saved(entry), requires=host.requirements)

    async def disable(self, name: str) -> None:
        """Unload the plugin now and remember it as disabled."""
        entry = self._entry(name)
        _requested('disable', name)
        requires = self._requirements(entry)
        await self.unload(name)
        self._store.save_plugin(entry.declaration.model_copy(update={'enabled': False}), requires=requires)
        await self._load_released(entry)

    async def remove(self, name: str) -> str:
        """Unload the plugin and forget its saved declaration; a shipped declaration comes back as declared."""
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
                self._store.save_plugin(entry.declaration.model_copy(update={'enabled': False}), requires=requires)
                await self._load_released(entry)
                return f'Disabled {name}. Delete {entry.path} to remove the plugin itself.'
        self._store.delete_plugin(name)
        shipped = self._project.get(name) or self._builtin.get(name)
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
        """Unload, re-import the module, and load again."""
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
        entry = self._entry(name)
        loaded = entry.loaded
        _requested('configure', name)
        _check_not_included(entry)
        if loaded is None:
            raise ValueError(f'Plugin {name} is not loaded; enable it before configuring.')
        if not loaded.plugin.has_configure:
            raise ValueError(f'Plugin {name} has no settings menu; replace its declaration with /plugins add.')
        before = self._entry(name).declaration.settings
        try:
            return await loaded.plugin.configure()
        finally:
            # Also when the menu fails after saving, so the running plugin matches what is saved.
            if self._entry(name).declaration.settings != before:
                await self._load_again(name)

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
        self._staged[name] = declaration
        try:
            # Like `enable`, refuse an included plugin before unloading anything, and save nothing.
            _check_not_included(self._entry(name))
        except ValueError:
            del self._staged[name]
            raise
        if existing is not None:
            await self.unload(name)
        try:
            await self.load(name)
        except PluginSettingsError:
            del self._staged[name]
            if loaded:
                await self.load(name)
            raise
        except BaseException:
            requires = self._requirements(self._entry(name))
            self._store.save_plugin(self._staged.pop(name), requires=requires)
            raise
        requires = self._requirements(self._entry(name))
        self._store.save_plugin(self._staged.pop(name), requires=requires)
        _requested('add', name)
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
                self._store.save_plugin(declaration, overwrite=False)
            _requested('add', declaration.id)
            await self._load_enabled(self._entry(declaration.id))
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

    def _import(self, entry: PluginEntry[DepsT], *, fresh: bool) -> ModuleType:
        if entry.path is not None:
            if not entry.path.is_absolute():
                # Saved approvals are shared by every repository, so a relative path would run whatever
                # file sits at it under the launch directory. Project files are anchored before approval.
                raise ValueError(
                    f'{entry.path} is relative, so it would load from whichever directory CLAI starts in. '
                    f'Use /plugins remove {entry.name} and approve it again, or give it an absolute path.'
                )
            return import_file(entry.name, entry.path)
        module_name = entry.declaration.factory.partition(':')[0]
        module = importlib.import_module(module_name)
        return importlib.reload(module) if fresh else module


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


def _requested(action: str, name: str) -> None:
    """UI telemetry for an action on a plugin known to exist (never a mistyped name), before disabling `observability`."""
    telemetry.record('plugin {plugin} {action}', plugin=name, action=action)


def _module_absent(entry: PluginEntry[DepsT], error: BaseException) -> bool:
    """Whether the plugin's own module (or a parent package) is missing, not one of its imports."""
    if entry.path is not None or not isinstance(error, ModuleNotFoundError) or error.name is None:
        return False
    module_name = entry.declaration.factory.partition(':')[0]
    return module_name == error.name or module_name.startswith(f'{error.name}.')


def _same_plugin(declaration: PluginSettings, shipped: PluginSettings | None) -> bool:
    """Whether `declaration` is `shipped` itself, or the store's enabled or disabled copy of it."""
    if shipped is None:
        return False
    return declaration.model_copy(update={'enabled': True}) == shipped.model_copy(update={'enabled': True})


async def _end_failed_session(plugin: Plugin[BaseModel, DepsT]) -> BaseException | None:
    """Return the handler's failure rather than raising it."""
    try:
        with fail_after(5):
            await plugin.on_session_end(SessionEnd(reason='error'))
    except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- reported by the caller.
        return exc
    return None
