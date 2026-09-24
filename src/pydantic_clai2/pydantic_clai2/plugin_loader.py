"""Load, unload, and reload plugins between turns. Discarding a host unloads its plugin."""

import asyncio
import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Generic

from anyio import CancelScope, fail_after
from anyio.lowlevel import checkpoint
from pydantic_ai import AgentStreamEvent
from pydantic_ai.capabilities import AbstractCapability, AgentCapability
from rich.console import Console

from . import theme
from .commands import Commands, plugins_command
from .config import PluginSettings
from .plugins import (
    Conversation,
    DepsT,
    FullScreen,
    HostEvent,
    PluginHost,
    Renderer,
    SessionEnd,
    SessionEndReason,
    SessionStart,
    TurnStart,
    bare_screen,
)
from .settings_store import SettingsStore
from .status import Status, StatusSegment

_FOLDER_PACKAGE = 'pydantic_clai2_plugins'


class PluginError(Exception):
    """A plugin failed while loading or while handling an event."""

    def __init__(self, plugin: str, error: BaseException) -> None:
        """Name the plugin so the user knows which one to fix."""
        super().__init__(f'Plugin {plugin!r}: {type(error).__name__}: {error}')
        self.plugin = plugin
        self.error = error


@dataclass(kw_only=True)
class PluginEntry(Generic[DepsT]):
    """One `/plugins` row: the saved declaration plus what is loaded right now."""

    declaration: PluginSettings
    path: Path | None
    builtin: bool = False
    project: bool = False
    host: PluginHost[DepsT] | None = None
    error: str | None = None

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
        if not self.declaration.enabled:
            return 'disabled'
        if self.host is not None:
            return 'enabled, loaded'
        return f'enabled, failed: {self.error}' if self.error else 'enabled, not loaded'


class PluginLoader(Generic[DepsT]):
    """Own every plugin's host; the shell asks it for capabilities and renderers each turn."""

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
        status: Status | None = None,
        full_screen: FullScreen = bare_screen,
    ) -> None:
        """`builtin` ships with CLAI, `project` comes from `.clai/settings.json`; the store overrides both.

        `full_screen` is handed to every host; the shell binds it to the live renderer per prompt.
        `conversation` and `status` are handed to every host; see `PluginHost` for the defaults.
        """
        self._store = store
        self._console = console
        self._commands = commands
        self._session_start = session_start
        self._full_screen = full_screen
        self._conversation = conversation
        self._status = status
        self._builtin = {declaration.id: declaration for declaration in builtin}
        self._project = {declaration.id: declaration for declaration in project}
        self._entries: dict[str, PluginEntry[DepsT]] = {}
        self._loaded: dict[str, PluginHost[DepsT]] = {}

    @property
    def plugins_dir(self) -> Path:
        """Folder scanned for drop-in plugin files."""
        return self._store.plugins_dir

    def entries(self) -> list[PluginEntry[DepsT]]:
        """Saved declarations plus drop-in files, keeping the loaded state of each."""
        folder = self._discover()
        declared = {declaration.id: declaration for declaration in self._store.plugins()}
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
                host=previous.host if previous else None,
                error=previous.error if previous else None,
            )
        for name, previous in self._entries.items():
            if name not in refreshed and previous.host is not None:
                refreshed[name] = previous
        self._entries = refreshed
        return list(refreshed.values())

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
        """Bound on every run, in load order."""
        return [capability for host in self._loaded.values() for capability in host.capabilities]

    def renderers(self) -> list[Renderer[AgentStreamEvent]]:
        """Consulted before the default display, in load order."""
        return [renderer for host in self._loaded.values() for renderer in host.renderers]

    def status_segments(self) -> list[StatusSegment]:
        """Appended to the status row, in load order."""
        return [segment for host in self._loaded.values() for segment in host.status_segments]

    async def load_all(self, *, fresh: bool = False) -> None:
        """Load enabled plugins, re-importing after a shell reload so host event types match."""
        for entry in self._registration_order():
            if entry.declaration.enabled and entry.host is None:
                try:
                    await self.load(entry.name, fresh=fresh)
                except PluginError as exc:
                    self._console.print(str(exc), style=theme.color(theme.ERROR), markup=False)

    async def load(self, name: str, *, fresh: bool = False) -> None:
        """Import, activate, and fire `session_start`. A failure leaves nothing registered."""
        entry = self._entry(name)
        if entry.host is not None:
            return
        host = PluginHost[DepsT](
            name=name,
            console=self._console,
            settings=entry.declaration.settings,
            full_screen=self._full_screen,
            conversation=self._conversation,
            status=self._status,
        )
        try:
            module = self._import(entry, fresh=fresh)
            _activate(module, entry.declaration, host)
            self._commands.register_many(host.commands)
            entry.host = host
            self._loaded[name] = host
            await _dispatch(host, self._session_start())
        except asyncio.CancelledError:
            await self._failed_load(entry, host)
            raise
        except Exception as exc:
            await self._failed_load(entry, host)
            entry.error = f'{type(exc).__name__}: {exc}'
            raise PluginError(name, exc) from exc
        entry.error = None

    async def _failed_load(self, entry: PluginEntry[DepsT], host: PluginHost[DepsT]) -> None:
        try:
            for handler in host.handlers:
                # Its own task keeps the handler's `CancelledError` apart from ours: the former is
                # reported, the latter propagates. The shield defers scope cancellation until cleanup
                # finishes; the `checkpoint()` below delivers it.
                cleanup = asyncio.create_task(_end_failed_session(handler))
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
        """Fire `session_end`, then drop everything the plugin registered."""
        entry = self._entry(name)
        if entry.host is None:
            return
        try:
            await _dispatch(entry.host, SessionEnd(reason=reason))
        except Exception as exc:  # noqa: BLE001 -- unloading must finish even if the plugin misbehaves.
            self._console.print(str(PluginError(name, exc)), style=theme.color(theme.ERROR), markup=False)
        finally:
            self._drop(entry)

    def _drop(self, entry: PluginEntry[DepsT]) -> None:
        if entry.host is not None:
            self._commands.unregister(command.name for command in entry.host.commands)
        entry.host = None
        self._loaded.pop(entry.name, None)

    async def close(self, reason: SessionEndReason) -> None:
        """Unload every plugin, last loaded first."""
        for name in reversed(list(self._loaded)):
            await self.unload(name, reason=reason)

    async def fire(self, event: HostEvent) -> None:
        """Dispatch to every loaded plugin. `turn_start` fails closed; the rest report and continue."""
        for name, host in list(self._loaded.items()):
            try:
                await _dispatch(host, event)
            except Exception as exc:
                if isinstance(event, TurnStart):
                    raise PluginError(name, exc) from exc
                self._console.print(str(PluginError(name, exc)), style=theme.color(theme.ERROR), markup=False)

    async def enable(self, name: str) -> None:
        """Remember the plugin as enabled and load it now."""
        entry = self._entry(name)
        self._store.save_plugin(entry.declaration.model_copy(update={'enabled': True}))
        await self.load(name)

    async def disable(self, name: str) -> None:
        """Unload the plugin now and remember it as disabled."""
        entry = self._entry(name)
        await self.unload(name)
        self._store.save_plugin(entry.declaration.model_copy(update={'enabled': False}))

    async def remove(self, name: str) -> str:
        """Unload the plugin and forget its saved declaration; a shipped declaration comes back as declared."""
        entry = self._entry(name)
        await self.unload(name)
        if entry.path is not None and not entry.shipped:
            self._store.save_plugin(entry.declaration.model_copy(update={'enabled': False}))
            return f'Disabled {name}. Delete {entry.path} to remove the plugin itself.'
        self._store.delete_plugin(name)
        shipped = self._project.get(name) or self._builtin.get(name)
        if shipped is None:
            return f'Removed {name}.'
        if shipped.enabled:
            await self.load(name)
        origin = 'declared by the project' if name in self._project else 'built in'
        return f'{name} is {origin}; restored its defaults. Use /plugins disable {name} to turn it off.'

    async def reload(self, name: str) -> None:
        """Unload, re-import the module, and load again."""
        if not self._entry(name).declaration.enabled:
            raise ValueError(f'Plugin {name} is disabled; enable it before reloading.')
        await self.unload(name)
        await self.load(name, fresh=True)

    async def command(self, args: list[str]) -> str:
        """Back `/plugins` with arguments; changes apply now and are saved."""
        if not args or args == ['list']:
            return (
                '\n'.join(f'{entry.name}: {entry.source} ({entry.state})' for entry in self.entries()) or 'No plugins.'
            )
        action, *rest = args
        if action == 'add':
            existing = next((entry for entry in self.entries() if rest and entry.name == rest[0]), None)
            if existing is not None and not existing.shipped:
                raise ValueError(f'Plugin {rest[0]} already exists; remove its declaration before replacing it.')
            if existing is not None:
                await self.unload(rest[0])
            plugins_command(self._store, args)
            await self.load(rest[0])
            if existing is None:
                return f'Added and loaded {rest[0]}.'
            return f'Replaced {"project" if existing.project else "built-in"} {rest[0]}.'
        if len(rest) != 1:
            raise ValueError(
                'Usage: /plugins [list|add ID MODULE[:ATTR] [JSON]|enable ID|disable ID|remove ID|reload ID]'
            )
        name = rest[0]
        if action == 'remove':
            return await self.remove(name)
        actions = {
            'enable': (self.enable, 'Enabled'),
            'disable': (self.disable, 'Disabled'),
            'reload': (self.reload, 'Reloaded'),
        }
        if action not in actions:
            raise ValueError(f'Unknown plugins action: {action}')
        run, past = actions[action]
        await run(name)
        return f'{past} {name}.'

    def _import(self, entry: PluginEntry[DepsT], *, fresh: bool) -> ModuleType:
        if entry.path is not None:
            return _import_file(entry.name, entry.path)
        module_name = entry.declaration.factory.partition(':')[0]
        module = importlib.import_module(module_name)
        return importlib.reload(module) if fresh else module


def _same_plugin(declaration: PluginSettings, shipped: PluginSettings | None) -> bool:
    """Whether `declaration` is `shipped` itself, or the store's enabled or disabled copy of it."""
    if shipped is None:
        return False
    return declaration.model_copy(update={'enabled': True}) == shipped.model_copy(update={'enabled': True})


def _import_file(name: str, path: Path) -> ModuleType:
    root = path.parent.parent if path.name == '__init__.py' else path.parent
    namespace = f'{_FOLDER_PACKAGE}_{hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]}'
    qualified = f'{namespace}.{name}'
    if namespace not in sys.modules:
        package = ModuleType(namespace)
        package.__path__ = [str(root)]
        sys.modules[namespace] = package
    for cached in list(sys.modules):
        if cached == qualified or cached.startswith(qualified + '.'):
            del sys.modules[cached]
    search = [str(path.parent)] if path.name == '__init__.py' else None
    spec = importlib.util.spec_from_file_location(qualified, path, submodule_search_locations=search)
    if spec is None or spec.loader is None:  # pragma: no cover -- importlib always builds a loader for a .py path.
        raise ImportError(f'Cannot load plugin from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    try:
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)  # Explicitly trusted plugin source.
    except BaseException:
        sys.modules.pop(qualified, None)
        raise
    return module


def _activate(module: ModuleType, declaration: PluginSettings, host: PluginHost[DepsT]) -> None:
    attr = declaration.factory.partition(':')[2] or 'activate'
    target: object = getattr(module, attr, None)
    if isinstance(target, type):
        if not issubclass(target, AbstractCapability):
            raise TypeError(f'{declaration.factory} is not a capability class')
        capability: AbstractCapability[DepsT] = target(**declaration.settings)  # pyright: ignore[reportUnknownVariableType]
        host.add(capability)
    elif callable(target):
        target(host)
    else:
        raise TypeError(f'{declaration.factory} has no callable {attr!r}')


async def _dispatch(host: PluginHost[DepsT], event: HostEvent) -> None:
    for handler in host.handlers:
        await handler(event)


async def _end_failed_session(handler: Callable[[HostEvent], Awaitable[None]]) -> BaseException | None:
    """Return the handler's failure rather than raising it: 3.10 tasks drop a `CancelledError`'s message."""
    try:
        with fail_after(5):
            await handler(SessionEnd(reason='error'))
    except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- reported by the caller.
        return exc
    return None
