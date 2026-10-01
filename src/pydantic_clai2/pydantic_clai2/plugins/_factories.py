"""Turn a plugin declaration into a `Plugin`: import its module, then find and build what it names."""

import hashlib
import importlib.util
import inspect
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel

from pydantic_ai.capabilities import AbstractCapability, AgentCapability
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.plugins import DepsT, NoSettings, Plugin, PluginHost

_FOLDER_PACKAGE = 'pydantic_clai2_plugins'


def import_file(name: str, path: Path) -> ModuleType:
    """Import a drop-in plugin file or package afresh, under a package private to its folder."""
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


class _CapabilityPlugin(Plugin[NoSettings, DepsT]):
    """A capability class declared directly, `package.module:Capability`, with its settings as keyword arguments."""

    def __init__(self, host: PluginHost[DepsT], capability: AbstractCapability[DepsT]) -> None:
        super().__init__(host, NoSettings())
        self.capability = capability

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        return (self.capability,)


def build(module: ModuleType, declaration: PluginSettings, host: PluginHost[DepsT]) -> Plugin[BaseModel, DepsT]:
    """Build the declared `Plugin` class, or wrap a capability class built from the declaration's settings."""
    attr = declaration.factory.partition(':')[2]
    target: object = getattr(module, attr, None) if attr else _plugin_class(module, declaration.factory)
    if isinstance(target, type) and issubclass(target, Plugin):
        plugin_type: type[Plugin[BaseModel, DepsT]] = target  # pyright: ignore[reportUnknownVariableType]
        return plugin_type.from_host(host)
    if isinstance(target, type) and issubclass(target, AbstractCapability):
        capability: AbstractCapability[DepsT] = target(**declaration.settings)  # pyright: ignore[reportUnknownVariableType]
        return _CapabilityPlugin(host, capability)
    legacy: object = getattr(module, attr or 'activate', None)
    if inspect.isfunction(legacy):
        raise TypeError(
            f'{declaration.factory} is an `activate(host)` function; plugins are now `Plugin` subclasses.'
            ' See "What a plugin declares" in PLUGINS.md.'
        )
    raise TypeError(f'{declaration.factory} is neither a `Plugin` nor a capability class')


def _plugin_class(module: ModuleType, factory: str) -> object:
    """The one public `Plugin` subclass `module` defines, so a declaration can name just the module."""
    found = [name for name, value in vars(module).items() if _declares_plugin(module, name, value)]
    if len(found) > 1:
        raise TypeError(f'{factory} defines several plugins ({", ".join(sorted(found))}); name one as {factory}:CLASS.')
    return getattr(module, found[0]) if found else None


def _declares_plugin(module: ModuleType, name: str, value: object) -> bool:
    return (
        not name.startswith('_')
        and isinstance(value, type)
        and issubclass(value, Plugin)
        and value.__module__ == module.__name__
    )


def settings_capability(plugin: Plugin[BaseModel, DepsT]) -> AbstractCapability[DepsT] | None:
    """The capability built directly from a declaration settings, not a plugin contribution."""
    return plugin.capability if isinstance(plugin, _CapabilityPlugin) else None
