"""Resolve `--agent MODULE:ATTR` to a Pydantic AI agent instance."""

import importlib
import sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec
from pathlib import Path

from pydantic_ai.agent import AbstractAgent


def import_agent(path: str) -> AbstractAgent[None, object]:
    """Import `MODULE:ATTR` and return the agent instance it names.

    When the module is not importable otherwise, the launch directory is appended to `sys.path`, so a module
    there resolves without installing it. Installed modules keep precedence: an installed module or regular
    package with the same top-level name wins over the launch directory. An importable module leaves `sys.path`
    as it was, so a launch directory module cannot replace a package the agent imports later. The module is found
    without importing anything, so one in a legacy `pkgutil`-style namespace package may count as not importable.
    (`python -m pydantic_clai2` already puts the launch directory first, as for any script.)
    The agent runs with `deps=None`.
    """
    module_name, _, attr = path.partition(':')
    if not module_name or not attr:
        raise ValueError(f'--agent expects MODULE:ATTR, for example pydantic_ai.main:my_cool_agent; got {path!r}')
    _append_launch_directory_unless_importable(module_name)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and (module_name == exc.name or module_name.startswith(f'{exc.name}.')):
            raise ImportError(f'--agent {path}: module {module_name!r} not found') from exc
        raise
    if not hasattr(module, attr):
        raise AttributeError(f'--agent {path}: module {module_name!r} has no attribute {attr!r}')
    target: object = getattr(module, attr)
    if isinstance(target, type):
        raise TypeError(f'--agent {path}: {attr!r} is a class; point at an Agent instance instead')
    if not isinstance(target, AbstractAgent):
        raise TypeError(f'--agent {path}: expected a Pydantic AI Agent instance, got {type(target).__name__}')
    # Generic parameters are erased at runtime; `chat` passes `deps=None` and only displays the output.
    return target  # pyright: ignore[reportUnknownVariableType]


def _append_launch_directory_unless_importable(module_name: str) -> None:
    # Only the whole module name tells: a regular package anywhere on `sys.path` takes precedence over namespace
    # package portions, so appending the launch directory for a target in an installed namespace package would
    # let a package there replace it. A target that is itself only a namespace package (no `origin`) holds no
    # agent, so it counts as missing.
    cwd = str(Path.cwd())
    if module_name in sys.modules or cwd in sys.path:
        return
    spec = _find_spec_without_importing(module_name)
    if spec is None or spec.origin is None:
        sys.path.append(cwd)


def _find_spec_without_importing(module_name: str) -> ModuleSpec | None:
    # Importing a parent package would run its code and leave it in `sys.modules`, where it would decide what the
    # target is imported from once the launch directory is appended. So a parent that is not imported yet is not
    # imported here: the `sys.meta_path` finders search its spec's `submodule_search_locations`, where the import
    # would search its `__path__`. A legacy `pkgutil`-style namespace package adds its other portions to `__path__`
    # only when its `__init__.py` runs, so a target in one of those counts as missing, and the launch directory is
    # appended after installed packages.
    parent_name, _, _ = module_name.rpartition('.')
    search_path: Sequence[str] | None = None
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is None:
            parent_spec = _find_spec_without_importing(parent_name)
            search_path = None if parent_spec is None else parent_spec.submodule_search_locations
        elif hasattr(parent, '__path__'):
            search_path = parent.__path__
        if search_path is None:
            return None
    return _find_spec_on_meta_path(module_name, search_path)


def _find_spec_on_meta_path(module_name: str, search_path: Sequence[str] | None) -> ModuleSpec | None:
    parent_name, _, tail = module_name.rpartition('.')
    specs = (finder.find_spec(module_name, search_path) for finder in sys.meta_path)
    try:
        return next((spec for spec in specs if spec is not None), None)
    except KeyError as exc:
        if search_path is None or exc.args != (parent_name,):
            raise
    # `PathFinder` found only namespace package portions, and ties the namespace path it returns to the parent in
    # `sys.modules`. The portions are the directories named like the package in the parent's.
    spec = ModuleSpec(module_name, None, is_package=True)
    portions = (Path(location, tail) for location in search_path)
    spec.submodule_search_locations = [str(portion) for portion in portions if portion.is_dir()]
    return spec
