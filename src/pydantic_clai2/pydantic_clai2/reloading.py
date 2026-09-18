"""Reload CLAI's modules before rebuilding the shell, without reloading its dependencies."""

import importlib
import inspect
import sys
from collections.abc import Callable
from graphlib import TopologicalSorter
from pathlib import Path
from types import ModuleType
from typing import TypeVar

T = TypeVar('T')


def reload_clai(build: Callable[[], T]) -> T:
    """Reload dependencies before their importers; restore module bindings if reload or rebuild fails."""
    modules = {
        name: module
        for name, module in sys.modules.copy().items()
        if name == 'pydantic_clai2' or name.startswith('pydantic_clai2.')
    }
    snapshots: dict[ModuleType, dict[str, object]] = {module: vars(module).copy() for module in modules.values()}
    # ponytail: current bindings define reload order; restart after import-graph changes.
    dependencies = {
        module: {
            dependency
            for value in namespace.values()
            if (dependency := inspect.getmodule(value)) in snapshots and dependency is not module
        }
        for module, namespace in snapshots.items()
    }
    ordered = tuple(TopologicalSorter(dependencies).static_order())
    importlib.invalidate_caches()
    try:
        for module in ordered:
            # Same-size edits within one timestamp tick must not reuse bytecode.
            Path(module.__cached__).unlink(missing_ok=True)
            importlib.reload(module)
        return build()
    except BaseException:
        for module, namespace in snapshots.items():
            vars(module).clear()
            vars(module).update(namespace)
        for name in sys.modules.copy():
            if name.startswith('pydantic_clai2.') and name not in modules:
                del sys.modules[name]
        raise
