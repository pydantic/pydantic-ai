"""Resolve `--agent MODULE:ATTR` to a Pydantic AI agent instance."""

import importlib
import importlib.util
import sys
from pathlib import Path

from pydantic_ai.agent import AbstractAgent


def import_agent(path: str) -> AbstractAgent[None, object]:
    """Import `MODULE:ATTR` and return the agent instance it names.

    When the module is not importable otherwise, the launch directory is appended to `sys.path`, so a module
    there resolves without installing it and installed modules keep precedence. An importable module leaves
    `sys.path` as it was, so a launch directory module cannot replace a package the agent imports later.
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
        if _reports_missing(exc, module_name):
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
    # agent, so it counts as missing. An imported module is skipped, since `find_spec` raises for one without a
    # spec, such as a script's `__main__`.
    if module_name in sys.modules:
        return
    try:
        # Finding a submodule imports its parent packages, as importing it would.
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError as exc:
        if not _reports_missing(exc, module_name):
            raise
        spec = None
    cwd = str(Path.cwd())
    if (spec is None or spec.origin is None) and cwd not in sys.path:
        sys.path.append(cwd)


def _reports_missing(exc: ModuleNotFoundError, module_name: str) -> bool:
    """Whether `exc` reports `module_name` or a parent package of it missing, rather than a module it imports."""
    return exc.name is not None and (module_name == exc.name or module_name.startswith(f'{exc.name}.'))
