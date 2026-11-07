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
    top_level = module_name.partition('.')[0]
    imported = sys.modules.get(top_level)
    # `find_spec` raises for an imported module without a spec, such as a script's `__main__`.
    spec = importlib.util.find_spec(top_level) if imported is None else imported.__spec__
    # A namespace package, which has no `origin`, can have a portion in the launch directory.
    if spec is None or spec.origin is None:
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.append(cwd)
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
