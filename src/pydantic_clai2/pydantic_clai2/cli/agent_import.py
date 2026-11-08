"""Resolve `--agent MODULE:ATTR` to a Pydantic AI agent instance."""

import importlib
import sys
from importlib.util import find_spec
from pathlib import Path

from pydantic_ai.agent import AbstractAgent


def import_agent(path: str) -> AbstractAgent[None, object]:
    """Import `MODULE:ATTR` and return the agent instance it names.

    The launch directory is appended to `sys.path`, after installed packages, only when nothing installed has the
    module's top-level name, so a module there resolves without installing it and an installed agent's later
    imports cannot resolve to it. A launch directory module whose top-level name is also installed fails with
    "module not found". (`python -m pydantic_clai2` already puts it first, as for any script.)
    The agent runs with `deps=None`.
    """
    module_name, _, attr = path.partition(':')
    if not module_name or not attr:
        raise ValueError(f'--agent expects MODULE:ATTR, for example pydantic_ai.main:my_cool_agent; got {path!r}')
    top_level = module_name.partition('.')[0]
    cwd = str(Path.cwd())
    if top_level not in sys.modules and find_spec(top_level) is None and cwd not in sys.path:
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
