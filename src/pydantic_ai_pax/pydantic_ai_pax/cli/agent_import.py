"""Resolve `--agent MODULE:ATTR` to a Pydantic AI agent instance and the deps it runs with."""

import importlib
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from types import NoneType
from typing import Any, get_args

from pydantic_ai.agent import AbstractAgent


@dataclass(frozen=True, kw_only=True)
class ImportedAgent:
    """An agent named by `--agent`, with the deps its factory returned (`None` when it returned none)."""

    agent: AbstractAgent[object, object]
    deps: object = None


def import_agent(path: str) -> ImportedAgent:
    """Import `MODULE:ATTR` and return the agent it names, with its deps.

    `ATTR` is an agent instance, or a zero-argument callable returning an agent or an `(agent, deps)` tuple.
    An agent without deps runs with `deps=None`, so one whose `deps_type` needs deps is refused here,
    before any model request.

    The launch directory is appended to `sys.path`, after installed packages, only when nothing installed has the
    module's top-level name, so a module there resolves without installing it and an installed agent's later
    imports cannot resolve to it. A launch directory module whose top-level name is also installed resolves
    against the installed package instead, so it usually fails with "module not found".
    (`python -m pydantic_ai_pax` already puts it first, as for any script.)
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
        raise TypeError(
            f'--agent {path}: {attr!r} is a class; point at an Agent instance or a factory function instead'
        )
    if (agent := _as_agent(target)) is not None:
        return _without_deps(path, agent)
    if not callable(target):
        raise TypeError(
            f'--agent {path}: expected a Pydantic AI Agent instance or a factory function, got {type(target).__name__}'
        )
    result: object = target()
    if (agent := _as_agent(result)) is not None:
        return _without_deps(path, agent)
    if (imported := _agent_and_deps(result)) is not None:
        return imported
    raise TypeError(
        f'--agent {path}: expected {attr!r} to return an Agent or an (agent, deps) tuple, got {type(result).__name__}'
    )


def _as_agent(value: object) -> AbstractAgent[object, object] | None:
    """`value` as an agent, if it is one; its generic parameters are erased at runtime, and Pax only displays output."""
    return value if isinstance(value, AbstractAgent) else None  # pyright: ignore[reportUnknownVariableType]


def _agent_and_deps(value: object) -> ImportedAgent | None:
    """`value` as an `(agent, deps)` pair, if it is one."""
    if not isinstance(value, tuple):
        return None
    pair: tuple[object, ...] = value  # pyright: ignore[reportUnknownVariableType]
    if len(pair) != 2 or (agent := _as_agent(pair[0])) is None:
        return None
    return ImportedAgent(agent=agent, deps=pair[1])


def _without_deps(path: str, agent: AbstractAgent[object, object]) -> ImportedAgent:
    """Refuse an agent whose `deps_type` needs deps, since none were supplied; it would fail mid-run otherwise.

    A `deps_type` that admits `None`, such as `Deps | None`, runs with `deps=None`.
    """
    deps_type: object = agent.deps_type
    if deps_type not in (NoneType, object, Any) and NoneType not in get_args(deps_type):
        name = deps_type.__name__ if isinstance(deps_type, type) else str(deps_type)
        raise TypeError(
            f'--agent {path}: the agent needs deps of type {name}; point --agent at a '
            'zero-argument function that returns (agent, deps) instead'
        )
    return ImportedAgent(agent=agent)
