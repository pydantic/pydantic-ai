"""Load and validate runtime-authored capability code without leaking `Any`.

Authored code is imported dynamically, so the values it produces are untyped at
the import boundary. Every value crossing back into the harness is narrowed with
`isinstance`/`issubclass` before use, so nothing typed `Any` escapes this module.
"""

from __future__ import annotations

import hashlib
import sys
import threading
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import TYPE_CHECKING, TypeGuard

from pydantic_ai._instructions import normalize_instructions, validate_instruction_name
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import InstructionPart
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.toolsets import AbstractToolset

if TYPE_CHECKING:
    from pathlib import Path


class CapabilityValidationError(Exception):
    """Raised when authored code cannot be imported, has the wrong shape, or fails to construct."""


_LOAD_LOCK = threading.RLock()
"""Serializes loads and their getter checks across threads.

An authored module's classes resolve names through its `sys.modules` entry, so a concurrent load of the same
file would swap the namespace a half-built class resolves against. Reentrant, so authored code that loads
another authored file in the same thread does not deadlock.
"""


def _is_capability_subclass(obj: object) -> TypeGuard[type[AbstractCapability[object]]]:
    """Narrow an attribute of unknown type to a concrete `AbstractCapability` subclass."""
    return isinstance(obj, type) and issubclass(obj, AbstractCapability)


def _check_toolset_return(value: object) -> None:
    """Reject a `get_toolset` return that the agent cannot wrap.

    The agent accepts an `AbstractToolset`, a toolset function, or None.
    """
    if value is not None and not isinstance(value, AbstractToolset) and not callable(value):
        raise CapabilityValidationError(
            f'get_toolset() returned {type(value).__name__}, expected an AbstractToolset, a callable, or None'
        )


def _check_instruction_items(items: Iterable[object]) -> None:
    """Reject a `get_instructions` item the run cannot render.

    Each must be text, an `InstructionPart`, or a callable, and a part's `name` must be one the agent
    accepts, as checked when it builds the capability's instructions.
    """
    for item in items:
        if isinstance(item, InstructionPart):
            if item.name is not None:
                validate_instruction_name(item.name)
        elif not isinstance(item, str) and not callable(item):
            raise CapabilityValidationError(
                f'get_instructions() returned a {type(item).__name__} item, '
                f'expected a str, an InstructionPart, or a callable'
            )


def _check_native_tool_items(items: Iterable[object]) -> None:
    """Reject a `get_native_tools` item the agent cannot use: each must be an `AbstractNativeTool` or a callable."""
    for item in items:
        if not isinstance(item, AbstractNativeTool) and not callable(item):
            raise CapabilityValidationError(
                f'get_native_tools() returned a {type(item).__name__} item, expected an AbstractNativeTool or a callable'
            )


def _check_model_settings_return(value: object) -> None:
    """Reject a `get_model_settings` return that the runtime cannot merge.

    The runtime merges the return with `merge_model_settings`, which needs a
    `ModelSettings` mapping, a callable, or None. Typed `object` so an authored
    override that violates its declared signature is narrowed here, not trusted.
    """
    if value is not None and not callable(value) and not isinstance(value, Mapping):
        raise CapabilityValidationError(
            f'get_model_settings() returned {type(value).__name__}, '
            f'expected a ModelSettings mapping, a callable, or None'
        )


def _check_serialization_name(value: object) -> None:
    """Reject a `get_serialization_name` return that is neither a spec name nor None."""
    if value is not None and not isinstance(value, str):
        raise CapabilityValidationError(
            f'get_serialization_name() returned {type(value).__name__}, expected a str or None'
        )


def _module_name(path: Path) -> str:
    """The `sys.modules` key for `path`, derived from its resolved path so a reload replaces its entry."""
    resolved = str(path.resolve())
    return f'_authored_capability_{path.stem}_{hashlib.sha256(resolved.encode()).hexdigest()[:12]}'


def _load_module(path: Path, name: str) -> ModuleType:
    """Execute `path` as a fresh module registered under `name`.

    The source is compiled directly instead of through the import system, so no
    bytecode cache is read or written: re-authoring under the same name always
    runs the new source. `dont_inherit` keeps this module's `__future__` imports
    out of the authored code.

    The module is registered in `sys.modules` because `dataclasses` and Pydantic look
    up a class's module there to resolve string annotations (as under
    `from __future__ import annotations`), during the class body and later. The
    caller undoes the registration if the load fails.
    """
    module = ModuleType(name)
    module.__file__ = str(path.resolve())
    code = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module


def _find_capability_class(module: ModuleType) -> type[AbstractCapability[object]]:
    """Return the single `AbstractCapability` subclass defined in `module`.

    Only classes whose `__module__` is the authored module count, so capabilities
    imported by the authored code (the base class itself, helpers from harness)
    are ignored. Anything other than exactly one is a validation error.
    """
    found: list[type[AbstractCapability[object]]] = []
    for attr_name in dir(module):
        obj: object = getattr(module, attr_name)
        if _is_capability_subclass(obj) and obj.__module__ == module.__name__:
            found.append(obj)
    if not found:
        raise CapabilityValidationError('no `AbstractCapability` subclass found in the authored code')
    if len(found) > 1:
        names = ', '.join(sorted(cls.__name__ for cls in found))
        raise CapabilityValidationError(
            f'expected exactly one `AbstractCapability` subclass, found {len(found)}: {names}'
        )
    return found[0]


def load_capability_instance(path: Path) -> AbstractCapability[object]:
    """Import, construct, and check the authored capability, returning the instance.

    Runs the static getters (`get_instructions`, `get_toolset`, `get_native_tools`,
    `get_model_settings`, `get_serialization_name`) and checks their returns, down to
    each instruction and native tool, against the types the agent accepts, so a wrong
    return type fails here instead of crashing the next `agent.run`. The async
    lifecycle hooks need a live `RunContext` and are not invoked. Anything else the
    authored code raises, `SystemExit` and other `BaseException` subclasses included,
    is reported as a validation error rather than ending the process; only
    `KeyboardInterrupt` propagates. Loads from different threads run one at a time.

    Returns an `AbstractCapability[object]`; `AgentDepsT` is contravariant, so the
    instance is accepted by any agent's `capabilities=` parameter.

    The module stays in `sys.modules` under a key derived from `path`, so loading
    the file again replaces its entry. A load that fails at any step, the checks
    included, puts back the entry it replaced, or removes its own, as the import
    system does for a failed import.
    """
    name = _module_name(path)
    with _LOAD_LOCK:
        # The entry itself, not `.get()`, so an entry holding `None` is put back too.
        previous = {name: sys.modules[name]} if name in sys.modules else None
        try:
            module = _load_module(path, name)
            cls = _find_capability_class(module)
            instance = cls()
            _check_instruction_items(normalize_instructions(instance.get_instructions()))
            _check_toolset_return(instance.get_toolset())
            _check_native_tool_items(instance.get_native_tools())
            _check_model_settings_return(instance.get_model_settings())
            _check_serialization_name(cls.get_serialization_name())
        except BaseException as exc:
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules.update(previous)
            if isinstance(exc, (CapabilityValidationError, KeyboardInterrupt)):
                raise
            raise CapabilityValidationError(f'{type(exc).__name__}: {exc}') from exc
    return instance


def validate_capability_file(path: Path) -> str:
    """Run the checks of `load_capability_instance` and return the capability class name."""
    return type(load_capability_instance(path)).__name__
