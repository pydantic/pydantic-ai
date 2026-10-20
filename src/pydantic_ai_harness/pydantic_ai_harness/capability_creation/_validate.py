"""Load and validate runtime-authored capability code without leaking `Any`.

Authored code is imported dynamically, so the values it produces are untyped at
the import boundary. Every value crossing back into the harness is narrowed with
`isinstance`/`issubclass` before use, so nothing typed `Any` escapes this module.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING, TypeGuard

from pydantic_ai._instructions import normalize_instructions
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset

if TYPE_CHECKING:
    from pathlib import Path


class CapabilityValidationError(Exception):
    """Raised when authored code cannot be imported, has the wrong shape, or fails to construct."""


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


def _load_module(path: Path) -> ModuleType:
    """Execute `path` as a fresh module.

    The source is compiled directly instead of through the import system, so no
    bytecode cache is read or written: re-authoring under the same name always
    runs the new source. `dont_inherit` keeps this module's `__future__` imports
    out of the authored code.

    The module is in `sys.modules` only while its body runs, under a name unique to
    this load, because `dataclasses` and Pydantic look up a class's module there to
    resolve string annotations (as under `from __future__ import annotations`).
    """
    module = ModuleType(path.stem)
    module.__name__ = f'_authored_capability_{path.stem}_{id(module):x}'
    module.__file__ = str(path)
    code = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
    sys.modules[module.__name__] = module
    try:
        exec(code, module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
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
    `get_model_settings`, `get_serialization_name`) and checks their returns with
    the same coercions the agent applies, so a wrong return type fails here instead
    of crashing the next `agent.run`. The async lifecycle hooks need a live
    `RunContext` and are not invoked. `SystemExit` raised by the authored code is
    reported as a validation error rather than ending the process.

    Returns an `AbstractCapability[object]`; `AgentDepsT` is contravariant, so the
    instance is accepted by any agent's `capabilities=` parameter.
    """
    try:
        module = _load_module(path)
        cls = _find_capability_class(module)
        instance = cls()
        normalize_instructions(instance.get_instructions())
        _check_toolset_return(instance.get_toolset())
        list(instance.get_native_tools())
        _check_model_settings_return(instance.get_model_settings())
        cls.get_serialization_name()
    except CapabilityValidationError:
        raise
    except (Exception, SystemExit) as exc:
        raise CapabilityValidationError(f'{type(exc).__name__}: {exc}') from exc
    return instance


def validate_capability_file(path: Path) -> str:
    """Run the checks of `load_capability_instance` and return the capability class name."""
    return type(load_capability_instance(path)).__name__
