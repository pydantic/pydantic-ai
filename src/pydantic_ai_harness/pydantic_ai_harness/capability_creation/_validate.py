"""Load and validate runtime-authored capability code without leaking `Any`.

Authored code is imported dynamically, so the values it produces are untyped at
the import boundary. Every value crossing back into the harness is narrowed with
`isinstance`/`issubclass` before use, so nothing typed `Any` escapes this module.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING, TypeGuard

from pydantic_ai._instructions import normalize_instructions
from pydantic_ai.capabilities import AbstractCapability

if TYPE_CHECKING:
    from pathlib import Path


class CapabilityValidationError(Exception):
    """Raised when authored code cannot be imported, has the wrong shape, or fails to construct."""


def _is_capability_subclass(obj: object) -> TypeGuard[type[AbstractCapability[object]]]:
    """Narrow an attribute of unknown type to a concrete `AbstractCapability` subclass."""
    return isinstance(obj, type) and issubclass(obj, AbstractCapability)


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
    """Import `path` as a fresh module, not registered in `sys.modules`.

    A fresh module object each call, plus suppressing the on-disk bytecode cache,
    means re-authoring under the same name always re-executes the new source.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise CapabilityValidationError(f'cannot create an import spec for {path.name}')
    module = importlib.util.module_from_spec(spec)
    # Don't let `exec_module` cache a `.pyc`. Re-authoring under the same name
    # changes the `.py`, but a stale cache would make the loader return the
    # previous bytecode for an unchanged file path, serving the old source.
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
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


def validate_capability_file(path: Path) -> str:
    """Import, construct, and exercise the static getters of the authored capability.

    Calls only the side-effect-free static getters (`get_instructions`,
    `get_toolset`, `get_native_tools`, `get_model_settings`,
    `get_serialization_name`). The async lifecycle hooks need a live `RunContext`
    and are not invoked here. Returns the capability class name on success.
    """
    try:
        module = _load_module(path)
        cls = _find_capability_class(module)
        instance = cls()
        # Run the getter returns through the same coercion/shape-checks the
        # runtime uses, so a wrong return type fails validation here instead of
        # crashing the next `agent.run`. `get_native_tools`/`get_toolset` returns
        # are exercised by `list(...)`; the instructions/model-settings returns
        # would otherwise slip through untyped.
        normalize_instructions(instance.get_instructions())
        instance.get_toolset()
        list(instance.get_native_tools())
        _check_model_settings_return(instance.get_model_settings())
        cls.get_serialization_name()
    except CapabilityValidationError:
        raise
    except Exception as exc:
        raise CapabilityValidationError(f'{type(exc).__name__}: {exc}') from exc
    return cls.__name__


def load_capability_instance(path: Path) -> AbstractCapability[object]:
    """Import the authored module and construct its capability instance.

    Returns an `AbstractCapability[object]`; `AgentDepsT` is contravariant, so the
    instance is accepted by any agent's `capabilities=` parameter.
    """
    try:
        module = _load_module(path)
        cls = _find_capability_class(module)
        return cls()
    except CapabilityValidationError:
        raise
    except Exception as exc:
        raise CapabilityValidationError(f'{type(exc).__name__}: {exc}') from exc
