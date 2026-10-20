"""Load and validate runtime-authored capability code without leaking `Any`.

Authored code is imported dynamically, so the values it produces are untyped at
the import boundary. Every value crossing back into the harness is narrowed with
`isinstance`/`issubclass` before use, so nothing typed `Any` escapes this module.
"""

from __future__ import annotations

import hashlib
import sys
import uuid
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


def _module_name(path: Path) -> str:
    """The `sys.modules` key for the authored source at `path`, derived from its resolved path."""
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    return f'_authored_capability_{path.stem}_{digest}'


def _load_module(source: Path, file: Path, *, name: str) -> ModuleType:
    """Execute the source at `source` as a fresh module named `name`, saved at `file`.

    The source is compiled directly instead of through the import system, so no
    bytecode cache is read or written: re-authoring under the same name always
    runs the new source. `dont_inherit` keeps this module's `__future__` imports
    out of the authored code. `__file__` and error messages name `file`, which
    differs from `source` only while a staged copy is validated.

    The module is registered in `sys.modules` because `dataclasses` and Pydantic look
    up a class's module there to resolve string annotations (as under
    `from __future__ import annotations`), during the class body and later.
    """
    module = ModuleType(name)
    module.__file__ = str(file.resolve())
    code = compile(source.read_bytes(), str(file), 'exec', dont_inherit=True)
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


def _construct(source: Path, file: Path, *, module_name: str) -> AbstractCapability[object]:
    """Load `source` as the module saved at `file` and run the checks of `load_capability_instance`."""
    try:
        module = _load_module(source, file, name=module_name)
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

    The module's `sys.modules` key is derived from `path`, so reloading a file
    replaces its entry instead of adding one.
    """
    return _construct(path, path, module_name=_module_name(path))


def validate_capability_file(path: Path) -> str:
    """Run the checks of `load_capability_instance` and return the capability class name.

    Nothing from the module is kept, so it is removed from `sys.modules` again.
    """
    return validate_capability_source(path, file=path)


def validate_capability_source(source: Path, *, file: Path) -> str:
    """Validate the source at `source` as if it were saved at `file`, returning the class name.

    The store validates a staged copy before it replaces `file`; the authored code
    sees `file` as its `__file__`, as it will once loaded. The module gets a
    `sys.modules` key of its own, so validating a file leaves alone the entry of an
    instance already loaded from it, which its classes resolve names through.
    """
    module_name = f'_authored_capability_validation_{uuid.uuid4().hex}'
    try:
        return type(_construct(source, file, module_name=module_name)).__name__
    finally:
        sys.modules.pop(module_name, None)
