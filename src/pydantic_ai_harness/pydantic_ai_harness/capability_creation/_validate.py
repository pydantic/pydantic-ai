"""Load and validate runtime-authored capability code without leaking `Any`.

Authored code is imported dynamically, so the values it produces are untyped at
the import boundary. Every value crossing back into the harness is narrowed with
`isinstance`/`issubclass` before use, so nothing typed `Any` escapes this module.
"""

from __future__ import annotations

import hashlib
import sys
import threading
import uuid
from collections.abc import Generator, Iterable, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
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


def _module_name(path: Path, code: bytes) -> str:
    """The `sys.modules` key for `code` loaded from `path`, derived from its resolved path and the code."""
    digest = hashlib.sha256(str(path.resolve()).encode() + b'\0' + code).hexdigest()[:12]
    return f'_authored_capability_{path.stem}_{digest}'


_LOAD_LOCKS: dict[str, threading.RLock] = {}
_LOAD_LOCKS_GUARD = threading.Lock()


def _load_lock(name: str) -> threading.RLock:
    """The lock that loads registering the module `name` take turns on.

    Two loads of the same source would otherwise swap each other's module in
    `sys.modules` while it runs, and could leave a failed one behind. Loads of
    other sources do not wait for each other. The lock is reentrant, so authored
    code that loads its own source runs that load instead of waiting on itself.
    """
    with _LOAD_LOCKS_GUARD:
        return _LOAD_LOCKS.setdefault(name, threading.RLock())


@contextmanager
def _registered(module: ModuleType, *, keep: bool) -> Generator[None]:
    """Register `module` in `sys.modules` for the block, and keep it only when `keep` is set and the block succeeds.

    `dataclasses` and Pydantic look up a class's module there to resolve string
    annotations (as under `from __future__ import annotations`), during the class
    body and later. When the module is not kept, the entry it replaced is put
    back, unless something else has replaced the module meanwhile.
    """
    name = module.__name__
    # The entry itself, not `.get()`, so an entry holding `None` is put back too.
    previous = {name: sys.modules[name]} if name in sys.modules else None
    sys.modules[name] = module
    kept = False
    try:
        yield
        kept = keep
    finally:
        if not kept and sys.modules.get(name) is module:
            if previous is None:
                del sys.modules[name]
            else:
                sys.modules.update(previous)


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


def _construct(source: Path, file: Path, *, keep: bool) -> AbstractCapability[object]:
    """Load `source` as the module saved at `file` and run the checks of `load_capability_instance`.

    The source is compiled directly instead of through the import system, so no
    bytecode cache is read or written: re-authoring under the same name always
    runs the new source. `dont_inherit` keeps this module's `__future__` imports
    out of the authored code. `__file__` and error messages name `file`, which
    differs from `source` only while a staged copy is validated.

    With `keep`, a module that passes stays in `sys.modules` under a key derived
    from `file` and the source, so loading unchanged source again replaces its
    entry, while classes loaded from an earlier version keep resolving names
    through theirs, and loads of the same source take turns. Without it, the
    module gets a key of its own and is removed again. A module that fails is
    never kept.
    """
    try:
        code = source.read_bytes()
        name = _module_name(file, code) if keep else f'_authored_capability_validation_{uuid.uuid4().hex}'
        module = ModuleType(name)
        module.__file__ = str(file.resolve())
        compiled = compile(code, str(file), 'exec', dont_inherit=True)
        turn: AbstractContextManager[object] = _load_lock(name) if keep else nullcontext()
        with turn, _registered(module, keep=keep):
            exec(compiled, module.__dict__)
            cls = _find_capability_class(module)
            instance = cls()
            _check_instruction_items(normalize_instructions(instance.get_instructions()))
            _check_toolset_return(instance.get_toolset())
            _check_native_tool_items(instance.get_native_tools())
            _check_model_settings_return(instance.get_model_settings())
            _check_serialization_name(cls.get_serialization_name())
    except (CapabilityValidationError, KeyboardInterrupt):
        raise
    except BaseException as exc:
        raise CapabilityValidationError(f'{type(exc).__name__}: {exc}') from exc
    return instance


def load_capability_instance(path: Path) -> AbstractCapability[object]:
    """Import, construct, and check the authored capability, returning the instance.

    Runs the static getters (`get_instructions`, `get_toolset`, `get_native_tools`,
    `get_model_settings`, `get_serialization_name`) and checks their returns, down to
    each instruction and native tool, against the types the agent accepts, so a wrong
    return type fails here instead of crashing the next `agent.run`. The async
    lifecycle hooks need a live `RunContext` and are not invoked. Anything else the
    authored code raises, `SystemExit` and other `BaseException` subclasses included,
    is reported as a validation error rather than ending the process; only
    `KeyboardInterrupt` propagates. Loads of the same source from different threads
    run one at a time. Loads of different sources run concurrently, so authored
    code that loads a second authored file while it is itself loading can deadlock
    against a second thread loading the same two files in the opposite order.

    Returns an `AbstractCapability[object]`; `AgentDepsT` is contravariant, so the
    instance is accepted by any agent's `capabilities=` parameter.

    The module stays in `sys.modules`, where its classes resolve string annotations,
    under a key derived from `path` and its source. Loading unchanged source again
    replaces that entry; an instance loaded from an earlier version keeps its own,
    so each version loaded from `path` stays registered for the life of the
    process. A load that fails leaves `sys.modules` as it was.
    """
    return _construct(path, path, keep=True)


def validate_capability_file(path: Path) -> str:
    """Run the checks of `load_capability_instance` and return the capability class name.

    Nothing from the module is kept: it is validated under a `sys.modules` key of
    its own and removed again, so an instance already loaded from `path` keeps its
    entry.
    """
    return validate_capability_source(path, file=path)


def validate_capability_source(source: Path, *, file: Path) -> str:
    """Validate the source at `source` as if it were saved at `file`, returning the class name.

    The store validates a staged copy before it replaces `file`; the authored code
    sees `file` as its `__file__`, as it will once loaded. Nothing from the module
    is kept in `sys.modules`.
    """
    return type(_construct(source, file, keep=False)).__name__
