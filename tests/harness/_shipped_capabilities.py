"""The capability classes `pydantic_ai_harness` ships, found by walking the package.

`test_capability_combine.py` checks each declares a combine policy, and `test_spec_conformance.py` checks each
spec schema against its class.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from typing import TypeGuard

import pydantic_ai_harness
from pydantic_ai.capabilities import AbstractCapability


def is_capability_class(obj: object) -> TypeGuard[type[AbstractCapability[object]]]:
    """Whether `obj` is a capability class, and not something that merely looks like one."""
    if not isinstance(obj, type):
        return False
    try:
        return issubclass(obj, AbstractCapability)
    except TypeError:  # pragma: lax no cover
        return False


def caused_by_missing_dependency(exc: BaseException) -> bool:
    """Whether `exc` is an uninstalled optional group rather than a broken capability.

    Follows the chain rather than matching a type, because the same condition reaches this walk in
    three shapes: `ModuleNotFoundError` from the import itself, a plain `ImportError` from a module
    that re-raises install guidance, and a `UserError` a capability raises while constructing
    something at import time. The last two are raised inside an `except ImportError` block, so the
    original is on `__cause__` or `__context__` either way.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ImportError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def shipped_capability_types() -> tuple[dict[str, type[AbstractCapability[object]]], list[str]]:
    """Every capability class in `pydantic_ai_harness`, public or not.

    Walks the package rather than reading an export list, so a capability that is never re-exported
    is still covered. Modules whose optional dependency group is not installed are skipped -- the
    import error means the capability could not have been used either.
    """
    found: dict[str, type[AbstractCapability[object]]] = {}
    skipped: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for module_info in pkgutil.walk_packages(pydantic_ai_harness.__path__, f'{pydantic_ai_harness.__name__}.'):
            try:
                module = importlib.import_module(module_info.name)
            except Exception as exc:
                if not caused_by_missing_dependency(exc):
                    # A real failure -- a broken module, a bad constant -- has to surface.
                    # Swallowing it drops the capability from `found`, and an exhaustiveness check
                    # over it then passes because the capability it should have caught is invisible
                    # rather than absent. That is not hypothetical: it is how `AWSLambdaDurability`
                    # reached this branch with no policy entry while this test was green locally.
                    raise
                skipped.append(module_info.name)
                continue
            for obj in vars(module).values():
                if is_capability_class(obj) and obj.__module__.startswith(pydantic_ai_harness.__name__):
                    known = found.setdefault(obj.__name__, obj)
                    assert known is obj, f'`{known.__module__}` and `{obj.__module__}` both define `{obj.__name__}`'
    return found, skipped
