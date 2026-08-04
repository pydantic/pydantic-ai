"""Collect public dataclasses whose constructor takes two or more positional fields.

Run as a script by `test_public_interface_contracts.py`, which parses the JSON object printed to
stdout. The logic lives in a real module rather than inline in the test so that ruff and pyright
check it like any other code — an enforcement guard that silently decays into one that always
passes is worse than no guard.

The test runs this out of process with the `COVERAGE_*` environment scrubbed: walking the package
imports every module, including ones whose optional dependency is missing in the current
environment, which executes their `raise ImportError` guards. Those lines are marked
`pragma: no cover`, so covering them here would fail CI's strict pragma audit.

Modules that fail to import are reported in `skipped` rather than dropped, so a walk that shrinks
because an optional dependency is absent is visible to the caller instead of quietly reducing the
check to nothing.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import pkgutil

import pydantic_ai
from pydantic_ai.models import StreamedResponse


def _takes_two_positional_arguments(cls: type) -> bool | None:
    """Whether a caller may pass two or more arguments positionally to `cls(...)`, or `None` if unreadable.

    Reads the constructor signature rather than `dataclasses.fields`, so the generated and the
    hand-written `@dataclass(init=False)` cases are measured the same way, and so no CPython
    private attribute is needed to tell them apart.
    """
    try:
        parameters = inspect.signature(cls).parameters.values()
    except (TypeError, ValueError):
        return None
    positional = 0
    for parameter in parameters:
        # `*args` accepts arbitrarily many positional arguments, so a hand-written `__init__` that
        # declares one is over the threshold however few named parameters precede it.
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
    return positional >= 2


def collect() -> dict[str, list[str]]:
    offenders: set[str] = set()
    skipped: list[str] = []
    unreadable: list[str] = []

    for module_info in pkgutil.walk_packages(
        pydantic_ai.__path__,
        'pydantic_ai.',
        # Without `onerror`, a package whose `__init__` raises `ImportError` takes its whole subtree
        # out of the walk silently: those modules reach neither `offenders` nor the `import_module`
        # guard below, so the gate would pass on a walk that quietly shrank. The callback is handed
        # the package name, not the exception -- the reason for that package's own failure is what
        # the guard below records.
        onerror=lambda name: skipped.append(f'{name}: package failed to import, subtree not walked'),
    ):
        try:
            module = importlib.import_module(module_info.name)
        except ImportError as exc:
            skipped.append(f'{module_info.name}: {exc}')
            continue

        for obj in vars(module).values():
            if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
                continue
            if obj.__module__ != module_info.name:
                continue

            name = f'{obj.__module__}.{obj.__qualname__}'
            # A leading underscore anywhere in the path means the module or the class is private.
            # The convention being enforced is about public construction breaking when a field is
            # added; internals can be refactored freely, and gating them would bury the real
            # signal under entries no caller can construct.
            if any(part.startswith('_') for part in name.split('.')):
                continue
            # `StreamedResponse` subclasses are the streaming protocol's implementation types: the
            # framework constructs them inside `Model.request_stream()` and the user only ever
            # receives one. Gating them would fail every new-provider PR and push each new
            # provider into a shape inconsistent with its siblings.
            if issubclass(obj, StreamedResponse):
                continue

            takes_positional_pair = _takes_two_positional_arguments(obj)
            if takes_positional_pair is None:
                unreadable.append(name)
            elif takes_positional_pair:
                offenders.add(name)

    return {'offenders': sorted(offenders), 'skipped': sorted(skipped), 'unreadable': sorted(unreadable)}


if __name__ == '__main__':
    print(json.dumps(collect()))
