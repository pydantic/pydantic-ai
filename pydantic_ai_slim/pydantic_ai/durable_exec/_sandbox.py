from __future__ import annotations

from typing import Any

from pydantic_ai.capabilities import AbstractCapability, WrapperCapability


def contributes_sandbox(capability: AbstractCapability[Any]) -> bool:
    """Whether the capability tree contains a `get_sandbox` override.

    Durable integrations reject sandbox-contributing capabilities up front: entering the
    contributed context manager would run I/O in workflow code. The check is static, so a
    contributor only produced at run time by a dynamic capability function cannot be caught
    here — the workflow engine then blocks the I/O itself, less legibly.

    `WrapperCapability.get_sandbox` is a pure forwarder, not a contribution; `apply()` also
    visits the wrapped capabilities, so a real contributor behind a wrapper is still found.
    """
    found = False

    def visit(leaf: AbstractCapability[Any]) -> None:
        nonlocal found
        get_sandbox = type(leaf).get_sandbox
        found = found or get_sandbox not in (AbstractCapability.get_sandbox, WrapperCapability.get_sandbox)

    capability.apply(visit)
    return found
