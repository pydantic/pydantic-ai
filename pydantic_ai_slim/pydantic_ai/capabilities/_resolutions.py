"""What each configured capability resolved to, so the agent run span can record the capabilities the user configured.

Binding (`for_agent`) and `for_run` can replace a capability with a copy, or with something else entirely, such as
Harness `Skills` returning a group of one capability per skill. Only the replacements are registered for the run, so
without this record a trace couldn't name the capability the user configured. Recording happens only while an agent
is binding or resolving its capabilities, and costs a dict entry per capability that was replaced.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .abstract import AbstractCapability

_recording: ContextVar[dict[int, tuple[AbstractCapability[Any], AbstractCapability[Any]]] | None] = ContextVar(
    '_recording_capability_resolutions', default=None
)


@contextmanager
def recording_resolutions(
    resolutions: dict[int, tuple[AbstractCapability[Any], AbstractCapability[Any]]],
) -> Generator[None]:
    """Record into `resolutions`, by the `id()` of each capability replaced in this block, it and its replacement.

    The capability is kept alongside its replacement so that its `id()` can't be reused by another object.
    """
    token = _recording.set(resolutions)
    try:
        yield
    finally:
        _recording.reset(token)


def record_resolution(capability: AbstractCapability[Any], resolved: AbstractCapability[Any]) -> None:
    """Note that `capability`'s `for_agent` or `for_run` returned `resolved`, if a recording is in progress."""
    if resolved is not capability and (resolutions := _recording.get()) is not None:
        resolutions[id(capability)] = (capability, resolved)


def resolution_chain(
    capability: AbstractCapability[Any],
    resolutions: Mapping[int, tuple[AbstractCapability[Any], AbstractCapability[Any]]],
) -> list[AbstractCapability[Any]]:
    """`capability` and what it became in turn, through its binding and its `for_run` replacement."""
    chain = [capability]
    # Bounded, so a replacement that resolves back to what it replaced can't loop.
    while (
        len(chain) <= len(resolutions)
        and (entry := resolutions.get(id(chain[-1]))) is not None
        and entry[0] is chain[-1]
    ):
        chain.append(entry[1])
    return chain
