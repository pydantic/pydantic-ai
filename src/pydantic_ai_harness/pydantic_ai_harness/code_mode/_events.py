"""Capability events emitted by `CodeMode`'s speculative execution.

Launches happen while the model is still streaming and adoption happens inside `run_code`'s
dispatch, so without these events the speculation lifecycle is invisible. They surface every
transition on the run's event stream for live UIs (the streamed code with its closed/provisional
boundary, per-call launch spans, hit/miss/evicted outcomes) and for other capabilities
subscribing with [`on_event`][pydantic_ai.capabilities.on_event].

Every event carries the `run_code` part's tool call id in the inherited
[`tool_call_id`][pydantic_ai.messages.CapabilityEvent.tool_call_id] field, so one streamed
snippet's events correlate across the stream and execution phases. Launch-scoped events share a
`launch_id` unique within the run. A launch always ends in exactly one of claimed or evicted; a
settled event arrives in between when stream traffic allows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_ai.messages import CapabilityEvent

CODE_MODE_EVENTS = 'code_mode'
"""Namespace for `CodeMode` capability events."""


@dataclass(kw_only=True)
class SpeculativeCodeUpdateEvent(CapabilityEvent, namespace=CODE_MODE_EVENTS):
    """The decoded `run_code` snippet grew.

    Emitted per argument delta while a `run_code` tool call streams, carrying the full decoded
    code prefix, so consumers render live code without re-implementing partial-JSON decoding.
    `closed_statements` marks the boundary between statements that are provably complete and
    the still-growing tail. Mirrors the model's delta cadence; debounce for fewer repaints.
    """

    code: str
    """The decoded code prefix streamed so far."""

    closed_statements: int
    """Top-level statements at the front of `code` that are provably complete."""


@dataclass(kw_only=True)
class SpeculativeCallLaunchedEvent(CapabilityEvent, namespace=CODE_MODE_EVENTS):
    """A sandbox tool call started executing while its snippet was still streaming."""

    launch_id: str
    """Identifier shared by this launch's later settled and claimed/evicted events."""

    sandbox_function: str
    """The function name as written in the snippet (possibly sanitized)."""

    wrapped_tool_name: str
    """The wrapped tool actually dispatched."""

    arguments: dict[str, object]
    """The literal keyword arguments the call was launched with."""

    line_start: int
    """1-based first line of the launching statement within the snippet."""

    line_end: int
    """1-based last line of the launching statement within the snippet."""

    phase: Literal['streaming', 'execution'] = 'streaming'
    """When the launch happened: `streaming` overlaps the model's own generation;
    `execution` is the pre-sandbox prefetch that parallelizes the snippet's sequential awaits."""


@dataclass(kw_only=True)
class SpeculativeCallSettledEvent(CapabilityEvent, namespace=CODE_MODE_EVENTS):
    """A launched call finished (successfully or not) while the stream was still flowing.

    Emitted from the stream watcher when it next observes an event after the launch's task
    completes, so it can trail completion by a few deltas. A launch that settles after the
    stream ends produces no settled event; its terminal claimed/evicted event carries the
    settled state instead.
    """

    launch_id: str
    """Identifier from this launch's `SpeculativeCallLaunchedEvent`."""

    outcome: Literal['ready', 'failed']
    """Whether the early run produced a result or an error (delivered at claim, like a cold call's)."""

    elapsed_ms: float
    """Wall-clock from launch to completion."""


@dataclass(kw_only=True)
class SpeculativeCallClaimedEvent(CapabilityEvent, namespace=CODE_MODE_EVENTS):
    """The executing snippet dispatched a call and adopted an in-flight launch: a hit."""

    launch_id: str
    """Identifier from this launch's `SpeculativeCallLaunchedEvent`."""

    nested_tool_call_id: str
    """The adopted call's id in message history (`{part_id}__{n}`), distinct from the launch id."""

    wrapped_tool_name: str

    ready_at_claim: bool
    """Whether the early run had already finished when the snippet asked for it."""

    elapsed_ms: float
    """Wall-clock from launch until the result was available to the claimant."""


@dataclass(kw_only=True)
class SpeculativeCallMissedEvent(CapabilityEvent, namespace=CODE_MODE_EVENTS):
    """The executing snippet dispatched a speculation-eligible call that no launch matched.

    Only emitted for calls whose function was eligible to speculate this step; dispatches of
    never-eligible tools are ordinary calls, not misses. A miss is always safe (the call runs
    cold), but a systematic miss pattern signals broken claim keying rather than bad luck.
    """

    sandbox_function: str
    wrapped_tool_name: str
    nested_tool_call_id: str


@dataclass(kw_only=True)
class SpeculativeCallEvictedEvent(CapabilityEvent, namespace=CODE_MODE_EVENTS):
    """A launch was discarded without being claimed: wasted work.

    Emitted when the snippet finishes without dispatching a matching call (untaken branch,
    rewritten plan). Launches still unclaimed when the run ends are cancelled without an
    event, since the stream has closed by then.
    """

    launch_id: str
    """Identifier from this launch's `SpeculativeCallLaunchedEvent`."""

    wrapped_tool_name: str

    state: Literal['pending', 'ready', 'failed']
    """Where the launch was when discarded: still running, or settled with a result or error."""
