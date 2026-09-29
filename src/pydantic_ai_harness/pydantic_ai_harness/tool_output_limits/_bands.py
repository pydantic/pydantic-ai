"""Size bands and the actions they trigger.

A band is a `(over, action)` pair: when a tool return's measured size is at least `over`,
its `action` runs. `ToolOutputLimits` holds an ordered band list and picks the first
match (largest threshold that fits), passing through anything below the smallest threshold.

Every action carries an optional `then` fallback, applied when the action cannot run --
`Spill` whose store errors, `Truncate`/`Summarize` on a binary payload, a `Summarize`
whose model call raises. `Spill(then=Truncate())` is the default: lossless when the store
works, a bounded truncation otherwise, never a silent drop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

from pydantic_ai_harness.tool_output_limits._payload import TruncationStrategy

if TYPE_CHECKING:
    from collections.abc import AsyncIterable

    from pydantic_ai.agent import EventStreamHandler
    from pydantic_ai.messages import AgentStreamEvent
    from pydantic_ai.models import Model
    from pydantic_ai.tools import RunContext

_DEFAULT_TRUNCATE_CHARS = 4_000
_DEFAULT_PREVIEW_CHARS = 1_000

# A summarizer callable: `(tool_name, full_text) -> summary`, sync or async.
SummarizeFunc = Callable[[str, str], str | Awaitable[str]]


@dataclass(frozen=True)
class Passthrough:
    """Leave the tool return untouched. Useful as an explicit no-op band."""


@dataclass(frozen=True)
class Truncate:
    """Clamp the stringified return to `max_chars`, including the truncation marker.

    Lossy, zero-cost, no read-back. `max_chars` is always characters, independent of the
    capability's `over_tokens` size unit. If the budget cannot fit both retained content and
    a complete marker, truncation keeps only the selected slice. A non-positive cap returns
    an empty string. Falls back to `then` for binary payloads, which cannot be stringify-truncated.
    """

    strategy: TruncationStrategy = TruncationStrategy.head_tail
    max_chars: int = _DEFAULT_TRUNCATE_CHARS
    then: Action | None = None

    _: KW_ONLY
    keep_tail_lines: int = 0
    """Prioritize the last N newline-delimited lines before truncating the remaining text.

    Zero preserves the selected strategy's usual behavior. LF and CRLF endings are kept.
    If these lines exceed `max_chars`, use tail truncation instead. If a complete marker
    would displace these lines, keep a bare tail slice within the cap.
    """

    def __post_init__(self) -> None:
        if self.keep_tail_lines < 0:
            raise ValueError('keep_tail_lines must be non-negative.')


@dataclass(frozen=True)
class Spill:
    """Persist the full return and replace it with a handle, preview, and shape sketch.

    Lossless: the model gets a `read_tool_result` handle to slice / grep / tail the full
    payload on demand. Falls back to `then` when no store accepts the write.
    """

    preview_chars: int = _DEFAULT_PREVIEW_CHARS
    then: Action | None = None


@dataclass(frozen=True)
class Summarize:
    """Replace the return with a size-gated LLM summary. The expensive band.

    `model=None` inherits the running agent's model (`ctx.model`), mirroring
    `SummarizingCompaction`. Pass a model id / instance to override, or a `summarize`
    callable to bypass the built-in prompt entirely. Summary usage folds into `ctx.usage`;
    the built-in agent receives the parent usage limits and reserves one request from a finite
    request limit for the pending parent request. Falls back to `then` on a binary payload or a
    failed call.

    The summary request is non-streaming unless `event_stream_handler` is set. Supplying any
    handler selects the streaming request path, which is what a summarizer endpoint that
    rejects non-streaming requests needs; pass `drain_summary_events` to take that path
    without handling the events.
    """

    model: str | Model | None = None
    summarize: SummarizeFunc | None = None
    then: Action | None = None
    event_stream_handler: EventStreamHandler[object] | None = None
    """If set, this handler is passed to the nested summary run, so the summarizer's own
    model-streaming events surface to the caller.

    Setting it also selects the streaming request path for the summary request. Left `None`,
    the request is non-streaming, which is what an endpoint that rejects streaming requests
    needs. The handler receives the summary run's own `RunContext`, never the outer run's,
    and the outer `Agent.run(...)` handler is not inherited. Ignored when `summarize` is set,
    since a custom callable performs no model request."""


Action = Passthrough | Truncate | Spill | Summarize


@dataclass(frozen=True)
class Band:
    """Trigger `action` once a return's measured size reaches `over` (chars or tokens)."""

    over: int
    action: Action


async def drain_summary_events(
    _ctx: RunContext[object],
    events: AsyncIterable[AgentStreamEvent],
) -> None:
    """An `event_stream_handler` that consumes summary events and discards them.

    Pass this as `Summarize(event_stream_handler=drain_summary_events)` when the summary
    endpoint requires a streaming request but the events themselves are not wanted. Supplying
    any handler selects the streaming request path; this one just discards what it receives.
    """
    async for _ in events:
        pass
