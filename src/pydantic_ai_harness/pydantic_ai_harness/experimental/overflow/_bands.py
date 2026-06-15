"""Size bands and the actions they trigger.

A band is a `(over, action)` pair: when a tool return's measured size is at least `over`,
its `action` runs. `OverflowingToolOutput` holds an ordered band list and picks the first
match (largest threshold that fits), passing through anything below the smallest threshold.

Every action carries an optional `then` fallback, applied when the action cannot run --
`Spill` whose store errors, `Truncate`/`Summarize` on a binary payload, a `Summarize`
whose model call raises. `Spill(then=Truncate())` is the default: lossless when the store
works, a bounded truncation otherwise, never a silent drop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai_harness.experimental.overflow._payload import TruncationStrategy

if TYPE_CHECKING:
    from pydantic_ai.models import Model

_DEFAULT_TRUNCATE_CHARS = 4_000
_DEFAULT_PREVIEW_CHARS = 1_000

# A summarizer callable: `(tool_name, full_text) -> summary`, sync or async.
SummarizeFunc = Callable[[str, str], str | Awaitable[str]]


@dataclass(frozen=True)
class Passthrough:
    """Leave the tool return untouched. Useful as an explicit no-op band."""


@dataclass(frozen=True)
class Truncate:
    """Clamp the stringified return to `max_chars`. Lossy, zero-cost, no read-back.

    `max_chars` is always characters, independent of the capability's `over_tokens` size
    unit (truncation is a character operation). Falls back to `then` for binary payloads,
    which cannot be stringify-truncated.
    """

    strategy: TruncationStrategy = TruncationStrategy.head_tail
    max_chars: int = _DEFAULT_TRUNCATE_CHARS
    then: Action | None = None


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
    no token caps are imposed. Falls back to `then` on a binary payload or a failed call.
    """

    model: str | Model | None = None
    summarize: SummarizeFunc | None = None
    then: Action | None = None


Action = Passthrough | Truncate | Spill | Summarize


@dataclass(frozen=True)
class Band:
    """Trigger `action` once a return's measured size reaches `over` (chars or tokens)."""

    over: int
    action: Action
