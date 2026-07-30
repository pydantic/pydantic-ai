"""`compact_now` -- run a compaction strategy outside an agent run."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, overload

from opentelemetry.trace import NoOpTracer
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.compaction._shared import SupportsFocus, compact_with_span

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer
    from pydantic_ai._run_context import AgentDepsT
    from pydantic_ai.models import Model

    from pydantic_ai_harness.compaction._shared import CompactionStrategy


@overload  # pragma: no cover -- a typing-only stub, never executed
async def compact_now(
    strategy: CompactionStrategy[None],
    messages: list[ModelMessage],
    *,
    model: Model | str,
    focus: str | None = None,
    usage: RunUsage | None = None,
    tracer: Tracer | None = None,
    tokenizer: Callable[[str], int] | None = None,
) -> list[ModelMessage]: ...


@overload  # pragma: no cover -- a typing-only stub, never executed
async def compact_now(
    strategy: CompactionStrategy[AgentDepsT],
    messages: list[ModelMessage],
    *,
    model: Model | str,
    deps: AgentDepsT,
    focus: str | None = None,
    usage: RunUsage | None = None,
    tracer: Tracer | None = None,
    tokenizer: Callable[[str], int] | None = None,
) -> list[ModelMessage]: ...


async def compact_now(
    strategy: CompactionStrategy[Any],
    messages: list[ModelMessage],
    *,
    model: Model | str,
    focus: str | None = None,
    deps: Any = None,
    usage: RunUsage | None = None,
    tracer: Tracer | None = None,
    tokenizer: Callable[[str], int] | None = None,
) -> list[ModelMessage]:
    """Compact `messages` with `strategy`, without an agent run in progress.

    A strategy's `compact` takes a `RunContext` because it may need the run's model and usage
    -- but an application holding a conversation *between* runs has neither, which is exactly
    when a user-invoked `/compact` happens. This builds a throwaway context so the same
    strategy the agent uses can be driven from a command handler, and the compacted history
    handed back to the next `agent.run(message_history=...)`.

    `compact_now` applies no trigger of its own, so a strategy whose `compact` is unconditional
    runs whatever the history size. A strategy that defines its own stop condition still honours
    it: `TieredCompaction` escalates only until the history fits its target, so a history already
    under target comes back unchanged -- by that strategy's own definition there is nothing left
    to reclaim. Pass the tier directly if you need it to run regardless.

    A compaction that changes the history emits the same `compact_messages` span the in-run
    path emits, so an instrumented application sees one shape however compaction was
    triggered. Pass `tracer` to record it; without one the span goes to a no-op tracer.

    Args:
        strategy: The strategy to run. Any `CompactionStrategy` works.
        messages: History to compact. Not mutated; the compacted list is returned.
        model: Model the strategy should use, needed by summarizing strategies that call one.
        focus: What the summary should prioritize. Passed over by strategies that cannot honour
            it; a composing strategy forwards it to the tiers that can.
        deps: Dependencies to expose on the throwaway context. Required for a strategy that
            declares a deps type, so one is never handed a context whose `deps` is `None` it
            has no way to expect.
        usage: Usage to accumulate into, so a summarization call can be billed to your own
            counter. A fresh `RunUsage` is used when omitted.
        tracer: Tracer the `compact_messages` span is started on. Defaults to a no-op tracer,
            which records nothing.
        tokenizer: Tokenizer the span's before/after counts are measured with. Pass the one the
            strategy was configured with, or the span reports the 4-characters heuristic for a
            run the in-run path measured differently. `CompactionStrategy` does not promise a
            `tokenizer` attribute, so it cannot be read off `strategy`.

    Example:
        ```python {test="skip"}
        from pydantic_ai_harness.compaction import SummarizingCompaction, compact_now

        strategy = SummarizingCompaction(max_fraction=0.9, keep_messages=20)
        history = await compact_now(
            strategy,
            history,
            model='anthropic:claude-sonnet-4-6',
            focus='the auth refactor, not the earlier CSS work',
        )
        ```
    """
    from pydantic_ai.models import infer_model

    focused = strategy.with_focus(focus) if focus is not None and isinstance(strategy, SupportsFocus) else strategy

    ctx = RunContext[Any](
        deps=deps,
        model=infer_model(model) if isinstance(model, str) else model,
        usage=usage if usage is not None else RunUsage(),
        tracer=tracer if tracer is not None else NoOpTracer(),
    )
    return await compact_with_span(
        ctx,
        strategy=type(focused).__name__,
        messages=messages,
        compact=lambda: focused.compact(messages, ctx),
        tokenizer=tokenizer,
    )
