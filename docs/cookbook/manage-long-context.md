---
title: Keep a tool-heavy run within its context window
description: Compact old tool output in tiers while retaining important findings in typed application state.
---

# Keep a tool-heavy run within its context window

Long investigations can exhaust a model's context window even when each individual tool result is modest. Clear old results first, trim messages only when necessary, and retain findings outside destructive message compaction.

```bash
pip/uv-add "pydantic-ai-slim[anthropic]" "pydantic-ai-harness==0.31.0"
export ANTHROPIC_API_KEY=your-api-key
```

```python {call_name="build_agent" dunder_name="not_main" noqa="I001"}
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic_ai_harness.compaction import (
    ClearToolResults,
    ContextUsageEvent,
    ReportContextUsage,
    SlidingWindowCompaction,
    TieredCompaction,
)

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-fable-5')
ContextObserver = Callable[[ContextUsageEvent], None]


@dataclass
class InvestigationState:
    """Findings that must survive destructive history compaction."""

    findings: list[int] = field(default_factory=list[int])


def build_agent(
    model: Model | str = DEFAULT_MODEL,
    *,
    observe_context: ContextObserver | None = None,
    target_fraction: float = 0.75,
    fallback_context_window: int = 200_000,
) -> Agent[InvestigationState, str]:
    """Build an investigator that clears old tool output before trimming history."""
    agent: Agent[InvestigationState, str] = Agent(
        model,
        name='bounded_investigator',
        deps_type=InvestigationState,
        instructions=(
            'Inspect records one at a time. Keep the original investigation goal in view, '
            'and finish with a concise list of the records that need attention.'
        ),
        capabilities=[
            TieredCompaction(
                tiers=[
                    ClearToolResults(max_tokens=1, keep_pairs=2),
                    SlidingWindowCompaction(max_tokens=1, keep_messages=10),
                ],
                target_fraction=target_fraction,
                fallback_context_window=fallback_context_window,
            ),
            ReportContextUsage(fallback_context_window=fallback_context_window),
        ],
    )

    @agent.tool
    async def read_record(ctx: RunContext[InvestigationState], record_id: int) -> str:
        needs_attention = record_id in {3, 7}
        if needs_attention and record_id not in ctx.deps.findings:
            ctx.deps.findings.append(record_id)
        status = 'needs attention' if needs_attention else 'healthy'
        retained = ', '.join(str(item) for item in ctx.deps.findings) or 'none'
        return f'Record {record_id}: status={status}; retained findings={retained}; details=' + (
            'diagnostic context ' * 80
        )

    if observe_context is not None:

        @agent.on_event(ContextUsageEvent)
        async def observe(ctx: RunContext[InvestigationState], event: ContextUsageEvent) -> None:
            del ctx
            observe_context(event)

    return agent


def main() -> None:
    def report(event: ContextUsageEvent) -> None:
        source = 'model profile' if event.resolved else 'configured fallback'
        print(f'context: {event.used_tokens}/{event.window_tokens} tokens ({source})')

    build_agent(observe_context=report).to_cli_sync(deps=InvestigationState())


if __name__ == '__main__':
    main()
```

`ClearToolResults` preserves tool-call/result pairing while replacing older payloads. `SlidingWindowCompaction` is the destructive fallback. Important findings belong in typed state because no history compactor can guarantee that an old message remains available.

## Related

See [Keep recent conversation history](bounded-history.md) for a core-only history processor and [Harness compaction](https://pydantic.dev/docs/ai/harness/compaction/) for all compaction strategies.
