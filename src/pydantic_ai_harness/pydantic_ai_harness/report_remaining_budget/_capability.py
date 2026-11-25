"""`ReportRemainingBudget` -- tells the model how much of its budget is left on every request."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import InstructionCapabilityPart, ModelRequest, OperatorSource, OperatorText
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.spend_limits import BudgetStatus, SpendLimits, Window

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

_WINDOW_LABELS: dict[Window, str] = {
    'run': 'spend limit for this run',
    'conversation': 'this conversation',
    'day': 'today',
    'month': 'this month',
    'total': 'in total',
}


@dataclass(kw_only=True)
class RemainingBudgetPart(InstructionCapabilityPart, namespace='report_remaining_budget', name='remaining_budget'):
    """The request-scoped status `ReportRemainingBudget` enqueues. Find statuses in history by this type."""


def _count(count: int, noun: str) -> str:
    count = max(count, 0)
    return f'{count} {noun}' if count == 1 else f'{count} {noun}s'


def _tokens(count: int, kind: str) -> str:
    count = max(count, 0)
    if count < 1000:
        return f'{count} {kind}'
    size, unit = (1000, 'k') if count < 1_000_000 else (1_000_000, 'M')
    tenths = count * 10 // size
    rounded = f'{tenths // 10}.{tenths % 10}'.removesuffix('.0') if tenths < 100 else str(tenths // 10)
    return f'about {rounded}{unit} {kind}'


def _usd(amount: Decimal) -> str:
    return f'${max(amount, Decimal(0)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)}'


def _run_line(ctx: RunContext[AgentDepsT]) -> str | None:
    limits, usage = ctx.usage_limits, ctx.usage
    if limits is None:
        return None
    items: list[str] = []
    if limits.request_limit is not None:
        items.append(f'{_count(limits.request_limit - usage.requests, "request")} including this one')
    if limits.tool_calls_limit is not None:
        items.append(_count(limits.tool_calls_limit - usage.tool_calls, 'tool call'))
    for limit, used, kind in (
        (limits.input_tokens_limit, usage.input_tokens, 'input tokens'),
        (limits.output_tokens_limit, usage.output_tokens, 'output tokens'),
        (limits.total_tokens_limit, usage.total_tokens, 'total tokens'),
    ):
        if limit is not None:
            items.append(_tokens(limit - used, kind))
    if limits.cost_limit is not None:
        items.append(_usd(limits.cost_limit - (usage.cost or 0)))
    return f'- this run: {", ".join(items)}' if items else None


def _budget_line(status: BudgetStatus) -> str | None:
    items: list[str] = []
    if status.remaining_usd is not None:
        items.append(_usd(status.remaining_usd))
    if status.remaining_tokens is not None:
        items.append(_tokens(status.remaining_tokens, 'tokens'))
    if not items:
        return None
    name = '' if status.budget.name == 'default' else f' ({status.budget.name})'
    return f'- {_WINDOW_LABELS[status.budget.window]}{name}: {", ".join(items)}'


@dataclass
class ReportRemainingBudget(AbstractCapability[AgentDepsT]):
    """Tell the model how much of the run's usage limits and spend budgets remain, on every model request.

    The run's figures come from the `UsageLimits` the run enforces (`RunContext.usage_limits`) and its
    usage so far, so nothing is configured twice. Pass the agent's `SpendLimits` to add the remainder
    of each of its budgets that has a ceiling, labelled by window so a daily remainder is not read as
    the run's.

    The report is enqueued as a [request-scoped][pydantic_ai.messages.CapabilityPart.scope]
    `RemainingBudgetPart` in the application's voice: the model sees it for the request it's part of
    only, and it stays in the message history; the prefix before it is still read from cache. This only
    discloses the budget; `UsageLimits` and `SpendLimits` still enforce it.

    Example:
        ```python
        from decimal import Decimal

        from pydantic_ai import Agent
        from pydantic_ai_harness import ReportRemainingBudget, SpendLimits
        from pydantic_ai_harness.spend_limits import Budget

        spend = SpendLimits(budgets=[Budget(usd=Decimal(10), window='day')])
        agent = Agent('openai:gpt-5.6-sol', capabilities=[spend, ReportRemainingBudget(spend_limits=spend)])
        ```
    """

    spend_limits: SpendLimits[AgentDepsT] | None = None
    """The agent's `SpendLimits`, whose budgets to report alongside the run's limits.

    Its store is read on every model request.
    """

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Enqueue the remaining budget into this request."""
        # A provider-resume turn continues the paused request, which already carried a report.
        if not request_context.messages or not isinstance(request_context.messages[-1], ModelRequest):
            return request_context
        lines = [_run_line(ctx)]
        if self.spend_limits is not None:
            lines.extend(_budget_line(status) for status in await self.spend_limits.status(ctx))
        if present := [line for line in lines if line is not None]:
            ctx.enqueue(
                RemainingBudgetPart(
                    # The capability's own wording, with counts and configured budget names filled in.
                    content=OperatorText('\n'.join(['Budget remaining:', *present])),
                    source=OperatorSource(),
                    scope='request',
                )
            )
        return request_context
