"""Cap how often the model can call tools in a run, and refuse exact repeats, by answering the model."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Set as AbstractSet
from dataclasses import dataclass, field, replace
from typing import Any, Literal, NoReturn, TypeAlias

from pydantic_ai.capabilities import AbstractCapability, WrapToolExecuteHandler
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, SkipToolExecution, UserError
from pydantic_ai.messages import ToolCallPart, ToolReturn
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

__all__ = ['ToolBudgetReason', 'ToolBudgets']

ToolBudgetReason: TypeAlias = Literal['repeat', 'per_tool', 'total']
"""Why `ToolBudgets` refused a call, as recorded under `'reason'` in the refusal's `ToolReturn.metadata`."""

_REFUSAL_ORIGIN = 'pydantic-ai-harness.tool_budgets.refusal.v1'
"""The `origin` in the `metadata` of the `ToolReturn` a refusal is answered with."""


@dataclass(kw_only=True)
class ToolBudgets(AbstractCapability[AgentDepsT]):
    """Cap tool calls per tool and per run, and refuse exact repeats, by answering the model instead of raising.

    A refused call doesn't run: the model gets a tool result that says which budget it hit and how much of it was
    used, and the run continues. `UsageLimits.tool_calls_limit` stays the hard ceiling that ends the run.

    Counts live on the instance `for_run` creates for each run, not in message history, so compaction can't hand
    a budget back. A call counts once its execution starts, whether it then succeeds, fails or defers; a call that
    another capability skips, or holds for approval, doesn't count.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import ToolBudgets

    budgets = ToolBudgets(per_tool={'web_search': 3}, total=20, refuse_repeats={'web_search'})
    agent = Agent('anthropic:claude-opus-5-5', capabilities=[budgets])
    ```
    """

    per_tool: Mapping[str, int] = field(default_factory=dict[str, int])
    """The most calls each named tool may run in one run. Tools not named here have no per-tool budget."""

    total: int | None = None
    """The most tool calls, across all tools, that may run in one run, or `None` for no total budget."""

    refuse_repeats: AbstractSet[str] = frozenset()
    """Tools whose call is refused when the same call, with the same arguments, already ran in this run.

    Opt in per tool: a lookup repeated with the same arguments returns what it returned before, but a polling tool
    legitimately repeats its call.
    """

    _calls: Counter[str] = field(default_factory=Counter[str], init=False, repr=False, compare=False)
    """How many calls to each tool have run in this run."""

    _ran: dict[str, str] = field(default_factory=dict[str, str], init=False, repr=False, compare=False)
    """The tool call id of each call to a `refuse_repeats` tool that ran, by canonical tool name and arguments."""

    def __post_init__(self) -> None:
        limits = {**{f'per_tool[{name!r}]': limit for name, limit in self.per_tool.items()}, 'total': self.total}
        for name, limit in limits.items():
            if limit is not None and limit < 1:
                raise UserError(f'`ToolBudgets.{name}` must be at least 1, got {limit}.')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> ToolBudgets[AgentDepsT]:
        """Return a fresh instance, so runs don't share their counts."""
        return replace(self)

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Refuse a call that is over budget or repeated, otherwise count it and run it.

        The checks and the count happen before the first `await`, so parallel calls can't overshoot a budget.
        """
        name = call.tool_name
        key = None
        if name in self.refuse_repeats:
            key = json.dumps([name, call.args_as_dict()], sort_keys=True, default=str)
            if (earlier := self._ran.get(key)) is not None:
                self._refuse(
                    ctx,
                    call,
                    'repeat',
                    f'`{name}` already ran with these exact arguments in this run (tool call `{earlier}`); its '
                    'result, or a summary of it, is earlier in this conversation. '
                    "Don't call it again with the same arguments.",
                )
        if (limit := self.per_tool.get(name)) is not None and self._calls[name] >= limit:
            self._refuse(
                ctx, call, 'per_tool', f'{limit} of {limit} `{name}` calls used; no more are allowed in this run.'
            )
        if self.total is not None and self._calls.total() >= self.total:
            self._refuse(
                ctx, call, 'total', f'{self.total} of {self.total} tool calls used; no more are allowed in this run.'
            )

        self._calls[name] += 1
        if key is not None:
            self._ran[key] = call.tool_call_id
        try:
            return await handler(args)
        except (SkipToolExecution, ApprovalRequired):
            self._calls[name] -= 1
            self._forget(key)
            raise
        except CallDeferred:
            raise
        except Exception:
            self._forget(key)
            raise

    def _forget(self, key: str | None) -> None:
        """Let a call that returned no result be made again."""
        if key is not None:
            del self._ran[key]

    def _refuse(
        self, ctx: RunContext[AgentDepsT], call: ToolCallPart, reason: ToolBudgetReason, message: str
    ) -> NoReturn:
        ctx.tracer.start_span(
            'tool call refused', attributes={'tool_budgets.reason': reason, 'tool_budgets.tool_name': call.tool_name}
        ).end()
        raise SkipToolExecution(
            ToolReturn(return_value=message, metadata={'origin': _REFUSAL_ORIGIN, 'reason': reason})
        )
