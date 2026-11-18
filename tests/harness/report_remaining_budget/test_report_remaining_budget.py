from __future__ import annotations

from decimal import Decimal

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits
from pydantic_ai_harness import ReportRemainingBudget, SpendLimits
from pydantic_ai_harness.report_remaining_budget import RemainingBudgetPart
from pydantic_ai_harness.spend_limits import Budget
from tests.function_model_prefix import REQUEST_SCOPED_PROMPT_DROPPED

pytestmark = pytest.mark.filterwarnings('ignore::pydantic_ai_harness.spend_limits.UnpricedModelWarning')


def _model(*usages: RequestUsage) -> FunctionModel:
    """Call the tool once per usage, reporting that usage, then answer."""
    replies = iter(usages)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        usage = next(replies, None)
        if usage is None:
            return ModelResponse(parts=[TextPart('done')], usage=RequestUsage(input_tokens=1, output_tokens=1))
        return ModelResponse(parts=[ToolCallPart('work', {})], usage=usage)

    return FunctionModel(respond)


def _agent(model: FunctionModel, *capabilities: AbstractCapability[None]) -> Agent[None, str]:
    agent = Agent(model, deps_type=type(None), capabilities=list(capabilities))

    @agent.tool_plain
    def work() -> str:
        return 'ok'

    return agent


def _reports(messages: list[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, RemainingBudgetPart) and part.scope == 'request'
    ]


class TestReportRemainingBudget:
    @pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
    async def test_reports_run_limits_on_every_request(self):
        agent = _agent(
            _model(
                RequestUsage(input_tokens=400, output_tokens=200, cost=Decimal('0.251')),
                RequestUsage(input_tokens=60_000, output_tokens=1_000, cost=Decimal('0.5')),
            ),
            ReportRemainingBudget(),
        )
        limits = UsageLimits(
            request_limit=3,
            tool_calls_limit=2,
            input_tokens_limit=2_000_000,
            output_tokens_limit=1_500,
            total_tokens_limit=62_000,
            cost_limit=Decimal('2'),
        )
        result = await agent.run('go', usage_limits=limits)
        assert _reports(result.all_messages()) == snapshot(
            [
                """\
Budget remaining:
- this run: 3 requests including this one, 2 tool calls, about 2M input tokens, about 1.5k output tokens, about 62k total tokens, $2.00\
""",
                """\
Budget remaining:
- this run: 2 requests including this one, 1 tool call, about 1.9M input tokens, about 1.3k output tokens, about 61k total tokens, $1.74\
""",
                """\
Budget remaining:
- this run: 1 request including this one, 0 tool calls, about 1.9M input tokens, 300 output tokens, 400 total tokens, $1.24\
""",
            ]
        )

    @pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
    async def test_reports_spend_budgets_by_window(self):
        spend = SpendLimits[None](
            budgets=[
                Budget(usd=Decimal(10), tokens=1_000_000, window='day'),
                Budget(tokens=500, window='month', name='tenant'),
                Budget(window='total', name='accounting'),
                Budget(usd=Decimal(3), window='run'),
            ]
        )
        agent = _agent(_model(RequestUsage(input_tokens=300, output_tokens=100)), spend, ReportRemainingBudget(spend))
        result = await agent.run('go', usage_limits=UsageLimits(request_limit=4))
        assert _reports(result.all_messages()) == snapshot(
            [
                """\
Budget remaining:
- this run: 4 requests including this one
- today: $10.00, about 1M tokens
- this month (tenant): 500 tokens
- spend limit for this run: $3.00\
""",
                """\
Budget remaining:
- this run: 3 requests including this one
- today: $10.00, about 999k tokens
- this month (tenant): 100 tokens
- spend limit for this run: $3.00\
""",
            ]
        )

    async def test_reports_nothing_without_limits(self):
        agent = _agent(_model(), ReportRemainingBudget())
        result = await agent.run('go', usage_limits=UsageLimits(request_limit=None))
        assert _reports(result.all_messages()) == []

    async def test_reports_nothing_outside_a_run_or_on_a_resumed_request(self):
        """`enqueue` raises outside a run, so returning normally shows nothing was reported."""
        capability = ReportRemainingBudget[None]()
        request_context = ModelRequestContext(
            model=TestModel(),
            messages=[ModelRequest.user_text_prompt('go')],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        )
        no_limits = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        assert await capability.before_model_request(no_limits, request_context) is request_context

        request_context.messages = [*request_context.messages, ModelResponse(parts=[TextPart('...')])]
        limited = RunContext(deps=None, model=TestModel(), usage=RunUsage(), usage_limits=UsageLimits())
        assert await capability.before_model_request(limited, request_context) is request_context
