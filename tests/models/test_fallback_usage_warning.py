from __future__ import annotations

import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from decimal import Decimal

import pytest

from pydantic_ai import Agent, ModelRequest, ModelResponse, RunContext, UsageNotReportedWarning, UserPromptPart
from pydantic_ai._run_context import set_current_run_context
from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from ..conftest import try_import
from .mock_openai import MockOpenAI, completion_message

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletionMessage
    from openai.types.completion_usage import CompletionUsage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def chat_model(content: str, *, reports_usage: bool) -> OpenAIChatModel:
    response = completion_message(
        ChatCompletionMessage(content=content, role='assistant'),
        usage=CompletionUsage(prompt_tokens=6, completion_tokens=3, total_tokens=9) if reports_usage else None,
    ).model_copy(update={'model': 'gpt-4o'})
    return OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock(response)))


def with_rejected_cost(successful_model: Model) -> FallbackModel:
    def reject_primary(response: ModelResponse) -> bool:
        return response.text == 'rejected'

    return FallbackModel(chat_model('rejected', reports_usage=True), successful_model, fallback_on=reject_primary)


@pytest.mark.parametrize('token_limit', [None, 100], ids=['cost-only', 'cost-and-tokens'])
async def test_fallback_unreported_usage_warns_before_adding_rejected_cost(
    allow_model_requests: None, token_limit: int | None
) -> None:
    # Exercise the adapter's missing-usage path; FunctionModel estimates omitted token counts.
    model = with_rejected_cost(chat_model('accepted', reports_usage=False))
    limits = UsageLimits(cost_limit=Decimal('1'), input_tokens_limit=token_limit)

    with pytest.warns(UsageNotReportedWarning, match="the response from 'gpt-4o' reported no token usage") as caught:
        result = await Agent(model).run('hello', usage_limits=limits)

    assert len(caught) == 1
    assert result.output == 'accepted'
    assert result.usage.total_tokens == 0
    assert result.usage.cost == Decimal('0.000045')


@pytest.mark.parametrize('run_context', [False, True], ids=['without-context', 'without-limits'])
async def test_fallback_without_usage_limits_does_not_warn(allow_model_requests: None, run_context: bool) -> None:
    model = with_rejected_cost(chat_model('accepted', reports_usage=False))
    context = (
        set_current_run_context(RunContext(deps=None, model=model, usage=RunUsage())) if run_context else nullcontext()
    )

    with context, warnings.catch_warnings():
        warnings.simplefilter('error', UsageNotReportedWarning)
        response = await model_request(model, [ModelRequest(parts=[UserPromptPart('hello')])])

    assert response.text == 'accepted'
    assert response.usage.total_tokens == 0
    assert response.usage.cost == Decimal('0.000045')


@dataclass(kw_only=True)
class CostReportingModel(TestModel):
    """Return explicit cost-only usage, including zero, without FunctionModel's token estimation."""

    reported_cost: Decimal

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await super().request(messages, model_settings, model_request_parameters)
        response.usage = RequestUsage(cost=self.reported_cost)
        return response


@pytest.mark.parametrize('reported_cost', [Decimal('0'), Decimal('0.1')], ids=['zero', 'nonzero'])
@pytest.mark.parametrize('fallback', [False, True], ids=['direct', 'fallback'])
@pytest.mark.parametrize('token_limit', [None, 100], ids=['cost-only', 'cost-and-tokens'])
async def test_provider_reported_cost_does_not_replace_missing_token_usage(
    allow_model_requests: None, reported_cost: Decimal, fallback: bool, token_limit: int | None
) -> None:
    model: Model = CostReportingModel(reported_cost=reported_cost, custom_output_text='accepted')
    if fallback:
        model = with_rejected_cost(model)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', UsageNotReportedWarning)
        result = await Agent(model).run(
            'hello', usage_limits=UsageLimits(cost_limit=Decimal('1'), input_tokens_limit=token_limit)
        )

    assert sum(issubclass(item.category, UsageNotReportedWarning) for item in caught) == int(token_limit is not None)
    assert result.output == 'accepted'
    assert result.usage.total_tokens == 0
    assert result.usage.cost == reported_cost + (Decimal('0.000045') if fallback else 0)
