from __future__ import annotations

import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from decimal import Decimal

import pytest

from pydantic_ai import Agent, ModelRequest, ModelResponse, RunContext, UsageLimitUnavailableWarning, UserPromptPart
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

    with pytest.warns(UsageLimitUnavailableWarning, match='response\\(s\\) omitted usage information') as caught:
        result = await Agent(model).run('hello', usage_limits=limits)

    assert len(caught) == 1
    assert result.output == 'accepted'
    assert result.usage.total_tokens == 0
    assert result.usage.unmeasured_requests == 1
    assert result.usage.cost == Decimal('0.000045')


@pytest.mark.parametrize('run_context', [False, True], ids=['without-context', 'without-limits'])
async def test_fallback_without_usage_limits_does_not_warn(allow_model_requests: None, run_context: bool) -> None:
    model = with_rejected_cost(chat_model('accepted', reports_usage=False))
    context = (
        set_current_run_context(RunContext(deps=None, model=model, usage=RunUsage())) if run_context else nullcontext()
    )

    with context, warnings.catch_warnings():
        warnings.simplefilter('error', UsageLimitUnavailableWarning)
        response = await model_request(model, [ModelRequest(parts=[UserPromptPart('hello')])])

    assert response.text == 'accepted'
    assert response.usage.total_tokens == 0
    assert response.usage.unmeasured_requests == 1
    assert response.usage.cost == Decimal('0.000045')


@dataclass(kw_only=True)
class CostReportingModel(TestModel):
    """Return explicit token/cost usage, including zero, without FunctionModel's token estimation."""

    reported_cost: Decimal
    input_tokens: int = 0
    output_tokens: int = 0

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await super().request(messages, model_settings, model_request_parameters)
        response.usage = RequestUsage(
            input_tokens=self.input_tokens, output_tokens=self.output_tokens, cost=self.reported_cost
        )
        return response


@pytest.mark.parametrize('rejected_count', [1, 2], ids=['one-missing-rejection', 'two-missing-rejections'])
@pytest.mark.parametrize('selected_cost', [None, Decimal('0.1')], ids=['computed-cost', 'explicit-cost'])
async def test_fallback_preserves_each_missing_usage_marker_before_known_response(
    allow_model_requests: None, rejected_count: int, selected_cost: Decimal | None
) -> None:
    def reject(response: ModelResponse) -> bool:
        return response.text == 'rejected'

    models: list[Model] = [chat_model('rejected', reports_usage=False) for _ in range(rejected_count)]
    if selected_cost is None:
        models.append(chat_model('accepted', reports_usage=True))
        expected_input_tokens, expected_output_tokens = 6, 3
    else:
        models.append(
            CostReportingModel(
                reported_cost=selected_cost,
                input_tokens=120,
                output_tokens=8,
                custom_output_text='accepted',
            )
        )
        expected_input_tokens, expected_output_tokens = 120, 8
    model = FallbackModel(models[0], *models[1:], fallback_on=reject)

    with pytest.warns(UsageLimitUnavailableWarning, match='response\\(s\\) omitted usage information') as caught:
        result = await Agent(model).run(
            'hello', usage_limits=UsageLimits(cost_limit=Decimal('1'), input_tokens_limit=200)
        )

    assert len(caught) == 1
    assert f'{rejected_count} response(s) omitted usage information' in str(caught[0].message)
    assert result.output == 'accepted'
    assert result.usage.unmeasured_requests == rejected_count
    assert result.usage.input_tokens == expected_input_tokens
    assert result.usage.output_tokens == expected_output_tokens
    if selected_cost is None:
        assert result.usage.cost is not None
        assert result.usage.cost > 0
    else:
        assert result.usage.cost == selected_cost


@pytest.mark.parametrize('reported_cost', [Decimal('0'), Decimal('0.1')], ids=['zero', 'nonzero'])
@pytest.mark.parametrize('fallback', [False, True], ids=['direct', 'fallback'])
@pytest.mark.parametrize('token_limit', [None, 100], ids=['cost-only', 'cost-and-tokens'])
async def test_cost_only_usage_is_known_zero_tokens_and_does_not_warn(
    allow_model_requests: None, reported_cost: Decimal, fallback: bool, token_limit: int | None
) -> None:
    model: Model = CostReportingModel(reported_cost=reported_cost, custom_output_text='accepted')
    if fallback:
        model = with_rejected_cost(model)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', UsageLimitUnavailableWarning)
        result = await Agent(model).run(
            'hello', usage_limits=UsageLimits(cost_limit=Decimal('1'), input_tokens_limit=token_limit)
        )

    assert sum(issubclass(item.category, UsageLimitUnavailableWarning) for item in caught) == 0
    assert result.output == 'accepted'
    assert result.usage.total_tokens == 0
    assert result.usage.unmeasured_requests == 0
    assert result.usage.cost == reported_cost + (Decimal('0.000045') if fallback else 0)
