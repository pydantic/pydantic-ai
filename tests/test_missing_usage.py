from __future__ import annotations

import warnings
from decimal import Decimal

import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, UsageLimitUnavailableWarning
from pydantic_ai._genai_prices import best_effort_price
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits


def test_unmeasured_requests_accumulate_and_subtract_exactly() -> None:
    before = RunUsage(requests=1, input_tokens=20, unmeasured_requests=1)
    after = before + RunUsage(requests=1, input_tokens=200, output_tokens=20, unmeasured_requests=2)

    assert after.unmeasured_requests == 3
    assert after.input_tokens == 220
    assert (after - before) == RunUsage(requests=1, input_tokens=200, output_tokens=20, unmeasured_requests=2)
    assert (RequestUsage(unmeasured_requests=1) + RequestUsage(unmeasured_requests=2)).unmeasured_requests == 3


def test_unmeasured_requests_roundtrip_and_default_is_omitted() -> None:
    request_adapter = TypeAdapter(RequestUsage)
    run_adapter = TypeAdapter(RunUsage)

    assert 'unmeasured_requests' not in request_adapter.dump_python(RequestUsage())
    assert 'unmeasured_requests' not in run_adapter.dump_python(RunUsage())

    usage = RunUsage(requests=2, input_tokens=20, unmeasured_requests=1)
    serialized = run_adapter.dump_json(usage)
    assert b'"unmeasured_requests":1' in serialized
    assert run_adapter.validate_json(serialized) == usage

    default_messages: list[ModelMessage] = [ModelResponse(parts=[], usage=RequestUsage())]
    default_serialized_messages = ModelMessagesTypeAdapter.dump_json(default_messages)
    assert b'"unmeasured_requests"' not in default_serialized_messages

    messages: list[ModelMessage] = [ModelResponse(parts=[], usage=RequestUsage(unmeasured_requests=2))]
    serialized_messages = ModelMessagesTypeAdapter.dump_json(messages)
    restored_messages = ModelMessagesTypeAdapter.validate_json(serialized_messages)
    assert isinstance(restored_messages[0], ModelResponse)
    assert restored_messages[0].usage.unmeasured_requests == 2


def test_unmeasured_requests_are_not_values_or_token_details() -> None:
    usage = RunUsage(unmeasured_requests=1)

    assert usage.has_values() is False
    assert 'gen_ai.usage.details.unmeasured_requests' not in usage.opentelemetry_attributes()


@pytest.mark.parametrize('usage_type', [RequestUsage, RunUsage], ids=['request', 'run'])
def test_missing_usage_is_not_priced_as_zero(usage_type: type[RequestUsage]) -> None:
    usage = usage_type(unmeasured_requests=1)

    assert best_effort_price(usage, model_name='gpt-4o', provider_name='openai') is None
    assert usage.unmeasured_requests == 1


@pytest.mark.parametrize('usage_type', [RequestUsage, RunUsage], ids=['request', 'run'])
def test_best_effort_price_keeps_known_partial_cost(usage_type: type[RequestUsage]) -> None:
    details = {'web_search_requests': 1}
    web_search_usage = usage_type(web_searches=1, unmeasured_requests=1, details=details)
    web_search_price = best_effort_price(web_search_usage, model_name='gpt-4o', provider_name='openai')
    assert web_search_price is not None
    assert web_search_price.total_price == Decimal('0.01')

    token_usage = usage_type(input_tokens=20, unmeasured_requests=1)
    token_price = best_effort_price(token_usage, model_name='gpt-4o', provider_name='openai')
    assert token_price is not None
    assert token_price.total_price > Decimal('0')

    assert web_search_usage.unmeasured_requests == 1
    assert web_search_usage.details == details


def test_response_cost_rejects_partial_cost() -> None:
    response = ModelResponse(
        parts=[],
        usage=RequestUsage(web_searches=1, unmeasured_requests=1, cost=Decimal('0.01')),
        model_name='gpt-4o',
        provider_name='openai',
    )

    with pytest.raises(ValueError, match='usage information is missing'):
        response.cost()


def test_known_zero_usage_remains_priceable() -> None:
    usage = RequestUsage(unmeasured_requests=0)

    price = best_effort_price(usage, model_name='gpt-4o', provider_name='openai')
    assert price is not None
    assert price.total_price == Decimal('0')


def test_configured_token_limit_warns_when_usage_is_unavailable() -> None:
    limits = UsageLimits(input_tokens_limit=100)
    usage = RunUsage(unmeasured_requests=1)

    with pytest.warns(UsageLimitUnavailableWarning, match='known token and cost totals are lower bounds'):
        limits.check_tokens(usage)


def test_configured_cost_limit_warns_when_usage_is_unavailable() -> None:
    limits = UsageLimits(cost_limit=Decimal('0.01'))
    usage = RunUsage(unmeasured_requests=1)

    with pytest.warns(UsageLimitUnavailableWarning, match='known token and cost totals are lower bounds'):
        limits.check_cost(usage)


def test_unavailable_usage_warning_requires_token_or_cost_limit() -> None:
    usage = RunUsage(unmeasured_requests=1)
    limits_without_token_or_cost = [UsageLimits(), UsageLimits(request_limit=2), UsageLimits(tool_calls_limit=2)]

    with warnings.catch_warnings():
        warnings.simplefilter('error', UsageLimitUnavailableWarning)
        for limits in limits_without_token_or_cost:
            limits.check_tokens(usage)


@pytest.mark.parametrize('has_marker', [True, False], ids=['explicitly-unmeasured', 'default-usage'])
async def test_agent_function_model_preserves_explicit_missing_usage(has_marker: bool) -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='world')], usage=RequestUsage(unmeasured_requests=int(has_marker)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', UsageLimitUnavailableWarning)
        result = await Agent(FunctionModel(respond)).run('hello', usage_limits=UsageLimits(input_tokens_limit=100))

    unavailable_warnings = [item for item in caught if issubclass(item.category, UsageLimitUnavailableWarning)]
    assert len(unavailable_warnings) == int(has_marker)
    assert result.usage.unmeasured_requests == int(has_marker)
    if has_marker:
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
    else:
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0
