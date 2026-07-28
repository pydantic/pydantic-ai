from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from vcr.cassette import Cassette

from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import ToolDefinition

from ..cassette_utils import single_request_body

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]


def refund_tool() -> ToolDefinition:
    return ToolDefinition(
        name='lookup_refund_policy',
        description='Look up the refund policy for an order.',
        parameters_json_schema={
            'type': 'object',
            'properties': {'order_id': {'type': 'string'}},
            'required': ['order_id'],
        },
    )


async def test_supported_model_calls_additional_tool(
    allow_model_requests: None, openai_api_key: str, vcr: Cassette
) -> None:
    """A supported model acts on the native item and calls a tool absent from top-level `tools`."""
    model = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(api_key=openai_api_key))
    tool = refund_tool()

    messages = model.prepare_messages(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Call lookup_refund_policy with order_id order-123. Do not answer directly.')
                ]
            ),
            ModelResponse(parts=[TextPart(content='I will load the refund capability.')]),
            ModelRequest(parts=[ToolAvailabilityDeltaPart(added=[tool.name])]),
        ]
    )
    response = await model.request(
        messages,
        None,
        ModelRequestParameters(function_tools=[tool]),
    )

    assert len(response.parts) == 1
    call = response.parts[0]
    assert isinstance(call, ToolCallPart)
    assert call.tool_name == 'lookup_refund_policy'
    assert call.args == '{"order_id":"order-123"}'
    assert call.tool_call_id
    assert call.id
    assert call.provider_name == 'openai'
    body = single_request_body(vcr)
    assert 'tools' not in body
    assert body['input'][-1] == snapshot(
        {
            'type': 'additional_tools',
            'role': 'developer',
            'tools': [
                {
                    'type': 'function',
                    'name': 'lookup_refund_policy',
                    'description': 'Look up the refund policy for an order.',
                    'parameters': {
                        'type': 'object',
                        'properties': {'order_id': {'type': 'string'}},
                        'required': ['order_id'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                }
            ],
        }
    )


async def test_unsupported_model_calls_tool_via_synthesized_fallback(
    allow_model_requests: None, openai_api_key: str, vcr: Cassette
) -> None:
    """An unsupported model receives the established synthesized search exchange."""
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    tool = refund_tool()

    messages = model.prepare_messages(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Call lookup_refund_policy with order_id order-123. Do not answer directly.')
                ]
            ),
            ModelResponse(parts=[TextPart(content='I will load the refund capability.')]),
            ModelRequest(parts=[ToolAvailabilityDeltaPart(added=[tool.name], tool_call_id='load-refunds')]),
        ]
    )
    response = await model.request(
        messages,
        None,
        ModelRequestParameters(function_tools=[tool]),
    )

    assert any(isinstance(part, ToolCallPart) and part.tool_name == tool.name for part in response.parts)
    body = single_request_body(vcr)
    assert all(item.get('type') != 'additional_tools' for item in body['input'])
    assert body['input'][-2:] == snapshot(
        [
            {
                'type': 'function_call',
                'name': 'search_tools',
                'arguments': '{"queries":["lookup_refund_policy"]}',
                'call_id': 'load-refunds',
            },
            {
                'type': 'function_call_output',
                'call_id': 'load-refunds',
                'output': '{"discovered_tools":[{"name":"lookup_refund_policy"}]}',
            },
        ]
    )
