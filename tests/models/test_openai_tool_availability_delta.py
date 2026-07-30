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
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile
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


def test_removal_raises_when_unsupported() -> None:
    """OpenAI's native item only adds tools, so a removal must fail rather than disappear."""
    model = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(api_key='test-key'))

    with pytest.raises(
        UserError,
        match=r"Model 'gpt-5\.6' cannot withdraw tools \['old_refund_tool'\]: tool removal is not supported\.",
    ):
        model.prepare_messages(
            [
                ModelRequest(
                    parts=[
                        ToolAvailabilityDeltaPart(
                            added=['lookup_refund_policy'],
                            removed=['old_refund_tool'],
                            tool_call_id='load-refunds',
                        )
                    ]
                )
            ]
        )


async def test_unsupported_model_raises_rather_than_emitting_the_item() -> None:
    """A delta reaching a model without native support names the step the caller missed.

    `prepare_messages` projects the delta onto the local tool-search exchange for every model outside the
    supported list, so only adapters that asked for the native item should see the part. `Model.request` is
    public and skips that projection, so a caller driving a model directly reaches this with a history that
    is otherwise perfectly valid — hence a `UserError` naming the missing call rather than an assertion
    about an internal invariant.

    Raising beats emitting the item anyway: this path removes the revealed tool from top-level `tools`, so
    quietly sending an item whose support we haven't verified is how an availability change goes missing.

    Every model on the first-party provider now supports the item, so the model here is one reached
    through an OpenAI-compatible endpoint instead — which is the shape that actually keeps the flag: those
    speak the Responses API without necessarily implementing this item.
    """
    model = OpenAIResponsesModel(
        'gpt-5.6',
        provider=OpenAIProvider(api_key='test-key'),
        # An explicit profile stands in for an OpenAI-compatible deployment: the provider enables the
        # flag for every model, and only a profile saying otherwise turns it off.
        profile=merge_profile(
            openai_model_profile('gpt-5.6'),
            OpenAIModelProfile(openai_responses_supports_tool_availability_delta=False),
        ),
    )
    assert model.profile.get('openai_responses_supports_tool_availability_delta', False) is False

    with pytest.raises(UserError, match='prepare_messages'):
        await model._map_messages(  # pyright: ignore[reportPrivateUsage]
            [ModelRequest(parts=[ToolAvailabilityDeltaPart(added=['lookup_refund_policy'])])],
            OpenAIResponsesModelSettings(),
            ModelRequestParameters(function_tools=[refund_tool()]),
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
    """An endpoint without `additional_tools` receives the established synthesized search exchange.

    Every model on the first-party OpenAI provider takes the native item now, so the case this covers is
    an OpenAI-compatible deployment — Azure, OpenRouter, vLLM — that speaks the Responses API without
    implementing it. The profile is what says so, and the wire is identical either way, which is why the
    recording made against `gpt-5` before the model gate was dropped still describes it exactly.
    """
    model = OpenAIResponsesModel(
        'gpt-5',
        provider=OpenAIProvider(api_key=openai_api_key),
        profile=merge_profile(
            openai_model_profile('gpt-5'),
            OpenAIModelProfile(openai_responses_supports_tool_availability_delta=False),
        ),
    )
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
