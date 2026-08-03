from __future__ import annotations

from typing import Any, cast

import pytest
from inline_snapshot import snapshot
from vcr.cassette import Cassette

from pydantic_ai import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ToolSearchReturnPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.native_tools._tool_search import ToolSearchTool
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


async def test_empty_local_search_return_does_not_emit_additional_tools() -> None:
    """A fruitless local search has no schemas to append to the Responses input."""
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key='test-key'))

    _, items = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [
            ModelRequest(
                parts=[
                    ToolSearchReturnPart(
                        content={'discovered_tools': [], 'message': 'No matching tools found.'},
                        tool_call_id='search-1',
                    )
                ]
            )
        ],
        OpenAIResponsesModelSettings(),
        ModelRequestParameters(),
    )

    assert [item.get('type') for item in items] == ['function_call_output']


async def test_item_carried_tool_call_gets_a_synthesized_namespace() -> None:
    """A tool a delta introduces travels in `additional_tools`, so its replayed calls are namespaced.

    The delta drops the tool's entry from the wire `tools` array in favor of the item declaration,
    and OpenAI rejects a call to an item-declared tool without a namespace as "does not exist in the
    default namespace".
    """
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key='test-key'))
    tool = ToolDefinition(name='foo', parameters_json_schema={'type': 'object', 'properties': {}}, defer_loading=True)
    _, parameters = model.prepare_request(
        None, ModelRequestParameters(function_tools=[tool], revealed_tool_names={tool.name})
    )

    _, items = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [
            ModelRequest(parts=[ToolAvailabilityDeltaPart(added=[tool.name])]),
            ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args={}, tool_call_id='call-1')]),
        ],
        OpenAIResponsesModelSettings(),
        parameters,
    )

    function_call = items[-1]
    assert function_call.get('type') == 'function_call'
    assert function_call.get('namespace') == tool.name


async def test_stored_reveal_does_not_namespace_plain_tool_call() -> None:
    """Reveal state alone cannot move an ordinary `tools`-array function out of the default namespace.

    Here the tool's name is in `revealed_tool_names` (e.g. restored run state) but nothing in this
    request's messages introduces it through an item, so it occupies a plain `tools` entry — tagging
    its replayed call with a namespace would be the inverse of the mismatch OpenAI rejects.
    """
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key='test-key'))
    tool = ToolDefinition(name='foo', parameters_json_schema={'type': 'object', 'properties': {}})
    parameters = ModelRequestParameters(function_tools=[tool], revealed_tool_names={tool.name})

    _, items = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [
            ModelRequest(parts=[UserPromptPart(content='Call foo.')]),
            ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args={}, tool_call_id='call-1')]),
        ],
        OpenAIResponsesModelSettings(),
        parameters,
    )

    function_call = items[-1]
    assert function_call.get('type') == 'function_call'
    assert 'namespace' not in function_call


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
            OpenAIModelProfile(tool_additions=None),
        ),
    )
    assert model.profile.get('tool_additions') is None

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
    # The same parameters the request goes out with: `prepare_messages` decides how to render the
    # delta from the tools it's told about, so handing it a different set than `request` gets is how
    # a caller ends up with a rendering that describes a request it never sends.
    parameters = ModelRequestParameters(function_tools=[tool])

    messages = model.prepare_messages(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Call lookup_refund_policy with order_id order-123. Do not answer directly.')
                ]
            ),
            ModelResponse(parts=[TextPart(content='I will load the refund capability.')]),
            ModelRequest(parts=[ToolAvailabilityDeltaPart(added=[tool.name])]),
        ],
        parameters,
    )
    response = await model.request(messages, None, parameters)

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


async def test_unsupported_model_calls_the_tool_the_announcement_revealed(
    allow_model_requests: None, openai_api_key: str, vcr: Cassette
) -> None:
    """An endpoint without `additional_tools` receives a mid-conversation availability announcement.

    Every model on the first-party OpenAI provider takes the native item now, so the case this covers is
    an OpenAI-compatible deployment — Azure, OpenRouter, vLLM — that speaks the Responses API without
    implementing it. The profile is what says so. Because Responses cannot hide schemas without native
    tool search, the revealed tool is already callable and the fallback only needs to announce it.
    """
    model = OpenAIResponsesModel(
        'gpt-5',
        provider=OpenAIProvider(api_key=openai_api_key),
        profile=merge_profile(
            openai_model_profile('gpt-5'),
            OpenAIModelProfile(tool_additions=None),
        ),
    )
    tool = refund_tool()
    # The same parameters the request goes out with: `prepare_messages` decides how to render the
    # delta from the tools it's told about, so handing it a different set than `request` gets is how
    # a caller ends up with a rendering that describes a request it never sends.
    parameters = ModelRequestParameters(function_tools=[tool])

    messages = model.prepare_messages(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Call lookup_refund_policy with order_id order-123. Do not answer directly.')
                ]
            ),
            ModelResponse(parts=[TextPart(content='I will load the refund capability.')]),
            ModelRequest(parts=[ToolAvailabilityDeltaPart(added=[tool.name], tool_call_id='load-refunds')]),
        ],
        parameters,
    )
    response = await model.request(messages, None, parameters)

    assert any(isinstance(part, ToolCallPart) and part.tool_name == tool.name for part in response.parts)
    body = single_request_body(vcr)
    assert all(item.get('type') != 'additional_tools' for item in body['input'])
    assert body['input'][-2:] == snapshot(
        [
            {'role': 'assistant', 'content': 'I will load the refund capability.'},
            {'role': 'system', 'content': 'The following tools have become available to you: `lookup_refund_policy`.'},
        ]
    )


async def test_openai_live_delta_preserves_the_warmed_cache_prefix(
    allow_model_requests: None, openai_api_key: str, vcr: Cassette
) -> None:
    """Two live `gpt-5.6` requests let the cassette prefix checker guard a delta after a warmed turn."""
    model = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(api_key=openai_api_key))
    tool = refund_tool()
    tool.defer_loading = True
    tool.with_native = ToolSearchTool.kind
    parameters = ModelRequestParameters(function_tools=[tool], native_tools=[ToolSearchTool()])
    before: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='Reply only with: ready')])]

    warm_response = await model.request(before, None, parameters)
    after = [*before, warm_response, ModelRequest(parts=[ToolAvailabilityDeltaPart(added=[tool.name])])]
    await model.request(after, None, parameters)

    assert len(cast(Any, vcr).requests) == 2
