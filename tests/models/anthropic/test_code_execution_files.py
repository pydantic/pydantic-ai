"""Tests for Anthropic `CodeExecutionTool.files` uploads, and for container reuse and recovery."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx2
import pytest
from inline_snapshot import snapshot
from pydantic import JsonValue

from pydantic_ai import Agent, ModelHTTPError
from pydantic_ai.capabilities import AbstractCapability, Citations, NativeTool
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    TextPart,
    UploadedFile,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import AbstractNativeTool, CodeExecutionTool, WebFetchTool, WebSearchTool

from ...conftest import RequestCapture, message, try_import
from ..conftest import AnthropicModelFactory, cache_breakpoints, content_blocks, message_shape

with try_import() as anthropic_imports_successful:
    from anthropic import (
        APIStatusError,
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
        omit as OMIT,
    )
    from anthropic.types.beta import BetaTextBlock, BetaUsage
    from anthropic.types.beta.beta_container_params import BetaContainerParams

    from pydantic_ai.models.anthropic import (
        AnthropicModel,
        AnthropicModelSettings,
        AnthropicStaleThinkingBlockWarning,
    )
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from ..test_anthropic import (
        MockAnthropic,
        completion_message,
        get_mock_chat_completion_kwargs,
    )
    from .test_thinking_block_binding import stale_thinking_block_error

pytestmark = [
    pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed'),
]


async def test_anthropic_request_projection_shapes():
    """The shared wire projections handle string content and cache breakpoints."""
    body: dict[str, JsonValue] = {
        'cache_control': {'type': 'ephemeral'},
        'system': [{'type': 'text', 'cache_control': {'type': 'ephemeral'}}],
        'messages': [
            {'role': 'user', 'content': 'hello'},
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': 'hi', 'cache_control': {'type': 'ephemeral'}}],
            },
        ],
    }

    assert (
        content_blocks(body, 'text'),
        message_shape(body),
        cache_breakpoints(body),
    ) == snapshot(
        (
            [{'type': 'text', 'text': 'hi', 'cache_control': {'type': 'ephemeral'}}],
            [('user', ['<str>']), ('assistant', ['text'])],
            (
                {'type': 'ephemeral'},
                ['system[0]', 'messages[1].content[0]'],
            ),
        )
    )


async def test_anthropic_code_execution_files_500_without_uploads_is_not_retried(allow_model_requests: None):
    """A 500 on a request carrying a history-resolved container id but no uploads raises as-is, with no second attempt.

    Not a VCR test: a container the API will not accept, with no `container_upload` in play, answers
    200 with an `unavailable` tool result, so this 500 shape can only be simulated. Pins that the container-drop
    retry needs *both* halves of the shape that actually 500s — an id we resolved from history *and*
    uploads on the wire — so an unrelated 500 never costs a caller a duplicate request. The mock
    would answer a second attempt with another 500, so a retry that fired would show up as a second
    request.
    """
    error = APIStatusError(
        'server error',
        response=httpx2.Response(status_code=500, request=httpx2.Request('POST', 'https://example.com/v1')),
        body={'error': 'server error'},
    )
    mock_client = MockAnthropic.create_mock([error, error])
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model, capabilities=[NativeTool(CodeExecutionTool())])
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Earlier turn.')]),
        ModelResponse(
            parts=[TextPart(content='Earlier answer.')],
            provider_name='anthropic',
            provider_details={'container_id': 'container_01EG1LKXFPoQJ9tpbsZ1dh74'},
        ),
    ]

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('hello', message_history=history)

    assert exc_info.value.status_code == 500
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)
    assert [kwargs['container'] for kwargs in completion_kwargs] == ['container_01EG1LKXFPoQJ9tpbsZ1dh74']


# The cases carry raw container values rather than built `AnthropicModelSettings`: this list is
# evaluated at import, and the settings type is undefined when anthropic isn't installed.
_PINNED_CONTAINER_WITH_SKILLS: BetaContainerParams = {
    'id': 'container_PINNED',
    'skills': [{'type': 'anthropic', 'skill_id': 'xlsx', 'version': 'latest'}],
}
_PAUSED_TURN_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content='Use the attached file.')]),
    ModelResponse(
        parts=[TextPart(content='Working on it.')],
        state='suspended',
        provider_name='anthropic',
        provider_details={'container_id': 'container_PAUSED'},
    ),
]


@pytest.mark.parametrize(
    'container,message_history,expected_container',
    [
        pytest.param('container_PINNED', None, 'container_PINNED', id='bare-id'),
        pytest.param({'id': 'container_PINNED'}, None, 'container_PINNED', id='id-only-dict'),
        pytest.param(_PINNED_CONTAINER_WITH_SKILLS, None, _PINNED_CONTAINER_WITH_SKILLS, id='dict-with-skills'),
        pytest.param(None, _PAUSED_TURN_HISTORY, 'container_PAUSED', id='pause-turn-reconnect'),
    ],
)
async def test_anthropic_code_execution_files_500_keeps_caller_container(
    allow_model_requests: None,
    container: BetaContainerParams | str | None,
    message_history: list[ModelMessage] | None,
    expected_container: BetaContainerParams | str,
):
    """A container the caller chose survives the 500 unchanged, uploads on the wire or not.

    Only an id *we* resolved from history may be dropped. Dropping a caller's container discards
    state they asked us to keep, and it would not self-heal: `_get_container` prefers their setting
    over the fresh id a retry earns, so the dropped id would come straight back on the next step. A
    `pause_turn` reconnect id is the caller's too — it is the turn they are resuming. The mock would
    answer a second attempt with another 500, so a retry that fired would show up as a second
    request.
    """
    error = APIStatusError(
        'server error',
        response=httpx2.Response(status_code=500, request=httpx2.Request('POST', 'https://example.com/v1')),
        body={'error': 'server error'},
    )
    mock_client = MockAnthropic.create_mock([error, error])
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    settings: AnthropicModelSettings = {} if container is None else {'anthropic_container': container}
    agent = Agent(
        model,
        capabilities=[NativeTool(CodeExecutionTool(files=[UploadedFile(file_id='file_x', provider_name='anthropic')]))],
        model_settings=settings,
    )
    if message_history is not None:
        reloaded_message_history = ModelMessagesTypeAdapter.validate_python(
            ModelMessagesTypeAdapter.dump_python(message_history, mode='json')
        )
        assert reloaded_message_history == message_history
    else:
        reloaded_message_history = None

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run(
            None if reloaded_message_history else 'Use the attached file.', message_history=reloaded_message_history
        )

    assert exc_info.value.status_code == 500
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)
    assert [kwargs['container'] for kwargs in completion_kwargs] == [expected_container]


async def test_anthropic_code_execution_files_suspended_history_without_container(allow_model_requests: None):
    """A suspended response without a container sends no container and does not enable recovery."""
    error = APIStatusError(
        'server error',
        response=httpx2.Response(status_code=500, request=httpx2.Request('POST', 'https://example.com/v1')),
        body={'error': 'server error'},
    )
    mock_client = MockAnthropic.create_mock([error, error])
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        model,
        capabilities=[NativeTool(CodeExecutionTool(files=[UploadedFile(file_id='file_x', provider_name='anthropic')]))],
    )
    history: list[ModelMessage] = [
        ModelResponse(parts=[TextPart(content='Working on it.')], state='suspended', provider_name='anthropic')
    ]

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run(None, message_history=history)

    assert exc_info.value.status_code == 500
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)
    assert [kwargs['container'] for kwargs in completion_kwargs] == [OMIT]


# `expect_retry` rather than the expected container list: the list would have to name `OMIT`, which
# comes from the anthropic SDK, and a parametrize decorator is evaluated at import — so the module
# would fail to collect in the CI variants that install pydantic-ai without anthropic.
@pytest.mark.parametrize(
    'status_code,expect_retry',
    [
        pytest.param(500, True, id='500-retried'),
        pytest.param(429, False, id='429-not-retried'),
    ],
)
async def test_anthropic_code_execution_files_500_with_uploads_drops_history_container(
    allow_model_requests: None, status_code: int, expect_retry: bool
):
    """A 500 on a history-resolved id with uploads on the wire is resent once, carrying no container at all.

    A successful retry is covered live by
    `test_code_execution_files_vcr.py::test_anthropic_code_execution_files_rejected_container_is_dropped_and_retried`;
    what this adds is the one-shot bound. Every attempt fails here, so the second request shows the
    retry drops the container and the third that never comes shows the drop is not a loop — the
    retry's own error is what surfaces. (`MockAnthropic` cannot answer the retry with a success: it
    only advances its response index on the way out, so an exception in a sequence is re-raised
    forever.)

    The `429` case is what pins the status half of the guard, and coverage cannot stand in for it:
    the `e.status_code != 500` operand shares its line with `not container_from_history`, which the
    caller-container test already takes, so the line reads fully covered while the non-500 shape goes
    unvisited. Without this param, deleting that operand leaves the suite green — and a rate-limited
    request on a resumed code-execution conversation would be silently duplicated, surfacing the
    retry's error in place of the original's `retry-after`.
    """
    error = APIStatusError(
        'server error',
        response=httpx2.Response(status_code=status_code, request=httpx2.Request('POST', 'https://example.com/v1')),
        body={'error': 'server error'},
    )
    mock_client = MockAnthropic.create_mock(error)
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        model,
        capabilities=[NativeTool(CodeExecutionTool(files=[UploadedFile(file_id='file_x', provider_name='anthropic')]))],
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Use the attached file.')]),
        ModelResponse(
            parts=[TextPart(content='Earlier answer.')],
            provider_name='anthropic',
            provider_details={'container_id': 'container_from_history'},
        ),
    ]

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('And now summarize it.', message_history=history)

    assert exc_info.value.status_code == status_code
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)
    expected_containers: list[object] = ['container_from_history', OMIT] if expect_retry else ['container_from_history']
    assert [kwargs['container'] for kwargs in completion_kwargs] == expected_containers


async def test_anthropic_code_execution_files_500_then_stale_thinking_block_still_retries(
    allow_model_requests: None,
):
    """A stale thinking block rejected only by the container fallback still reaches the drop retry.

    The fallback runs inside the handler that catches the original 500, so its own error used to
    propagate without ever being classified — the run failed on a 400 the retry exists to absorb.
    The third request is what pins the other half: the container the 500 disowned stays dropped,
    rather than being resent by a retry that reads the container the first attempt used.
    """
    server_error = APIStatusError(
        'server error',
        response=httpx2.Response(status_code=500, request=httpx2.Request('POST', 'https://example.com/v1')),
        body={'error': 'server error'},
    )
    mock_client = MockAnthropic.create_mock(
        [
            server_error,
            stale_thinking_block_error(),
            completion_message(
                [BetaTextBlock(text='Summarized.', type='text')], usage=BetaUsage(input_tokens=1, output_tokens=1)
            ),
        ]
    )
    model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        model,
        capabilities=[NativeTool(CodeExecutionTool(files=[UploadedFile(file_id='file_x', provider_name='anthropic')]))],
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Use the attached file.')]),
        ModelResponse(
            parts=[TextPart(content='Earlier answer.')],
            provider_name='anthropic',
            provider_details={'container_id': 'container_from_history'},
        ),
    ]

    with pytest.warns(AnthropicStaleThinkingBlockWarning):
        result = await agent.run('And now summarize it.', message_history=history)

    assert result.output == snapshot('Summarized.')
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)
    expected_containers: list[object] = ['container_from_history', OMIT, OMIT]
    assert [kwargs['container'] for kwargs in completion_kwargs] == expected_containers
    assert completion_kwargs[-1]['extra_body'] == snapshot(
        {'thinking': {'type': 'adaptive', 'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}}
    )


async def test_anthropic_code_execution_files_container_fallback_surfaces_its_error(
    allow_model_requests: None,
):
    """An unrelated error from the no-container fallback replaces the original container 500."""
    server_error = APIStatusError(
        'server error',
        response=httpx2.Response(status_code=500, request=httpx2.Request('POST', 'https://example.com/v1')),
        body={'error': 'server error'},
    )
    rate_limit_error = APIStatusError(
        'rate limited',
        response=httpx2.Response(status_code=429, request=httpx2.Request('POST', 'https://example.com/v1')),
        body={'error': 'rate limited'},
    )
    mock_client = MockAnthropic.create_mock([server_error, rate_limit_error])
    model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        model,
        capabilities=[NativeTool(CodeExecutionTool(files=[UploadedFile(file_id='file_x', provider_name='anthropic')]))],
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Use the attached file.')]),
        ModelResponse(
            parts=[TextPart(content='Earlier answer.')],
            provider_name='anthropic',
            provider_details={'container_id': 'container_from_history'},
        ),
    ]

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('And now summarize it.', message_history=history)

    assert exc_info.value.status_code == 429
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)
    assert [kwargs['container'] for kwargs in completion_kwargs] == ['container_from_history', OMIT]


async def test_anthropic_code_execution_files_append_to_every_user_message(allow_model_requests: None):
    """Pins the internal `_map_message` placement: uploads attach to *every* user message that can carry one (covering all of them reaches the turn being generated while keeping each byte-identical as history grows), and none are added when history has no user message.

    Three user messages, not two: first-and-last is the same set as every-user-message on a
    two-turn history, so a two-turn snapshot cannot catch a regression that only tags the ends.

    Not a VCR test: the recordings in `test_code_execution_files_vcr.py` do catch a placement
    regression — they read the outbound request through the `request_capture` hook — but they
    are all two-user-turn histories, and the no-user-message branch is unreachable through an
    agent run, which needs a prompt, so asserting the mapped messages directly is what covers it.
    """
    c = completion_message([BetaTextBlock(text='Response', type='text')], BetaUsage(input_tokens=10, output_tokens=5))
    mock_client = MockAnthropic.create_mock(c)
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    parameters = ModelRequestParameters(
        native_tools=[
            CodeExecutionTool(files=[UploadedFile(file_id='file_anthropic', provider_name='anthropic')]),
        ]
    )

    _, messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        [
            ModelRequest(parts=[UserPromptPart(content='Use the attached file.')]),
            ModelResponse(parts=[TextPart(content='Previous response')]),
            ModelRequest(parts=[UserPromptPart(content='And now summarize it.')]),
            ModelResponse(parts=[TextPart(content='Summary so far')]),
            ModelRequest(parts=[UserPromptPart(content='Now the average.')]),
        ],
        parameters,
        AnthropicModelSettings(),
    )

    assert messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': 'Use the attached file.', 'type': 'text'},
                    {'file_id': 'file_anthropic', 'type': 'container_upload'},
                ],
            },
            {'role': 'assistant', 'content': [{'text': 'Previous response', 'type': 'text'}]},
            {
                'role': 'user',
                'content': [
                    {'text': 'And now summarize it.', 'type': 'text'},
                    {'file_id': 'file_anthropic', 'type': 'container_upload'},
                ],
            },
            {'role': 'assistant', 'content': [{'text': 'Summary so far', 'type': 'text'}]},
            {
                'role': 'user',
                'content': [
                    {'text': 'Now the average.', 'type': 'text'},
                    {'file_id': 'file_anthropic', 'type': 'container_upload'},
                ],
            },
        ]
    )

    _, messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        [ModelResponse(parts=[TextPart(content='Previous response')])],
        parameters,
        AnthropicModelSettings(),
    )

    assert messages == snapshot([{'role': 'assistant', 'content': [{'text': 'Previous response', 'type': 'text'}]}])


@dataclass(frozen=True)
class HistoryContainerCase:
    id: str
    client: str
    model_name: str
    container_sent: bool
    suspended: bool = False
    native_tools: list[AbstractNativeTool] = field(default_factory=list[AbstractNativeTool])
    model_settings: AnthropicModelSettings | None = None
    expected_tools: list[tuple[str, list[str] | None]] = field(default_factory=list[tuple[str, list[str] | None]])


_CODE_EXECUTION_WIRE_TOOL: dict[str, object] = {'type': 'code_execution_20250825', 'name': 'code_execution'}

HISTORY_CONTAINER_CASES = [
    HistoryContainerCase('api', 'api', 'claude-sonnet-4-6', container_sent=False),
    HistoryContainerCase('bedrock', 'bedrock', 'us.anthropic.claude-sonnet-4-6', container_sent=False),
    HistoryContainerCase('vertex', 'vertex', 'claude-sonnet-4-6', container_sent=False),
    HistoryContainerCase('mantle', 'mantle', 'anthropic.claude-sonnet-4-6', container_sent=True),
    HistoryContainerCase('foundry', 'foundry', 'claude-sonnet-4-6', container_sent=False),
    HistoryContainerCase('api-suspended', 'api', 'claude-sonnet-4-6', container_sent=False, suspended=True),
    HistoryContainerCase(
        'mantle-suspended', 'mantle', 'anthropic.claude-sonnet-4-6', container_sent=True, suspended=True
    ),
    HistoryContainerCase(
        'api-extra-body-code-execution',
        'api',
        'claude-sonnet-4-6',
        container_sent=True,
        suspended=True,
        model_settings={'extra_body': {'tools': [_CODE_EXECUTION_WIRE_TOOL]}},
        expected_tools=[('code_execution_20250825', None)],
    ),
    HistoryContainerCase(
        'api-extra-body-replaces-code-execution',
        'api',
        'claude-sonnet-4-6',
        container_sent=False,
        native_tools=[CodeExecutionTool()],
        model_settings={'extra_body': {'tools': [{'type': 'web_search_20250305', 'name': 'web_search'}]}},
        expected_tools=[('web_search_20250305', None)],
    ),
    HistoryContainerCase(
        'api-extra-body-newer-code-execution',
        'api',
        'claude-sonnet-4-6',
        container_sent=True,
        suspended=True,
        model_settings={'extra_body': {'tools': [{'type': 'code_execution_20260521', 'name': 'code_execution'}]}},
        expected_tools=[('code_execution_20260521', None)],
    ),
    HistoryContainerCase(
        'api-extra-body-newer-web-fetch',
        'api',
        'claude-sonnet-4-6',
        container_sent=True,
        model_settings={'extra_body': {'tools': [{'type': 'web_fetch_20260309', 'name': 'web_fetch'}]}},
        expected_tools=[('web_fetch_20260309', None)],
    ),
    HistoryContainerCase(
        'api-extra-body-dict-values',
        'api',
        'claude-sonnet-4-6',
        container_sent=True,
        model_settings={'extra_body': {'tools': {'tool': _CODE_EXECUTION_WIRE_TOOL}.values()}},
        expected_tools=[('code_execution_20250825', None)],
    ),
    HistoryContainerCase(
        'api-extra-body-frozenset-callers',
        'api',
        'claude-sonnet-4-6',
        container_sent=True,
        model_settings={
            'extra_body': {
                'tools': [
                    {
                        'type': 'web_search_20250305',
                        'name': 'web_search',
                        'allowed_callers': frozenset({'code_execution_20260120'}),
                    }
                ]
            }
        },
        expected_tools=[('web_search_20250305', ['code_execution_20260120'])],
    ),
    HistoryContainerCase(
        'api-empty-callers',
        'api',
        'claude-sonnet-4-6',
        container_sent=False,
        native_tools=[WebSearchTool()],
        model_settings={'anthropic_web_tools_allowed_callers': []},
        expected_tools=[('web_search_20260209', [])],
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in HISTORY_CONTAINER_CASES])
async def test_anthropic_history_container_without_code_execution_by_client(
    allow_model_requests: None, case: HistoryContainerCase
):
    """A request without code execution doesn't send the history container where the API is known to reject it.

    That includes the container of a suspended response the run resumes. The direct API rejects it. The Bedrock
    Messages API hasn't been verified, so it keeps sending it, and the legacy Bedrock, Vertex AI and Foundry clients
    don't send container ids at all. Tools in `extra_body` replace the built ones on the wire, so they decide whether
    the request has code execution.

    Unit test: recording the other clients needs their credentials, recording a suspended response needs a live
    `pause_turn`, and the claim is the request shape.
    """
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                'id': 'msg_1',
                'type': 'message',
                'role': 'assistant',
                'model': case.model_name,
                'content': [{'type': 'text', 'text': 'OK'}],
                'stop_reason': 'end_turn',
                'usage': {'input_tokens': 5, 'output_tokens': 1},
            },
        )

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Earlier turn.')]),
        ModelResponse(
            parts=[TextPart(content='Earlier answer.')],
            provider_name='anthropic',
            provider_details={'container_id': 'container_from_history'},
            state='suspended' if case.suspended else 'complete',
        ),
    ]
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        clients: dict[str, Callable[[], Any]] = {
            'api': lambda: AsyncAnthropic(api_key='x', http_client=http_client),
            'bedrock': lambda: AsyncAnthropicBedrock(api_key='x', aws_region='us-east-1', http_client=http_client),
            'mantle': lambda: AsyncAnthropicBedrockMantle(api_key='x', aws_region='us-east-1', http_client=http_client),
            'vertex': lambda: AsyncAnthropicVertex(
                project_id='p', region='us-east5', access_token='x', http_client=http_client
            ),
            'foundry': lambda: AsyncAnthropicFoundry(
                api_key='x', base_url='https://example.com/anthropic', http_client=http_client
            ),
        }
        model = AnthropicModel(case.model_name, provider=AnthropicProvider(anthropic_client=clients[case.client]()))
        capabilities: list[AbstractCapability[None]] = [NativeTool(tool) for tool in case.native_tools]
        await Agent(model, deps_type=type(None), capabilities=capabilities, model_settings={'max_tokens': 1024}).run(
            None if case.suspended else 'Follow up.', message_history=history, model_settings=case.model_settings
        )

    [body] = bodies
    tools: list[dict[str, Any]] = body.get('tools', [])
    container_sent = 'container' in body
    assert container_sent == case.container_sent
    assert [(tool['type'], tool.get('allowed_callers')) for tool in tools] == case.expected_tools


def _pause_turn_segment(blocks: list[dict[str, Any]], stop_reason: str, stream: bool) -> httpx2.Response:
    """One segment of a turn whose code execution container is `container_A`."""
    container = {'id': 'container_A', 'expires_at': '2026-11-01T00:00:00Z'}
    usage = {'input_tokens': 5, 'output_tokens': 1}
    message = {
        'id': f'msg_{stop_reason}',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-6',
        'usage': usage,
        'container': container,
    }
    if not stream:
        return httpx2.Response(200, json={**message, 'content': blocks, 'stop_reason': stop_reason})
    events: list[dict[str, Any]] = [
        {'type': 'message_start', 'message': {**message, 'content': [], 'stop_reason': None}}
    ]
    for index, block in enumerate(blocks):
        if block['type'] == 'text':
            start: dict[str, Any] = {'type': 'text', 'text': ''}
            delta: dict[str, Any] | None = {'type': 'text_delta', 'text': block['text']}
        elif block['type'] == 'server_tool_use':
            start = {**block, 'input': {}}
            delta = {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}
        else:
            start, delta = block, None
        events.append({'type': 'content_block_start', 'index': index, 'content_block': start})
        if delta is not None:
            events.append({'type': 'content_block_delta', 'index': index, 'delta': delta})
        events.append({'type': 'content_block_stop', 'index': index})
    events += [
        {'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage},
        {'type': 'message_stop'},
    ]
    body = ''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events)
    return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=body.encode())


@pytest.mark.parametrize('stream', [pytest.param(False, id='request'), pytest.param(True, id='stream')])
@pytest.mark.parametrize(
    'native_tools,extra_body_tool,server_tool_call',
    [
        pytest.param([], _CODE_EXECUTION_WIRE_TOOL, ('code_execution', {'code': 'print(1)'}), id='code-execution'),
        pytest.param(
            [CodeExecutionTool()],
            {'type': 'code_execution_20260521', 'name': 'code_execution'},
            ('bash_code_execution', {'command': 'ls'}),
            id='newer-code-execution-bash',
        ),
        pytest.param(
            [WebSearchTool()],
            {'type': 'web_search_20260318', 'name': 'web_search'},
            ('code_execution', {'code': 'print(1)'}),
            id='newer-web-search',
        ),
    ],
)
async def test_anthropic_pause_turn_reconnect_keeps_extra_body_code_execution_container(
    allow_model_requests: None,
    stream: bool,
    native_tools: list[AbstractNativeTool],
    extra_body_tool: dict[str, object],
    server_tool_call: tuple[str, dict[str, object]],
):
    """A `pause_turn` reconnect keeps code execution that reaches the wire through `extra_body`, at any version.

    The reconnect sends the container and replays the paused turn's in-flight sandbox call.

    Unit test: a live `pause_turn` with a container can't be requested reliably, and the claim is the request shape.
    """
    name, input = server_tool_call
    bodies: list[dict[str, Any]] = []
    segments = iter(
        [
            ([{'type': 'server_tool_use', 'id': 'srvtoolu_1', 'name': name, 'input': input}], 'pause_turn'),
            ([{'type': 'text', 'text': 'Done.'}], 'end_turn'),
        ]
    )

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return _pause_turn_segment(*next(segments), stream=stream)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        model = AnthropicModel(
            'claude-sonnet-4-6',
            provider=AnthropicProvider(anthropic_client=AsyncAnthropic(api_key='x', http_client=http_client)),
        )
        settings: AnthropicModelSettings = {'max_tokens': 1024, 'extra_body': {'tools': [extra_body_tool]}}
        agent = Agent(model, capabilities=[NativeTool(tool) for tool in native_tools], model_settings=settings)
        if stream:
            async with agent.run_stream('Run the code.') as result:
                output = await result.get_output()
        else:
            output = (await agent.run('Run the code.')).output

    assert output == 'Done.'
    assert [(body.get('container'), body['tools']) for body in bodies] == [
        (None, [extra_body_tool]),
        ('container_A', [extra_body_tool]),
    ]
    assert [
        block['name']
        for message in bodies[1]['messages']
        if message['role'] == 'assistant'
        for block in message['content']
        if block['type'] == 'server_tool_use'
    ] == [name]


_PENDING_CODE_EXECUTION_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content='Search the web for X.')]),
    ModelResponse(
        parts=[
            TextPart(content='Searching.'),
            NativeToolCallPart(
                tool_name='code_execution',
                args={'code': 'r = await web_search({"query": "X"})'},
                tool_call_id='srvtoolu_pending',
                provider_name='anthropic',
            ),
        ],
        provider_name='anthropic',
        provider_details={'container_id': 'container_A'},
        state='suspended',
    ),
]


@pytest.mark.parametrize('stream', [pytest.param(False, id='request'), pytest.param(True, id='stream')])
@pytest.mark.parametrize(
    'native_tool,container,history,expected_container',
    [
        pytest.param(WebSearchTool(), 'container_A', None, 'container_A', id='explicit-container-web-search'),
        pytest.param(WebFetchTool(), 'container_A', None, 'container_A', id='explicit-container-web-fetch'),
        pytest.param(
            WebSearchTool(),
            {'skills': [{'type': 'anthropic', 'skill_id': 'xlsx', 'version': 'latest'}]},
            None,
            {'skills': [{'type': 'anthropic', 'skill_id': 'xlsx', 'version': 'latest'}]},
            id='skills-container',
        ),
        pytest.param(
            WebSearchTool(), None, _PENDING_CODE_EXECUTION_HISTORY, 'container_A', id='resumed-pending-code-execution'
        ),
    ],
)
async def test_anthropic_citations_keep_default_web_tool_callers_when_the_request_needs_code_execution(
    allow_model_requests: None,
    stream: bool,
    native_tool: AbstractNativeTool,
    container: BetaContainerParams | str | None,
    history: list[ModelMessage] | None,
    expected_container: BetaContainerParams | str,
):
    """`Citations()` leaves the web tools on Anthropic's default caller when the request depends on code execution.

    An explicit or Skills container, or a resumed turn whose code execution call is still pending, needs the
    sandbox: routing the web tools to direct calls would leave a container on a request without code execution,
    which Anthropic rejects. Their results come back uncited, as they do without this setting. The response's sandbox
    calls are kept, streamed too.

    Unit test: the claim is the request shape and the response filter, and recording these needs a live container.
    """
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        blocks: list[dict[str, Any]] = [
            {'type': 'server_tool_use', 'id': 'srvtoolu_1', 'name': 'code_execution', 'input': {'code': 'print(1)'}},
            {
                'type': 'code_execution_tool_result',
                'tool_use_id': 'srvtoolu_1',
                'content': {
                    'type': 'code_execution_result',
                    'stdout': '1',
                    'stderr': '',
                    'return_code': 0,
                    'content': [],
                },
            },
            {'type': 'text', 'text': 'Done.'},
        ]
        return _pause_turn_segment(blocks, 'end_turn', stream=stream)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        model = AnthropicModel(
            'claude-sonnet-4-6',
            provider=AnthropicProvider(anthropic_client=AsyncAnthropic(api_key='x', http_client=http_client)),
        )
        settings: AnthropicModelSettings = {'max_tokens': 1024}
        if container is not None:
            settings['anthropic_container'] = container
        agent = Agent(model, capabilities=[NativeTool(native_tool), Citations()], model_settings=settings)
        prompt = None if history else 'Look it up.'
        if stream:
            async with agent.run_stream(prompt, message_history=history) as result:
                await result.get_output()
                messages = result.all_messages()
        else:
            messages = (await agent.run(prompt, message_history=history)).all_messages()

    [body] = bodies
    assert [tool.get('allowed_callers') for tool in body['tools']] == [None]
    assert body['container'] == expected_container
    response = message(messages, ModelResponse, index=-1)
    assert [part.tool_call_id for part in response.parts if isinstance(part, NativeToolCallPart)][-1] == 'srvtoolu_1'


async def test_anthropic_container_from_message_history_not_sent_for_function_tool(allow_model_requests: None):
    """A function tool named `code_execution` isn't code execution, so the container id from history isn't sent.

    Unit test: no recording sends a function tool named after a server tool.
    """
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock(c)
    agent = Agent(AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client)))

    @agent.tool_plain
    def code_execution(code: str) -> str:  # pragma: no cover
        return code

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[TextPart(content='world')],
            provider_name='anthropic',
            provider_details={'container_id': 'container_from_history'},
        ),
    ]
    await agent.run('follow up', message_history=history)

    assert get_mock_chat_completion_kwargs(mock_client)[0]['container'] is OMIT


@pytest.mark.vcr()
async def test_anthropic_container_from_history_not_sent_without_code_execution(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """A follow-up without code execution doesn't send the container id from history.

    Anthropic rejects it with `400 Container identifier can only be provided when using the code execution tool`.
    """
    model = anthropic_model('claude-sonnet-5-5', capture=True)
    first = await Agent(model, capabilities=[NativeTool(CodeExecutionTool())]).run(
        'Use the code execution tool to compute 3 * 12390.'
    )
    second = await Agent(model).run('Say the result in one sentence.', message_history=first.all_messages())

    first_response = message(first.all_messages(), ModelResponse, index=-1)
    assert first_response.provider_details is not None
    assert isinstance(first_response.provider_details.get('container_id'), str)
    assert [('container' in body, 'tools' in body) for body in request_capture.bodies('/v1/messages')] == snapshot(
        [(False, True), (False, False)]
    )
    assert second.output == snapshot('The result of 3 × 12390 is 37,170.')
