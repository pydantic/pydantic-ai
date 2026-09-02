"""Unit tests for Anthropic `CodeExecutionTool.files` uploads and container recovery."""

from __future__ import annotations

import httpx2
import pytest
from inline_snapshot import snapshot
from pydantic import JsonValue

from pydantic_ai import Agent, ModelHTTPError
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UploadedFile,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import CodeExecutionTool

from ...conftest import try_import
from ..conftest import cache_breakpoints, content_blocks, message_shape

with try_import() as anthropic_imports_successful:
    from anthropic import APIStatusError, omit as OMIT
    from anthropic.types.beta import BetaTextBlock, BetaUsage
    from anthropic.types.beta.beta_container_params import BetaContainerParams

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from ..test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

pytestmark = [
    pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
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
