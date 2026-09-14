"""Feature-central tests for `CodeExecutionTool.files` uploaded-file support.

The Anthropic and OpenAI baseline provider round trips each pass a foreign-provider
file. Additional Anthropic round-trip regressions cover multi-turn placement, cache
stability, mixed tools, and rejected-container recovery.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import pytest
from inline_snapshot import snapshot
from pydantic import JsonValue

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UploadedFile, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import CodeExecutionTool

from .conftest import RequestCapture, try_import
from .models.conftest import content_blocks, message_shape

with try_import() as anthropic_imports_successful:
    import anthropic

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as openai_imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
]

_CSV_BYTES = b'item,value\napple,30\nbanana,70\n'
_PROMPT = 'Use the code execution tool to read the uploaded CSV file and report the sum of the `value` column.'


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_code_execution_files(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
):
    """Upload a real file to the Anthropic Files API and have code execution read it."""
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key, http_client=request_capture.client)
    uploaded = await client.beta.files.upload(
        file=('data.csv', _CSV_BYTES, 'text/csv'),
        betas=['files-api-2025-04-14'],
    )

    try:
        model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
        agent = Agent(
            model,
            capabilities=[
                NativeTool(
                    CodeExecutionTool(
                        files=[
                            UploadedFile(file_id=uploaded.id, provider_name='anthropic'),
                            UploadedFile(file_id='file-other-provider', provider_name='openai'),
                        ]
                    )
                )
            ],
        )

        result = await agent.run(_PROMPT)
    finally:
        await client.beta.files.delete(uploaded.id, betas=['files-api-2025-04-14'])
        await client.close()

    assert '100' in result.output

    # The uploaded file goes up as a `container_upload` block (the API only accepts this under
    # the `files-api-2025-04-14` beta, which the model auto-enables); the foreign-provider file
    # is filtered out, so only the anthropic file id is sent.
    # Read off the wire rather than `vcr.requests`: the default matchers ignore the body, so a
    # recording cannot disagree with the filter it recorded — a regression that sent both file ids
    # would replay green against it.
    container_uploads = content_blocks(request_capture.body('/v1/messages'), 'container_upload')
    assert container_uploads == [{'type': 'container_upload', 'file_id': uploaded.id}]


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_code_execution_files_multi_turn(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
):
    """Across two turns, the container is reused by id *and* the `container_upload` block is re-sent.

    `pending_container_uploads` is recomputed from the static `CodeExecutionTool.files` config on
    every request and appended to every user message, so turn 2 re-sends the `container_upload` block
    — once per user message — for a file the reused container already holds. This pins, against the
    live API, that Anthropic tolerates that redundant re-send: turn 2 carries both
    `container=<id from turn 1>` and two copies of the same `container_upload`, and still succeeds.
    (The re-send is also what keeps each user message byte-identical across turns, so it must not be
    gated to a single turn — see `test_anthropic_code_execution_files_cache_prefix_stable`.)
    """
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key, http_client=request_capture.client)
    uploaded = await client.beta.files.upload(
        file=('data.csv', _CSV_BYTES, 'text/csv'),
        betas=['files-api-2025-04-14'],
    )

    try:
        model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
        agent = Agent(
            model,
            capabilities=[
                NativeTool(CodeExecutionTool(files=[UploadedFile(file_id=uploaded.id, provider_name='anthropic')]))
            ],
        )

        first = await agent.run(_PROMPT)
        second = await agent.run(
            'Use the code execution tool again to report the average of the `value` column.',
            message_history=first.all_messages(),
        )
    finally:
        await client.beta.files.delete(uploaded.id, betas=['files-api-2025-04-14'])
        await client.close()

    assert '100' in first.output
    assert '50' in second.output

    # The container id from turn 1's response is what turn 2 reuses.
    first_response = first.all_messages()[-1]
    assert isinstance(first_response, ModelResponse)
    container_id = (first_response.provider_details or {}).get('container_id')
    assert container_id

    first_body, second_body = request_capture.bodies('/v1/messages')

    upload_block = {'type': 'container_upload', 'file_id': uploaded.id}
    # Turn 1: fresh container (no `container` param), one user message, so one upload block.
    assert 'container' not in first_body
    assert content_blocks(first_body, 'container_upload') == [upload_block]
    # Turn 2: container reused by id, *and* the upload block re-sent on each of the two user
    # messages — the redundant re-send is accepted by the API.
    assert second_body['container'] == container_id
    assert content_blocks(second_body, 'container_upload') == [upload_block, upload_block]
    # One upload per user message, not two on one: the count alone would not tell them apart.
    assert message_shape(second_body) == snapshot(
        [
            ('user', ['text', 'container_upload']),
            (
                'assistant',
                [
                    'text',
                    'server_tool_use',
                    'bash_code_execution_tool_result',
                    'text',
                    'server_tool_use',
                    'bash_code_execution_tool_result',
                    'text',
                ],
            ),
            ('user', ['text', 'container_upload']),
        ]
    )


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_code_execution_files_fresh_container_multi_turn(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
):
    """A multi-turn history with no container to reuse still gets the file into the fresh container.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/7775. `test_anthropic_code_execution_files_multi_turn` can't catch it: its
    turn 2 reuses turn 1's container by id, so the file is already inside regardless of where the
    `container_upload` block lands. Here turn 1 never runs code, so no `container_id` reaches the
    history and turn 2 allocates a fresh container — and the server only materializes an upload that
    falls inside the turn it is generating, which is why appending to the first user message alone
    stranded the block in turn 1 and left the container empty.

    The wire assertions read `request_capture`, not the cassette: the default matchers ignore the
    body, so a first-message-only injection replays against this recording and the round-trip half
    of this test passes on the unfixed code. The capture hook sees what the code actually built.
    """
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key, http_client=request_capture.client)
    uploaded = await client.beta.files.upload(
        file=('data.csv', _CSV_BYTES, 'text/csv'),
        betas=['files-api-2025-04-14'],
    )

    try:
        model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
        agent = Agent(
            model,
            capabilities=[
                NativeTool(CodeExecutionTool(files=[UploadedFile(file_id=uploaded.id, provider_name='anthropic')]))
            ],
        )

        first = await agent.run('Reply with just the word `ready`. Do not use any tools.')
        second = await agent.run(_PROMPT, message_history=first.all_messages())
    finally:
        await client.beta.files.delete(uploaded.id, betas=['files-api-2025-04-14'])
        await client.close()

    # Turn 1 ran no code, so it left no container behind for turn 2 to reuse.
    first_response = first.all_messages()[-1]
    assert isinstance(first_response, ModelResponse)
    assert not (first_response.provider_details or {}).get('container_id')

    # Turn 2 read the uploaded file out of the container it allocated for itself.
    assert '100' in second.output

    second_body = request_capture.body('/v1/messages', index=1)
    assert 'container' not in second_body

    # The block is on every user message, so it is on the last one — the only position the server
    # acts on. Asserting the last one alone would pass for a tail-only injection, which busts the
    # cacheable prefix instead (`test_anthropic_code_execution_files_cache_prefix_stable`).
    assert message_shape(second_body) == snapshot(
        [('user', ['text', 'container_upload']), ('assistant', ['text']), ('user', ['text', 'container_upload'])]
    )
    upload_block = {'type': 'container_upload', 'file_id': uploaded.id}
    assert content_blocks(second_body, 'container_upload') == [upload_block, upload_block]


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_code_execution_files_with_function_tool(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
):
    """A tool-result-only user message must *not* get an upload block, and the file still arrives.

    Attaching one there makes the model open its very next response with a code-execution call
    alongside the function-tool call, which the API then leaves unexecuted and rejects on the
    following request with `bash_code_execution` tool use ... was found without ... (reproduced 6/6
    live). Skipping those messages costs nothing: a tool result never opens a turn, so the prompt
    that did still carries the block and is still inside the turn being generated.
    """
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key, http_client=request_capture.client)
    uploaded = await client.beta.files.upload(
        file=('data.csv', _CSV_BYTES, 'text/csv'),
        betas=['files-api-2025-04-14'],
    )

    try:
        model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
        agent = Agent(
            model,
            capabilities=[
                NativeTool(CodeExecutionTool(files=[UploadedFile(file_id=uploaded.id, provider_name='anthropic')]))
            ],
        )

        @agent.tool_plain
        def get_units() -> str:
            """Return the unit the `value` column is measured in."""
            return 'kg'

        result = await agent.run(
            'First call `get_units`. Then use the code execution tool to read the uploaded CSV and '
            'report the sum of the `value` column with that unit.'
        )
    finally:
        await client.beta.files.delete(uploaded.id, betas=['files-api-2025-04-14'])
        await client.close()

    assert '100' in result.output
    assert 'kg' in result.output

    # The request that follows the tool return: its last user message is the tool result and carries
    # no upload, while the prompt that opened the turn still does.
    bodies = request_capture.bodies('/v1/messages')
    assert message_shape(bodies[-1]) == snapshot(
        [
            ('user', ['text', 'container_upload']),
            ('assistant', ['text', 'tool_use', 'server_tool_use']),
            ('user', ['tool_result']),
        ]
    )


# A real container from an earlier recording, created on 2026-06-30 and retried on 2026-08-28,
# about 59 days later. It had expired past Anthropic's documented 30-day lifetime. Anthropic answers
# a `container_upload` aimed at this history-resolved id with a generic `api_error` 500 rather than
# the 404 it gives for an id that never existed. The missing typed error or discriminator is tracked
# upstream in https://github.com/pydantic/pydantic-ai/issues/7833.
_DEAD_CONTAINER_ID = 'container_01EG1LKXFPoQJ9tpbsZ1dh74'


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_code_execution_files_rejected_container_is_dropped_and_retried(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture, vcr: Any
):
    """A rejected container id resolved from history is dropped and the request resent once.

    The history-resolved id was created on 2026-06-30 and retried on 2026-08-28, about 59 days later,
    so it had expired past Anthropic's documented 30-day lifetime. Pairing that expired id with
    `container_upload` answers 500 — the generic `api_error` body any internal failure produces —
    rather than the 404 it gives for an id that never existed. The cause is not readable off the
    response, so this test pins the shape that reproduces. The *remedy* is documented — "Send the
    request again without the `container` parameter to get a new container" — and that is exactly
    what the two requests here show: the first carries the expired id, the second carries no
    container at all, and the fresh container gets the file. The missing typed error or discriminator
    is tracked upstream in https://github.com/pydantic/pydantic-ai/issues/7833.

    `max_retries=0` keeps the SDK's own retry out of the way, so the two captured requests are ours.
    It also makes playback match live: the real 500 carries `x-should-retry: false` and the SDK stops
    at one attempt, but the cassette serializer drops that header, so a replayed 500 would otherwise
    be retried by the SDK and silently consume the second recorded interaction.
    """
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key, http_client=request_capture.client, max_retries=0)
    uploaded = await client.beta.files.upload(
        file=('data.csv', _CSV_BYTES, 'text/csv'),
        betas=['files-api-2025-04-14'],
    )

    try:
        model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
        agent = Agent(
            model,
            capabilities=[
                NativeTool(CodeExecutionTool(files=[UploadedFile(file_id=uploaded.id, provider_name='anthropic')]))
            ],
        )

        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='Earlier turn, long enough ago that the container is gone.')]),
            ModelResponse(
                parts=[TextPart(content='Earlier answer.')],
                provider_name='anthropic',
                provider_details={'container_id': _DEAD_CONTAINER_ID},
            ),
        ]
        result = await agent.run(_PROMPT, message_history=history)
    finally:
        await client.beta.files.delete(uploaded.id, betas=['files-api-2025-04-14'])
        await client.close()

    assert '100' in result.output

    recorded_error = vcr.responses[1]
    recorded_error_body = json.loads(recorded_error['body']['string'])
    assert (recorded_error['status']['code'], recorded_error_body['error']['type']) == snapshot((500, 'api_error'))

    # Both attempts are on the wire, and only the first carries the dead id.
    bodies = request_capture.bodies('/v1/messages')
    assert [body.get('container') for body in bodies] == [_DEAD_CONTAINER_ID, None]


# Large instructions so the cacheable prefix (system + tools + user text) clears Anthropic's
# ~1024-token minimum for claude-sonnet — otherwise a cache miss would be a too-small-to-cache
# artifact rather than a real signal. The text is request-stable across turns.
_CACHE_INSTRUCTIONS = (
    'You are a meticulous data analyst. Always use the code execution tool to read and compute '
    'over any attached CSV file before answering. '
) + ' '.join(f'Guideline {i}: prefer exact arithmetic over estimation when analyzing tabular data.' for i in range(120))


# `mode` (not a built `AnthropicModelSettings`) so this list stays import-safe when anthropic isn't
# installed — the slim test job collects this module with the `try_import` symbols undefined.
@dataclass(frozen=True)
class _CacheCase:
    id: str
    model: str
    mode: Literal['messages', 'automatic']


_CACHE_CASES = [
    _CacheCase('messages-sonnet-4-6', 'claude-sonnet-4-6', 'messages'),
    _CacheCase('messages-sonnet-5', 'claude-sonnet-5', 'messages'),
    _CacheCase('automatic-sonnet-4-6', 'claude-sonnet-4-6', 'automatic'),
    _CacheCase('automatic-sonnet-5', 'claude-sonnet-5', 'automatic'),
]


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in _CACHE_CASES])
async def test_anthropic_code_execution_files_cache_prefix_stable(
    case: _CacheCase, allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
):
    """The `container_upload` injection keeps the cacheable prefix stable across steps.

    The upload blocks are recomputed from the static `CodeExecutionTool.files` config every request
    and appended to *every* user message, so each of them stays byte-identical as history grows.
    This pins both halves of the guarantee against the live API:

    (a) Structural — every user message carries the `container_upload`, and the first one is
        identical across every request, so the upload never moves and never perturbs the prefix.
    (b) Real reuse — turn 2 reads back at least everything turn 1 wrote
        (`cache_read >= turn-1 cache_write`). This is the property `cache_read > 0` could not prove:
        a moving injection point would shrink the reused prefix below what turn 1 cached.

    Parametrized over the per-block (`anthropic_cache_messages`) and automatic (`anthropic_cache`)
    cache-control paths and across models, whose breakpoint placement differs.
    """
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key, http_client=request_capture.client)
    uploaded = await client.beta.files.upload(
        file=('data.csv', _CSV_BYTES, 'text/csv'),
        betas=['files-api-2025-04-14'],
    )

    settings = (
        AnthropicModelSettings(anthropic_cache_messages=True)
        if case.mode == 'messages'
        else AnthropicModelSettings(anthropic_cache=True)
    )
    try:
        model = AnthropicModel(case.model, provider=AnthropicProvider(anthropic_client=client))
        agent = Agent(
            model,
            capabilities=[
                NativeTool(CodeExecutionTool(files=[UploadedFile(file_id=uploaded.id, provider_name='anthropic')]))
            ],
            instructions=_CACHE_INSTRUCTIONS,
            model_settings=settings,
        )

        first = await agent.run(_PROMPT)
        second = await agent.run(
            'Use the code execution tool again to report the average of the `value` column.',
            message_history=first.all_messages(),
        )
    finally:
        await client.beta.files.delete(uploaded.id, betas=['files-api-2025-04-14'])
        await client.close()

    bodies = request_capture.bodies('/v1/messages')
    assert len(bodies) == 2

    # (a) Prefix stability: every user message carries the upload, so no message gains or loses one
    #     as history grows, and the first user message's content is identical across both requests.
    #     The `cache_control` breakpoint marker is stripped before comparing — in `messages` mode it
    #     lands on the first user message in turn 1 (when it is also the last) but not in turn 2,
    #     which is orthogonal to stability: the cached *content* (text + upload) is what must not move.
    first_user_contents: list[list[dict[str, JsonValue]]] = []
    messages_by_body: list[list[tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]]] = []
    for body in bodies:
        messages = body.get('messages')
        assert isinstance(messages, list)
        typed_messages: list[tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]] = []
        for message in messages:
            assert isinstance(message, dict)
            content = message.get('content')
            assert isinstance(content, list)
            typed_content: list[dict[str, JsonValue]] = []
            for block in content:
                assert isinstance(block, dict)
                typed_content.append(block)
            typed_messages.append((message, typed_content))
        messages_by_body.append(typed_messages)

        user_contents = [content for message, content in typed_messages if message.get('role') == 'user']
        upload_contents = [
            content for content in user_contents if any(block.get('type') == 'container_upload' for block in content)
        ]
        assert len(upload_contents) == len(user_contents)
        first_user_contents.append(
            [{key: value for key, value in block.items() if key != 'cache_control'} for block in user_contents[0]]
        )
    assert first_user_contents[0] == first_user_contents[1]

    # (b) Cache-control placement differs by mode, but `container_upload` is never the breakpoint.
    if case.mode == 'messages':
        for body, typed_messages in zip(bodies, messages_by_body):
            cached_blocks = [block for _, content in typed_messages for block in content if 'cache_control' in block]
            assert cached_blocks and all(block.get('type') != 'container_upload' for block in cached_blocks)
            assert 'cache_control' not in body
    else:
        for body, typed_messages in zip(bodies, messages_by_body):
            assert body['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}
            assert all('cache_control' not in block for _, content in typed_messages for block in content)

    # (c) The prefix is genuinely reused: turn 2 reads back at least everything turn 1 wrote.
    assert first.usage.cache_write_tokens > 0
    assert second.usage.cache_read_tokens >= first.usage.cache_write_tokens


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_code_execution_files(allow_model_requests: None, openai_api_key: str, vcr: Any):
    """Upload a real file to the OpenAI Files API and have code execution read it."""
    client = openai.AsyncOpenAI(api_key=openai_api_key)
    uploaded = await client.files.create(file=('data.csv', _CSV_BYTES), purpose='assistants')

    try:
        model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=client))
        agent = Agent(
            model,
            capabilities=[
                NativeTool(
                    CodeExecutionTool(
                        files=[
                            UploadedFile(file_id=uploaded.id, provider_name='openai'),
                            UploadedFile(file_id='file-other-provider', provider_name='anthropic'),
                        ]
                    )
                )
            ],
        )

        result = await agent.run(_PROMPT)
    finally:
        await client.files.delete(uploaded.id)
        await client.close()

    assert '100' in result.output

    # The uploaded file id goes into the `code_interpreter` container `file_ids`; the
    # foreign-provider file is filtered out.
    responses_request = [r for r in vcr.requests if '/v1/responses' in r.uri][0]
    tools = json.loads(responses_request.body)['tools']
    code_interpreter = [t for t in tools if t['type'] == 'code_interpreter'][0]
    assert code_interpreter['container'] == {'type': 'auto', 'file_ids': [uploaded.id]}


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
def test_openai_code_execution_files_all_filtered():
    """Files set but none match the provider: no `file_ids` is sent (unit-tested branch).

    This is not a VCR test because there is no observable round-trip — the whole point
    is that nothing file-related reaches the provider, so we assert the built request.
    """
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key='mock-api-key'))
    parameters = ModelRequestParameters(
        native_tools=[
            CodeExecutionTool(
                files=[
                    UploadedFile(file_id='file-anthropic', provider_name='anthropic'),
                    UploadedFile(file_id='file-google', provider_name='google-gla'),
                ]
            )
        ],
    )

    tools = model._get_native_tools(parameters)  # pyright: ignore[reportPrivateUsage]

    assert tools == snapshot([{'type': 'code_interpreter', 'container': {'type': 'auto'}}])
