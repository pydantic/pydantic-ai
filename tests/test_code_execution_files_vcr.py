"""Feature-central tests for `CodeExecutionTool.files` (uploaded-file support).

The two VCR tests upload a real file via the provider's Files API and then run an
agent with `CodeExecutionTool(files=[...])`, proving the upload round-trip end to
end: the file reference reaches the provider (Anthropic `container_upload` block +
`files-api-2025-04-14` beta; OpenAI Responses `code_interpreter` container
`file_ids`) and the model reads the uploaded file. Each test also passes a
foreign-provider `UploadedFile` to show it is filtered out on the wire.

`test_openai_code_execution_files_all_filtered` is the one branch the round-trip
can't exercise (files set, but none match the provider, so no `file_ids` is sent):
it stays a unit test on the request-building path, kept here so the whole feature
lives in one file.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.messages import ModelResponse, UploadedFile
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import CodeExecutionTool

from .conftest import try_import

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
async def test_anthropic_code_execution_files(allow_model_requests: None, anthropic_api_key: str, vcr: Any):
    """Upload a real file to the Anthropic Files API and have code execution read it."""
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
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
    messages_request = [r for r in vcr.requests if '/v1/messages' in r.uri][0]
    assert 'beta=true' in messages_request.uri
    user_content = json.loads(messages_request.body)['messages'][0]['content']
    container_uploads = [block for block in user_content if block['type'] == 'container_upload']
    assert container_uploads == [{'type': 'container_upload', 'file_id': uploaded.id}]


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_code_execution_files_multi_turn(allow_model_requests: None, anthropic_api_key: str, vcr: Any):
    """Across two turns, the container is reused by id *and* the `container_upload` block is re-sent.

    `pending_container_uploads` is recomputed from the static `CodeExecutionTool.files` config on
    every request, so turn 2 re-appends the `container_upload` block for a file the reused container
    already holds. This pins, against the live API, that Anthropic tolerates that redundant re-send:
    turn 2 carries both `container=<id from turn 1>` and the same `container_upload`, and still
    succeeds. If the API ever started rejecting it, the append would need gating to the first turn.
    """
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
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

    messages_requests = [r for r in vcr.requests if '/v1/messages' in r.uri]
    assert len(messages_requests) == 2
    first_body, second_body = (json.loads(r.body) for r in messages_requests)

    first_uploads = [
        block
        for message in first_body['messages']
        for block in message['content']
        if block['type'] == 'container_upload'
    ]
    second_uploads = [
        block
        for message in second_body['messages']
        for block in message['content']
        if block['type'] == 'container_upload'
    ]

    upload_block = {'type': 'container_upload', 'file_id': uploaded.id}
    # Turn 1: fresh container (no `container` param), file uploaded.
    assert 'container' not in first_body
    assert first_uploads == [upload_block]
    # Turn 2: container reused by id, *and* the same upload block re-sent — accepted by the API.
    assert second_body['container'] == container_id
    assert second_uploads == [upload_block]


# Large instructions so the cacheable prefix (system + tools + user text) clears Anthropic's
# ~1024-token minimum for claude-sonnet — otherwise a cache miss would be a too-small-to-cache
# artifact rather than a real signal. The text is request-stable across turns.
_CACHE_INSTRUCTIONS = (
    'You are a meticulous data analyst. Always use the code execution tool to read and compute '
    'over any attached CSV file before answering. '
) + ' '.join(f'Guideline {i}: prefer exact arithmetic over estimation when analyzing tabular data.' for i in range(120))


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_code_execution_files_caching_multi_turn(
    allow_model_requests: None, anthropic_api_key: str, vcr: Any
):
    """Prompt caching survives the `container_upload` tail-injection across multiple steps.

    With `anthropic_cache_messages=True`, `cache_control` is placed by reverse-scanning the last
    message for the last *cacheable* block. Because `container_upload` is excluded from
    `_ANTHROPIC_CACHEABLE_PARAM_TYPES` and is appended after that scan runs, the marker lands on the
    text block and never on the `container_upload`. This pins both halves of the guarantee against
    the live API: (a) each turn `cache_control` sits on the pre-upload text, not on the upload, and
    (b) turn 2 gets a real cache *read* — the cached prefix is reused despite the upload sitting
    right after the cached block, proving the injection doesn't poison the prefix.
    """
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
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
            instructions=_CACHE_INSTRUCTIONS,
            model_settings=AnthropicModelSettings(anthropic_cache_messages=True),
        )

        first = await agent.run(_PROMPT)
        second = await agent.run(
            'Use the code execution tool again to report the average of the `value` column.',
            message_history=first.all_messages(),
        )
    finally:
        await client.beta.files.delete(uploaded.id, betas=['files-api-2025-04-14'])
        await client.close()

    # (a) Placement: every request's last user message ends with `text`(cache_control), then an
    #     uncached `container_upload`. The upload block never receives `cache_control`.
    messages_requests = [r for r in vcr.requests if '/v1/messages' in r.uri]
    assert len(messages_requests) == 2
    for request in messages_requests:
        last_user_content = [m for m in json.loads(request.body)['messages'] if m['role'] == 'user'][-1]['content']
        cached_blocks = [b for b in last_user_content if 'cache_control' in b]
        upload_blocks = [b for b in last_user_content if b['type'] == 'container_upload']
        # exactly one cache breakpoint, and it is the text block (not the upload)
        assert [b['type'] for b in cached_blocks] == ['text']
        assert upload_blocks and all('cache_control' not in b for b in upload_blocks)

    # (b) Cross-step cache hit: turn 1 writes the prefix, turn 2 reads it back.
    assert first.usage.cache_write_tokens > 0
    assert second.usage.cache_read_tokens > 0


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_code_execution_files_automatic_caching_multi_turn(
    allow_model_requests: None, anthropic_api_key: str, vcr: Any
):
    """The automatic (top-level `cache_control`) caching path also survives the upload injection.

    `anthropic_cache=True` sends a request-level `cache_control` param and lets the server place the
    breakpoint; unlike `anthropic_cache_messages`, it never mutates message blocks, so there is no
    placement to get wrong. This pins that across multiple steps: every request carries the top-level
    param and leaves all message blocks (including `container_upload`) untouched, and turn 2 still
    gets a real cache read — the upload sitting in the message doesn't poison the server-side prefix.
    """
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
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
            instructions=_CACHE_INSTRUCTIONS,
            model_settings=AnthropicModelSettings(anthropic_cache=True),
        )

        first = await agent.run(_PROMPT)
        second = await agent.run(
            'Use the code execution tool again to report the average of the `value` column.',
            message_history=first.all_messages(),
        )
    finally:
        await client.beta.files.delete(uploaded.id, betas=['files-api-2025-04-14'])
        await client.close()

    # Every request carries the top-level `cache_control` param and leaves message blocks unmutated:
    # no content block (least of all the `container_upload`) receives a per-block `cache_control`.
    messages_requests = [r for r in vcr.requests if '/v1/messages' in r.uri]
    assert len(messages_requests) == 2
    for request in messages_requests:
        body = json.loads(request.body)
        assert body['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}
        all_blocks = [b for m in body['messages'] for b in m['content']]
        assert any(b['type'] == 'container_upload' for b in all_blocks)
        assert all('cache_control' not in b for b in all_blocks)

    # Cross-step cache hit: turn 1 writes the prefix, turn 2 reads it back.
    assert first.usage.cache_write_tokens > 0
    assert second.usage.cache_read_tokens > 0


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
