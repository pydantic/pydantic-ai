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

    from pydantic_ai.models.anthropic import AnthropicModel
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
