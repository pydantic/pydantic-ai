"""`count_tokens` on a request whose prompt carries an Anthropic-hosted `UploadedFile`."""

from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, UploadedFile, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from ...conftest import try_import

with try_import() as imports_successful:
    import anthropic

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.vcr,
]


async def test_count_tokens_rejects_uploaded_file_in_prompt(allow_model_requests: None, anthropic_api_key: str):
    """The count endpoint rejects the file source an `UploadedFile` renders, so such a request can't be counted.

    This is left unfixed on purpose: leaving the document out of the count would undercount the request by the
    whole document, and a `count_tokens_before_request` usage limit relies on the count to stop an oversized
    request before it's sent.

    If a re-recording returns a count instead, the endpoint accepts file sources now: remove this test along with the
    uploaded-file caveat under Token counting in `docs/models/anthropic.md`.
    """
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
    uploaded = await client.beta.files.upload(
        file=('notes.txt', b'The capital of France is Paris.\n', 'text/plain'),
        betas=['files-api-2025-04-14'],
    )

    try:
        model = AnthropicModel('claude-fable-5-2', provider=AnthropicProvider(anthropic_client=client))
        prompt = UserPromptPart(
            content=[
                'What does the file say?',
                UploadedFile(file_id=uploaded.id, provider_name='anthropic', media_type='text/plain'),
            ]
        )
        with pytest.raises(ModelHTTPError) as exc_info:
            await model.count_tokens([ModelRequest(parts=[prompt])], None, ModelRequestParameters())
    finally:
        await client.beta.files.delete(uploaded.id, betas=['files-api-2025-04-14'])
        await client.close()

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == snapshot(
        {
            'type': 'error',
            'error': {
                'type': 'invalid_request_error',
                'message': 'File sources are not supported in the token counting endpoint.',
            },
            'request_id': 'req_011ChBGxpX83mqirifRsnDVa',
        }
    )
