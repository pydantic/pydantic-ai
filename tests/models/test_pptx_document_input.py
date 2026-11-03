"""A `.pptx` sent as `BinaryContent` through OpenAI.

A `.pptx` has no entry in Pydantic AI's document format table: `BinaryContent.is_document` holds because the
media type is neither audio, image nor video, and `.format` is the extension `mimetypes` knows the type by.
Cassette tests, because what each API does with the file is the point: the Responses API reads it, and Chat
Completions, whose `file` part takes PDFs only, rejects it with its own error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cassetter import Cassette
from inline_snapshot import snapshot

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import ModelHTTPError

from ..cassette_utils import single_request_body
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]

PPTX_MEDIA_TYPE = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'


@pytest.fixture(scope='module')
def pptx_content(assets_path: Path) -> BinaryContent:
    """`tests/assets/dummy.pptx`, a one-slide deck titled "Dummy PPTX file"; `from_path` infers the media type."""
    content = BinaryContent.from_path(assets_path / 'dummy.pptx')
    assert content.media_type == PPTX_MEDIA_TYPE
    return content


async def test_pptx_binary_content_responses(
    allow_model_requests: None, openai_api_key: str, pptx_content: BinaryContent, vcr: Cassette
):
    """The Responses API reads a `.pptx` sent as `input_file`, with the filename OpenAI picks the parser from."""
    model = OpenAIResponsesModel('gpt-4.1-mini', provider=OpenAIProvider(api_key=openai_api_key))

    result = await Agent(model).run(['Quote every piece of text in this file verbatim, nothing else.', pptx_content])

    assert result.output == snapshot("""\
Dummy PPTX file
A single slide used to test document input.\
""")
    [message] = single_request_body(vcr)['input']
    [_, file] = message['content']
    assert file['type'] == 'input_file'
    assert file['filename'] == 'filename.pptx'
    assert file['file_data'].startswith(f'data:{PPTX_MEDIA_TYPE};base64,')


async def test_pptx_binary_content_chat(
    allow_model_requests: None, openai_api_key: str, pptx_content: BinaryContent, vcr: Cassette
):
    """Chat Completions is sent the same `.pptx` as a `file` part and rejects it: its `file_data` takes PDFs only.

    That refusal is the API's to make, so it reaches the user as the API's own error rather than being raised
    before the request.
    """
    model = OpenAIChatModel('gpt-4.1-mini', provider=OpenAIProvider(api_key=openai_api_key))

    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model).run(['Quote every piece of text in this file verbatim, nothing else.', pptx_content])

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == snapshot(
        {
            'message': "Invalid file data: 'messages[0].content[1].file.file_data'. Expected a base64-encoded data URL with an application/pdf MIME type (e.g. 'data:application/pdf;base64,SGVsbG8sIFdvcmxkIQ=='), but got unsupported MIME type 'application/vnd.openxmlformats-officedocument.presentationml.presentation'.",
            'type': 'invalid_request_error',
            'param': 'messages[0].content[1].file.file_data',
            'code': 'invalid_value',
        }
    )
    [message] = single_request_body(vcr)['messages']
    [_, file] = message['content']
    assert file['type'] == 'file'
    assert file['file']['filename'] == 'filename.pptx'
