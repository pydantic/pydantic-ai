import asyncio

import pytest

from pydantic_ai import ToolReturnPart, UserPromptPart
from pydantic_ai.messages import UploadedFile

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

from .mock_openai import MockOpenAIResponses, response_message

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


# Direct mapping tests catch request payload regressions that VCR cassette matching may not.
def test_openai_responses_uploaded_image_file_input():
    mock_client = MockOpenAIResponses.create_mock(response_message([]))
    model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(openai_client=mock_client))

    message = asyncio.run(
        model._map_user_prompt(  # pyright: ignore[reportPrivateUsage]
            part=UserPromptPart(
                content=[
                    'Describe this image',
                    UploadedFile(
                        file_id='file-image-123',
                        provider_name='openai',
                        media_type='image/png',
                        vendor_metadata={'detail': 'high'},
                    ),
                ]
            )
        )
    )

    assert message == {
        'role': 'user',
        'content': [
            {'text': 'Describe this image', 'type': 'input_text'},
            {'type': 'input_image', 'file_id': 'file-image-123', 'detail': 'high'},
        ],
    }


def test_openai_responses_uploaded_image_file_tool_return():
    output = asyncio.run(
        OpenAIResponsesModel._map_tool_return_output(  # pyright: ignore[reportPrivateUsage]
            ToolReturnPart(
                tool_name='get_image',
                content=[
                    'Tool returned this image',
                    UploadedFile(file_id='file-image-456', provider_name='openai', media_type='image/webp'),
                ],
                tool_call_id='call_123',
            )
        )
    )

    assert output == [
        {'type': 'input_text', 'text': 'Tool returned this image'},
        {'type': 'input_image', 'file_id': 'file-image-456', 'detail': 'auto'},
    ]
