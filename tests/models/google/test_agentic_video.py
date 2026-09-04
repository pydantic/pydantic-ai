from __future__ import annotations

import pytest
from httpx2 import Timeout

from pydantic_ai import NativeToolCallPart, NativeToolReturnPart, TextPart, UserPromptPart, VideoUrl
from pydantic_ai.agent import Agent

from ..._inline_snapshot import snapshot
from ...conftest import IsStr, RequestCapture, try_import
from ...parts_from_messages import part_types_from_messages

supports_media_processing = False
with try_import() as imports_successful:
    from google.genai.types import PartDict

    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    supports_media_processing = 'media_processing' in PartDict.__annotations__

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.skipif(not supports_media_processing, reason='requires google-genai>=2.20.0'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.mark.parametrize(
    ('stream', 'expected_part_types'),
    [
        pytest.param(
            False,
            snapshot([[UserPromptPart], [NativeToolCallPart, NativeToolReturnPart, TextPart]]),
            id='non-streaming',
        ),
        pytest.param(
            True,
            snapshot(
                [
                    [UserPromptPart],
                    [NativeToolCallPart, NativeToolReturnPart, NativeToolCallPart, NativeToolReturnPart, TextPart],
                ]
            ),
            id='streaming',
        ),
    ],
)
async def test_agentic_video_processing(
    stream: bool,
    expected_part_types: list[object],
    allow_model_requests: None,
    gemini_api_key: str,
    request_capture: RequestCapture,
) -> None:
    """Agentic video requests work, expose Google's processing steps, and remain replayable."""
    request_capture.client.timeout = Timeout(60)
    provider = GoogleProvider(api_key=gemini_api_key, http_client=request_capture.client)
    agent = Agent(GoogleModel('gemini-3.7-flash', provider=provider))
    prompt = [
        'In one sentence, which animals appear and roughly when?',
        VideoUrl(
            url='https://www.youtube.com/watch?v=lCdaVNyHtjU',
            vendor_metadata={'media_processing': 'AGENTIC'},
        ),
    ]

    if stream:
        async with agent.run_stream(prompt) as result:
            await result.get_output()
            messages = result.all_messages()
    else:
        result = await agent.run(prompt)
        messages = result.all_messages()

    assert request_capture.body()['contents'] == snapshot(
        [
            {
                'parts': [
                    {'text': 'In one sentence, which animals appear and roughly when?'},
                    {
                        'fileData': {
                            'fileUri': 'https://www.youtube.com/watch?v=lCdaVNyHtjU',
                            'mimeType': 'video/mp4',
                        },
                        'mediaProcessing': 'AGENTIC',
                    },
                ],
                'role': 'user',
            }
        ]
    )
    assert part_types_from_messages(messages) == expected_part_types

    follow_up = await agent.run('Which animal appears first?', message_history=messages)
    assert isinstance(follow_up.output, str)
    assert request_capture.body(index=1)['contents'] == snapshot(
        [
            {
                'parts': [
                    {'text': 'In one sentence, which animals appear and roughly when?'},
                    {
                        'fileData': {
                            'fileUri': 'https://www.youtube.com/watch?v=lCdaVNyHtjU',
                            'mimeType': 'video/mp4',
                        },
                        'mediaProcessing': 'AGENTIC',
                    },
                ],
                'role': 'user',
            },
            {'parts': [{'text': IsStr(), 'thoughtSignature': IsStr()}], 'role': 'model'},
            {'parts': [{'text': 'Which animal appears first?'}], 'role': 'user'},
        ]
    )
