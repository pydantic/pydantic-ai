"""Tests for `GoogleModel`'s handling of agentic media processing responses.

With `media_processing='AGENTIC'`, Gemini navigates the video itself and reports each
internal segment/transcript fetch as two Parts carrying only a `thought_signature`: the
`processing_call` and its `processing_result`. Both the streaming and non-streaming paths
surface each pair as a `NativeToolCallPart`/`NativeToolReturnPart` named `media_processing`,
like other provider-executed tools. Their signatures are rejected with
`400 Invalid thought signature` if replayed in history, so the pairs are never echoed and the
replayed history carries only the final Part's own signature.

The request-side mapping of `vendor_metadata['media_processing']` is covered in
`test_media.py` alongside `media_resolution`; these tests pin the response side and,
per path, that the outgoing request actually carried the field.
"""

from __future__ import annotations as _annotations

import base64
from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest
from pytest_mock import MockerFixture

from pydantic_ai import Agent, BinaryContent, ModelResponse, TextPart
from pydantic_ai.messages import NativeToolCallPart, NativeToolReturnPart

from ..._inline_snapshot import snapshot
from ...conftest import try_import

with try_import() as imports_successful:
    from google.genai.types import GenerateContentResponse

    from pydantic_ai.models.google import (
        GoogleModel,
        _content_model_response,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
]

AGENTIC_VIDEO = BinaryContent(
    data=b'\x00\x00\x00\x00', media_type='video/mp4', vendor_metadata={'media_processing': 'AGENTIC'}
)
FINAL_SIGNATURE = b'sig-final'
FINAL_SIGNATURE_B64 = base64.b64encode(FINAL_SIGNATURE).decode()
STEP_1_B64 = base64.b64encode(b'sig-step-1').decode()
STEP_2_B64 = base64.b64encode(b'sig-step-2').decode()


def _response(parts: list[dict[str, Any]], *, finish_reason: str | None = 'STOP') -> GenerateContentResponse:
    candidate: dict[str, Any] = {'content': {'role': 'model', 'parts': parts}}
    if finish_reason:
        candidate['finish_reason'] = finish_reason
    return GenerateContentResponse.model_validate(
        {
            'response_id': 'resp-1',
            'model_version': 'gemini-test',
            'candidates': [candidate],
            'usage_metadata': {'prompt_token_count': 10, 'candidates_token_count': 5},
        }
    )


async def _aiter(chunks: list[GenerateContentResponse]) -> AsyncIterator[GenerateContentResponse]:
    for chunk in chunks:
        yield chunk


def _sent_media_processing(call_kwargs: Mapping[str, Any]) -> object:
    """The `media_processing` value on the video Part of the outgoing request."""
    [video_part] = [part for part in call_kwargs['contents'][0]['parts'] if 'inline_data' in part]
    return video_part['media_processing']


async def test_agentic_steps_surface_as_native_tool_pair(allow_model_requests: None, mocker: MockerFixture):
    """Non-streaming: the two bare-signature Parts become one `media_processing` call/return
    pair sharing a `tool_call_id`, the final Part keeps its own signature, and replaying the
    response sends back exactly that one signature — the pair is omitted."""
    model = GoogleModel('gemini-3.7-flash', provider=GoogleProvider(api_key='test-key'))
    generate_content = mocker.patch.object(
        model.client.aio.models,
        'generate_content',
        return_value=_response(
            [
                {'thought_signature': b'sig-step-1'},
                {'thought_signature': b'sig-step-2'},
                {'text': 'An otter appears last.', 'thought_signature': FINAL_SIGNATURE},
            ]
        ),
    )

    result = await Agent(model).run(['Which animal appears last?', AGENTIC_VIDEO])

    assert result.output == 'An otter appears last.'
    assert _sent_media_processing(generate_content.call_args.kwargs) == 'AGENTIC'

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    call, ret, text = response.parts
    assert isinstance(call, NativeToolCallPart) and isinstance(ret, NativeToolReturnPart)
    assert (call.tool_name, call.args, call.provider_name) == ('media_processing', None, 'google')
    assert (ret.tool_name, ret.content, ret.provider_name) == ('media_processing', None, 'google')
    assert ret.tool_call_id == call.tool_call_id
    assert call.provider_details == {'thought_signature': STEP_1_B64}
    assert ret.provider_details == {'thought_signature': STEP_2_B64}
    assert text == snapshot(
        TextPart(
            content='An otter appears last.',
            provider_name='google',
            provider_details={'thought_signature': FINAL_SIGNATURE_B64},
        )
    )
    assert _content_model_response(response, frozenset({'google'})) == snapshot(
        {'role': 'model', 'parts': [{'text': 'An otter appears last.', 'thought_signature': FINAL_SIGNATURE}]}
    )


async def test_agentic_steps_surface_as_native_tool_pair_streaming(allow_model_requests: None, mocker: MockerFixture):
    """Streaming: the same shape arrives as leading bare-signature chunks, then text deltas,
    then a final delta carrying the turn's signature. The result must match the
    non-streaming path exactly."""
    model = GoogleModel('gemini-3.7-flash', provider=GoogleProvider(api_key='test-key'))
    generate_content_stream = mocker.patch.object(
        model.client.aio.models,
        'generate_content_stream',
        return_value=_aiter(
            [
                _response([{'thought_signature': b'sig-step-1'}], finish_reason=None),
                _response([{'thought_signature': b'sig-step-2'}], finish_reason=None),
                _response([{'text': 'An otter '}], finish_reason=None),
                _response([{'text': 'appears last.', 'thought_signature': FINAL_SIGNATURE}]),
            ]
        ),
    )

    async with Agent(model).run_stream(['Which animal appears last?', AGENTIC_VIDEO]) as result:
        output = await result.get_output()

    assert output == 'An otter appears last.'
    assert _sent_media_processing(generate_content_stream.call_args.kwargs) == 'AGENTIC'

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    call, ret, text = response.parts
    assert isinstance(call, NativeToolCallPart) and isinstance(ret, NativeToolReturnPart)
    assert (call.tool_name, call.args, call.provider_name) == ('media_processing', None, 'google')
    assert (ret.tool_name, ret.content, ret.provider_name) == ('media_processing', None, 'google')
    assert ret.tool_call_id == call.tool_call_id
    assert call.provider_details == {'thought_signature': STEP_1_B64}
    assert ret.provider_details == {'thought_signature': STEP_2_B64}
    assert text == snapshot(
        TextPart(
            content='An otter appears last.',
            provider_name='google',
            provider_details={'thought_signature': FINAL_SIGNATURE_B64},
        )
    )
    assert _content_model_response(response, frozenset({'google'})) == snapshot(
        {'role': 'model', 'parts': [{'text': 'An otter appears last.', 'thought_signature': FINAL_SIGNATURE}]}
    )


async def test_agentic_odd_step_count_keeps_trailing_call(allow_model_requests: None, mocker: MockerFixture):
    """Steps arrive call-then-result, so an odd count leaves a trailing call with no return."""
    model = GoogleModel('gemini-3.7-flash', provider=GoogleProvider(api_key='test-key'))
    mocker.patch.object(
        model.client.aio.models,
        'generate_content',
        return_value=_response(
            [
                {'thought_signature': b'sig-step-1'},
                {'thought_signature': b'sig-step-2'},
                {'thought_signature': b'sig-step-3'},
                {'text': 'Done.', 'thought_signature': FINAL_SIGNATURE},
            ]
        ),
    )

    result = await Agent(model).run(['Describe it.', AGENTIC_VIDEO])

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    first_call, first_return, second_call, text = response.parts
    assert isinstance(first_call, NativeToolCallPart) and isinstance(first_return, NativeToolReturnPart)
    assert isinstance(second_call, NativeToolCallPart) and isinstance(text, TextPart)
    assert first_return.tool_call_id == first_call.tool_call_id
    assert second_call.tool_call_id != first_call.tool_call_id
    assert _content_model_response(response, frozenset({'google'})) == snapshot(
        {'role': 'model', 'parts': [{'text': 'Done.', 'thought_signature': FINAL_SIGNATURE}]}
    )
