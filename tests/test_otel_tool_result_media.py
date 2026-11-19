"""Media in a tool's result is recorded in spans as the GenAI message part a user prompt records it as.

These run against `FunctionModel` rather than a cassette: what's under test is our own OTel serialization
of the tool's return value, which never reaches a provider.
"""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai import Agent, BinaryImage, ImageUrl, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings

from ._inline_snapshot import snapshot
from .conftest import try_import

with try_import() as imports_successful:
    from logfire.testing import CaptureLogfire

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='logfire not installed'),
]

IMAGE = BinaryImage(data=b'\x89PNGkiwi', media_type='image/png')
URL = ImageUrl('https://example.com/kiwi.png')


async def test_tool_result_media_recorded_as_message_parts(capfire: CaptureLogfire) -> None:
    """The tool span's result, and the tool response in the chat and agent run spans, carry the same parts.

    Media is found wherever the tool put it in the containers it returned; everything else is recorded as before.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('gen_images', {})])
        return ModelResponse(parts=[TextPart('a kiwi')])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[Instrumentation(settings=InstrumentationSettings())],
        name='agent',
    )

    @agent.tool_plain
    def gen_images() -> dict[str, Any]:
        return {'image': IMAGE, 'images': [URL, IMAGE], 'caption': 'a kiwi'}

    await agent.run('make images')

    spans = {
        span['name']: span['attributes'] for span in capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    }
    tool_result = spans['execute_tool gen_images']['gen_ai.tool.call.result']
    [chat_response] = spans['chat function:respond:']['gen_ai.input.messages'][-1]['parts']
    [run_response] = spans['invoke_agent agent']['pydantic_ai.all_messages'][2]['parts']

    assert tool_result == snapshot(
        {
            'image': {'type': 'blob', 'mime_type': 'image/png', 'modality': 'image', 'content': 'iVBOR2tpd2k='},
            'images': [
                {'type': 'uri', 'modality': 'image', 'mime_type': 'image/png', 'uri': 'https://example.com/kiwi.png'},
                {'type': 'blob', 'mime_type': 'image/png', 'modality': 'image', 'content': 'iVBOR2tpd2k='},
            ],
            'caption': 'a kiwi',
        }
    )
    assert chat_response['result'] == tool_result
    assert run_response['result'] == tool_result
