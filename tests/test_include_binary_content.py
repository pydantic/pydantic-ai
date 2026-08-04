"""`InstrumentationSettings(include_binary_content=False)` must not leak base64 into any span.

These run against `FunctionModel` rather than a cassette: the leak is in our own OTel serialization
of `BinaryContent`, which is provider-independent, so a recording would only add a base64 image
payload to the repo without exercising anything the fake model doesn't.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    BinaryImage,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolReturnPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.profiles import ModelProfile

from ._inline_snapshot import snapshot
from .conftest import IsStr, try_import

with try_import() as imports_successful:
    from logfire.testing import CaptureLogfire

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='logfire not installed'),
    pytest.mark.anyio,
]

IMAGE = BinaryImage(data=b'\x89PNG' + b'kiwi' * 32, media_type='image/png')

REDACTED_IMAGE = snapshot({'media_type': 'image/png', 'vendor_metadata': None, 'kind': 'binary', 'identifier': IsStr()})
"""The shape a `BinaryImage` serializes to once its data is excluded: everything but `data`."""


def image_returning_tool_agent(settings: InstrumentationSettings) -> Agent[None, str]:
    """An agent whose tool returns an image, and which then answers with text."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('gen_image', {})])
        return ModelResponse(parts=[TextPart('a kiwi')])

    agent = Agent(FunctionModel(respond), capabilities=[Instrumentation(settings=settings)], name='agent')

    @agent.tool_plain
    def gen_image() -> BinaryImage:
        return IMAGE

    return agent


def image_output_agent(settings: InstrumentationSettings) -> Agent[None, BinaryImage]:
    """An agent whose own output is an image."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[FilePart(content=IMAGE)])

    return Agent(
        FunctionModel(respond, profile=ModelProfile(supports_image_output=True)),
        capabilities=[Instrumentation(settings=settings)],
        output_type=BinaryImage,
        name='agent',
    )


def image_argument_output_function_agent(settings: InstrumentationSettings) -> Agent[None, str]:
    """An agent whose output function receives an image as its argument."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart('final_result', {'image': {'data': IMAGE.base64, 'media_type': IMAGE.media_type}})]
        )

    def describe(image: BinaryImage) -> str:
        return image.media_type

    return Agent(
        FunctionModel(respond), capabilities=[Instrumentation(settings=settings)], output_type=describe, name='agent'
    )


def text_agent(settings: InstrumentationSettings) -> Agent[None, str]:
    """An agent that just answers, for cases where the image enters through message history."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('a kiwi')])

    return Agent(FunctionModel(respond), capabilities=[Instrumentation(settings=settings)], name='agent')


@dataclass(frozen=True)
class Case:
    """One span attribute that serializes arbitrary values and so could carry binary content."""

    id: str
    build: Callable[[InstrumentationSettings], Agent[None, Any]]
    span_name: str
    attribute: str
    redacted: Any
    """The attribute's value once `include_binary_content=False` excludes the image data."""
    history: list[ModelMessage] = field(default_factory=list[ModelMessage])


CASES = [
    Case(
        id='tool_result',
        build=image_returning_tool_agent,
        span_name='execute_tool gen_image',
        attribute='gen_ai.tool.call.result',
        redacted=REDACTED_IMAGE,
    ),
    Case(
        id='tool_return_message',
        build=image_returning_tool_agent,
        span_name='invoke_agent agent',
        attribute='pydantic_ai.all_messages',
        redacted=snapshot(
            [
                {'role': 'user', 'parts': [{'type': 'text', 'content': 'make an image'}]},
                {
                    'role': 'assistant',
                    'parts': [{'type': 'tool_call', 'id': IsStr(), 'name': 'gen_image', 'arguments': {}}],
                },
                {
                    'role': 'user',
                    'parts': [
                        {
                            'type': 'tool_call_response',
                            'id': IsStr(),
                            'name': 'gen_image',
                            'result': REDACTED_IMAGE,
                        }
                    ],
                },
                {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'a kiwi'}]},
            ]
        ),
    ),
    Case(
        id='final_result',
        build=image_output_agent,
        span_name='invoke_agent agent',
        attribute='final_result',
        redacted=REDACTED_IMAGE,
    ),
    Case(
        id='output_function_arguments',
        build=image_argument_output_function_agent,
        span_name='execute_tool final_result',
        attribute='gen_ai.tool.call.arguments',
        redacted=REDACTED_IMAGE,
    ),
    Case(
        # Reachable from the UI adapters, which rehydrate a native tool's prior result back into
        # `BinaryContent` when a client re-sends message history.
        id='native_tool_return_message',
        build=text_agent,
        span_name='invoke_agent agent',
        attribute='pydantic_ai.all_messages',
        redacted=snapshot(
            [
                {'role': 'user', 'parts': [{'type': 'text', 'content': 'draw a kiwi'}]},
                {
                    'role': 'assistant',
                    'parts': [
                        {
                            'type': 'tool_call_response',
                            'id': 'call-1',
                            'name': 'image_generation',
                            'builtin': True,
                            'result': REDACTED_IMAGE,
                        }
                    ],
                },
                {'role': 'user', 'parts': [{'type': 'text', 'content': 'make an image'}]},
                {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'a kiwi'}]},
            ]
        ),
        history=[
            ModelRequest(parts=[UserPromptPart(content='draw a kiwi')]),
            ModelResponse(
                parts=[
                    NativeToolReturnPart(
                        tool_name='image_generation',
                        tool_call_id='call-1',
                        content=IMAGE,
                        provider_name='openai',
                    )
                ]
            ),
        ],
    ),
]


async def run_and_read_attribute(case: Case, capfire: CaptureLogfire, *, include_binary_content: bool) -> Any:
    capfire.exporter.clear()
    agent = case.build(InstrumentationSettings(include_binary_content=include_binary_content))
    await agent.run('make an image', message_history=case.history)
    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    # The last matching span: the run's final model request is the one that has seen the image.
    attributes = [span['attributes'] for span in spans if span['name'] == case.span_name][-1]
    return attributes[case.attribute]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_binary_content_omitted_from_span_attribute(case: Case, capfire: CaptureLogfire) -> None:
    """Each attribute carries the image by default, and only its media type once binary is excluded.

    Asserting the default first keeps the case honest: if the attribute stopped carrying binary
    content altogether, the redaction assertion below would pass without proving anything.
    """
    included = await run_and_read_attribute(case, capfire, include_binary_content=True)
    assert IMAGE.base64 in json.dumps(included)

    assert await run_and_read_attribute(case, capfire, include_binary_content=False) == case.redacted


@pytest.mark.parametrize('context', [None, {}, {'include_binary_content': True}, {'unrelated': False}, 'not-a-mapping'])
def test_message_history_round_trip_preserves_binary_content(context: Any) -> None:
    """Binary data must survive any dump that isn't instrumentation asking for it to be excluded.

    `BinaryContent` omits its `data` only for the exact context instrumentation passes, so every
    other dump — message history above all — is untouched, including one carrying a user's own
    serialization context of whatever shape.
    """
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=['look at this', IMAGE])]),
        ModelResponse(parts=[FilePart(content=IMAGE)]),
        ModelRequest(parts=[ToolReturnPart(tool_name='gen_image', content=IMAGE, tool_call_id='1')]),
    ]

    dumped = ModelMessagesTypeAdapter.dump_json(messages, context=context)
    assert IMAGE.base64 in dumped.decode()
    assert ModelMessagesTypeAdapter.validate_json(dumped) == messages
