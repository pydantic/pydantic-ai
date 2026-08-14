"""`InstrumentationSettings(include_binary_content=False)` must not leak base64 into any span.

These run against `FunctionModel` rather than a cassette: the leak is in our own OTel serialization
of `BinaryContent`, which is provider-independent, so a recording would only add a base64 image
payload to the repo without exercising anything the fake model doesn't.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    BinaryImage,
    CallDeferred,
    DeferredToolRequests,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolReturnPart,
    TextPart,
    ToolCallPart,
    ToolReturn,
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

REDACTED_IMAGE: dict[str, Any] = {
    'media_type': 'image/png',
    'vendor_metadata': None,
    'kind': 'binary',
    # Pinned rather than matched: `identifier` defaults to a hash of the data, so it survives as a
    # fingerprint of the very bytes the flag excludes. `docs/logfire.md` says so; a matcher here
    # would hide it from anyone reading the cases.
    'identifier': '734658',
}
"""The shape a `BinaryImage` is recorded as once its data is excluded: everything but `data`."""


class IsSingleEntryDict:
    """Matches a one-entry dict whose only value equals `value`, without pinning its key.

    `dirty_equals` matchers like `IsStr()` are unhashable, so they can't stand in for a dict key
    that's a randomly generated `tool_call_id` (e.g. `DeferredToolRequests.metadata`).
    """

    def __init__(self, value: Any) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, dict):
            return False  # pragma: no cover
        entries: dict[str, Any] = other  # pyright: ignore[reportUnknownVariableType]
        return len(entries) == 1 and next(iter(entries.values())) == self.value

    def __repr__(self) -> str:
        return f'IsSingleEntryDict({self.value!r})'  # pragma: no cover


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


def tool_return_agent(settings: InstrumentationSettings) -> Agent[None, str]:
    """An agent whose tool wraps the image in a `ToolReturn`, in each of its value fields."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('gen_image', {})])
        return ModelResponse(parts=[TextPart('a kiwi')])

    agent = Agent(FunctionModel(respond), capabilities=[Instrumentation(settings=settings)], name='agent')

    # `vendor_metadata` is typed `dict[str, Any]`, so it can hold binary content of its own.
    thumbnailed = BinaryImage(data=IMAGE.data, media_type='image/png', vendor_metadata={'thumbnail': IMAGE})

    @agent.tool_plain
    def gen_image() -> ToolReturn:
        return ToolReturn(return_value=thumbnailed, content=['here it is', IMAGE], metadata={'img': IMAGE})

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


def image_returning_output_function_agent(settings: InstrumentationSettings) -> Agent[None, BinaryImage]:
    """An agent whose output function *returns* an image, recorded on its own tool span."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('final_result', {'media_type': 'image/png'})])

    def render(media_type: str) -> BinaryImage:
        return IMAGE

    return Agent(
        FunctionModel(respond), capabilities=[Instrumentation(settings=settings)], output_type=render, name='agent'
    )


def text_agent(settings: InstrumentationSettings) -> Agent[None, str]:
    """An agent that just answers, for cases where the image enters through message history."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('a kiwi')])

    return Agent(FunctionModel(respond), capabilities=[Instrumentation(settings=settings)], name='agent')


def deferring_tool_agent(settings: InstrumentationSettings) -> Agent[None, Any]:
    """An agent whose tool defers with an image attached, the way an approval UI would want it."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('defer_image', {})])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[Instrumentation(settings=settings)],
        output_type=[str, DeferredToolRequests],
        name='agent',
    )

    @agent.tool_plain
    def defer_image() -> str:
        raise CallDeferred(metadata={'img': IMAGE})

    return agent


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
    metadata: dict[str, Any] | None = None
    """Passed to `Agent.run`, for the attribute that records the run's own metadata."""


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
        # `ToolReturn` is the framework's own wrapper, so it's walked into: a user's own model
        # holding a `BinaryImage` is not, and would still carry the image here.
        id='tool_return_result',
        build=tool_return_agent,
        span_name='execute_tool gen_image',
        attribute='gen_ai.tool.call.result',
        redacted=snapshot(
            {
                'return_value': {**REDACTED_IMAGE, 'vendor_metadata': {'thumbnail': REDACTED_IMAGE}},
                'content': ['here it is', REDACTED_IMAGE],
                'metadata': {'img': REDACTED_IMAGE},
                'tools': None,
                'kind': 'tool-return',
            }
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
        # The other attribute on the output function's own span: what it returned, as opposed to
        # the arguments case above, which reads `gen_ai.tool.call.arguments` on the same span.
        id='output_function_result',
        build=image_returning_output_function_agent,
        span_name='execute_tool final_result',
        attribute='gen_ai.tool.call.result',
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
    Case(
        # The OTel-spec attribute the message sinks feed, which travels a different route than
        # `pydantic_ai.all_messages`: through the per-run message JSON cache rather than one dump.
        id='input_messages',
        build=image_returning_tool_agent,
        span_name='chat function:respond:',
        attribute='gen_ai.input.messages',
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
            ]
        ),
    ),
    Case(
        id='run_metadata',
        build=text_agent,
        span_name='invoke_agent agent',
        attribute='metadata',
        redacted=snapshot({'img': REDACTED_IMAGE}),
        metadata={'img': IMAGE},
    ),
    Case(
        id='deferral_metadata',
        build=deferring_tool_agent,
        span_name='execute_tool defer_image',
        attribute='pydantic_ai.tool.deferral.metadata',
        redacted=snapshot({'img': REDACTED_IMAGE}),
    ),
    Case(
        # The same metadata the deferral span above records, reaching the run's own output through
        # `DeferredToolRequests`: redacting one span and not its peer would defeat the flag.
        id='deferred_requests_output',
        build=deferring_tool_agent,
        span_name='invoke_agent agent',
        attribute='final_result',
        redacted=snapshot(
            {
                'calls': [
                    {
                        'tool_name': 'defer_image',
                        'args': {},
                        'tool_call_id': IsStr(),
                        'tool_kind': None,
                        'id': None,
                        'provider_name': None,
                        'provider_details': None,
                        'part_kind': 'tool-call',
                    }
                ],
                'approvals': [],
                'metadata': IsSingleEntryDict({'img': REDACTED_IMAGE}),
            }
        ),
    ),
]


async def run_and_read_attribute(case: Case, capfire: CaptureLogfire, *, include_binary_content: bool) -> Any:
    capfire.exporter.clear()
    agent = case.build(InstrumentationSettings(include_binary_content=include_binary_content))
    await agent.run('make an image', message_history=case.history, metadata=case.metadata)
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


def output_agent(output: Callable[[], Any]) -> Agent[None, Any]:
    """An agent whose output function returns a value the redaction walk has to survive."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('final_result', {})])

    return Agent(
        FunctionModel(respond),
        capabilities=[Instrumentation(settings=InstrumentationSettings(include_binary_content=False))],
        output_type=output,
        name='agent',
    )


async def test_self_referential_output_is_recorded_without_its_binary(capfire: CaptureLogfire) -> None:
    """A value that contains itself must neither crash the run nor smuggle the image out.

    The walk marks a repeat visit instead of recursing, so the attribute is still redacted. Handing
    the value back at that point would have defeated the flag: the caller falls back to `str`, whose
    `BinaryContent` repr prints the data.
    """

    def cycle() -> dict[str, Any]:
        output: dict[str, Any] = {'img': IMAGE}
        output['self'] = output
        return output

    result = await output_agent(cycle).run('make an image')
    assert result.output['self'] is result.output

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    assert [span['attributes']['final_result'] for span in spans if span['name'] == 'invoke_agent agent'] == snapshot(
        [{'img': REDACTED_IMAGE, 'self': '<circular reference>'}]
    )


async def test_output_the_walk_cannot_traverse_does_not_crash_the_run(capfire: CaptureLogfire) -> None:
    """Instrumentation must not fail a run over a container that objects to being walked.

    Without the flag such a value reaches `serialize_any`, which falls back rather than raising, so
    turning binary exclusion on must not make the same value fatal.
    """

    class Detached(Mapping[str, Any]):
        def __iter__(self) -> Iterator[str]:
            raise RuntimeError('cannot iterate detached rows')

        def __getitem__(self, key: str) -> Any:
            raise KeyError(key)  # pragma: no cover

        def __len__(self) -> int:
            return 0  # pragma: no cover

    def detached() -> dict[str, Any]:
        return {'rows': Detached()}

    await output_agent(detached).run('make an image')

    spans = capfire.exporter.exported_spans_as_dict()
    assert [span['attributes']['final_result'] for span in spans if span['name'] == 'invoke_agent agent'] == snapshot(
        ['"Unable to redact binary content: RuntimeError"']
    )


async def test_the_walks_own_failure_does_not_report_a_binary_carrying_message(capfire: CaptureLogfire) -> None:
    """The fallback names the exception's type, never its message.

    An exception raised out of a user's container carries a user-controlled message, which can embed
    a `BinaryContent` whose repr prints the data. Interpolating it would have leaked through the very
    branch that exists to keep the flag honored when the walk can't finish.
    """

    class Guarded(Mapping[str, Any]):
        def __iter__(self) -> Iterator[str]:
            raise RuntimeError(IMAGE)

        def __getitem__(self, key: str) -> Any:
            raise KeyError(key)  # pragma: no cover

        def __len__(self) -> int:
            return 0  # pragma: no cover

    def guarded() -> dict[str, Any]:
        return {'rows': Guarded()}

    await output_agent(guarded).run('make an image')

    spans = capfire.exporter.exported_spans_as_dict()
    attribute = next(span['attributes']['final_result'] for span in spans if span['name'] == 'invoke_agent agent')
    assert attribute == snapshot('"Unable to redact binary content: RuntimeError"')
    assert 'kiwi' not in attribute


def test_binary_nested_in_a_user_type_is_not_redacted(capfire: CaptureLogfire) -> None:
    """The documented boundary: the walk doesn't rebuild types it doesn't own.

    Redacting a field of a user's model would mean reconstructing it, changing how everything else
    in the attribute serializes. `docs/logfire.md` says so; this pins that the docs stay true.
    """

    class Wrapper(BaseModel):
        image: BinaryImage

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart('final_result', {'image': {'data': IMAGE.base64, 'media_type': 'image/png'}})]
        )

    agent = Agent(
        FunctionModel(respond),
        capabilities=[Instrumentation(settings=InstrumentationSettings(include_binary_content=False))],
        output_type=Wrapper,
        name='agent',
    )

    agent.run_sync('make an image')

    spans = capfire.exporter.exported_spans_as_dict()
    final_result = [span['attributes']['final_result'] for span in spans if span['name'] == 'invoke_agent agent']
    assert IMAGE.base64 in json.dumps(final_result)


def test_redacted_tool_return_keeps_the_tools_it_made_available(capfire: CaptureLogfire) -> None:
    """Redacting a `ToolReturn` must not drop the deferred tools it revealed.

    `ToolReturn.tools` names the tools the call made available, and telemetry is where that
    reveal is observable — dropping it would hide why a later tool call became legal.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('gen_image', {})])
        return ModelResponse(parts=[TextPart('a kiwi')])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[Instrumentation(settings=InstrumentationSettings(include_binary_content=False))],
        name='agent',
    )

    @agent.tool_plain
    def gen_image() -> ToolReturn:
        return ToolReturn(return_value='done', metadata={'img': IMAGE}, tools=['lookup_refund_policy'])

    agent.run_sync('make an image')

    spans = capfire.exporter.exported_spans_as_dict()
    tool_spans = json.dumps([span['attributes'] for span in spans if span['name'] == 'execute_tool gen_image'])
    assert 'lookup_refund_policy' in tool_spans
    # The binary the same `ToolReturn` carried is still redacted.
    assert IMAGE.base64 not in tool_spans


def test_redacted_shapes_keep_every_field_but_the_data() -> None:
    """A field added to a redacted type has to be added to its redacted shape too.

    The redaction spells each type's fields out rather than dumping and dropping `data`, so a new
    field would otherwise silently stop reaching telemetry. This pins the field sets it mirrors.
    """
    dumped = ModelMessagesTypeAdapter.dump_python([ModelRequest(parts=[UserPromptPart(content=[IMAGE])])], mode='json')
    assert set(dumped[0]['parts'][0]['content'][0]) == set(REDACTED_IMAGE) | {'data'}

    assert {f.name for f in fields(ToolReturn)} == {'return_value', 'content', 'metadata', 'tools', 'kind'}
    assert {f.name for f in fields(DeferredToolRequests)} == {'calls', 'approvals', 'metadata'}


def test_message_history_round_trip_preserves_binary_content() -> None:
    """Binary data must survive a message history dump: the redaction is instrumentation's alone.

    `BinaryContent` serializes the same way it always did — telemetry redacts the values it's about
    to record rather than changing how the type dumps — so message history is untouched.
    """
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=['look at this', IMAGE])]),
        ModelResponse(parts=[FilePart(content=IMAGE)]),
        ModelRequest(parts=[ToolReturnPart(tool_name='gen_image', content=IMAGE, tool_call_id='1')]),
    ]

    dumped = ModelMessagesTypeAdapter.dump_json(messages)
    assert IMAGE.base64 in dumped.decode()
    assert ModelMessagesTypeAdapter.validate_json(dumped) == messages
