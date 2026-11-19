"""Every message part type is classified for how instrumentation renders it into OTel GenAI messages.

`InstrumentationSettings.messages_to_otel_messages` turns history into `gen_ai.input.messages`,
`gen_ai.output.messages` and `pydantic_ai.all_messages`. A request's consecutive parts that share a role make up one
message, so a part type that renders no OTel part but still takes a role becomes an empty message and splits the
messages around it. Each member of `ModelRequestPart` and `ModelResponsePart` has an entry here, with a part built
the way real code builds one, saying what it renders as at each instrumentation version, or why it renders as
nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import pytest

from pydantic_ai.messages import (
    BinaryContent,
    CapabilityInstructionSource,
    CapabilityLoadDeltaPart,
    CapabilityPart,
    CompactionPart,
    FilePart,
    InstructionBaselinePart,
    InstructionCapabilityPart,
    InstructionDeltaPart,
    InstructionId,
    InstructionPart,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelSettingsDeltaPart,
    NativeToolCallPart,
    NativeToolReturnPart,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    OperatorSource,
    RetryFeedbackPart,
    RetryInfo,
    RetryPromptPart,  # pyright: ignore[reportDeprecated]  # TODO(v3): remove RetryPromptPart
    SpeechPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.instrumented import InstrumentationSettings

from .conftest import legacy_retry_prompt_part
from .manifests import assert_classified, union_members

_PartT = TypeVar('_PartT', covariant=True)

Role = Literal['system', 'user', 'assistant', 'tool']


@dataclass(frozen=True)
class Rendered(Generic[_PartT]):
    """A part type traces show: at least one OTel part, in a message with the role each version gives it."""

    build: Callable[[], _PartT]
    v5: Role
    v6: Role
    reason: str


@dataclass(frozen=True)
class Silent(Generic[_PartT]):
    """A part type traces leave out on purpose: a message renders exactly as it would without the part.

    A response renders as one assistant message even with no parts, the envelope
    `test_chat_span_closed_for_contentless_response` in `tests/realtime/test_instrumentation.py` pins, so a silent
    response part leaves that envelope rather than no message.
    """

    build: Callable[[], _PartT]
    reason: str


@dataclass(kw_only=True)
class RequestsLeftPart(InstructionCapabilityPart, namespace='instrumentation_part_coverage'):
    """An instruction a capability enqueues mid-run, in the application's voice."""


_TOOL_RESPONSE_ROLE = (
    'A `tool_call_response`, which version 6 moves to the `tool` role the semantic conventions pair it with.'
)

_REQUEST_PARTS: dict[type, Rendered[ModelRequestPart] | Silent[ModelRequestPart]] = {
    SystemPromptPart: Rendered(
        lambda: SystemPromptPart('You are a weather assistant.'),
        v5='system',
        v6='system',
        reason="The application's prompt, in the system voice.",
    ),
    UserPromptPart: Rendered(
        lambda: UserPromptPart('What is the weather in Paris?'),
        v5='user',
        v6='user',
        reason="The user's turn.",
    ),
    SpeechPart: Rendered(
        lambda: SpeechPart(speaker='user', transcript='What is the weather in Paris?'),
        v5='user',
        v6='user',
        reason="A realtime user turn: its transcript renders as text and its retained audio as a blob, in the user's turn.",
    ),
    ToolSearchReturnPart: Rendered(
        lambda: ToolSearchReturnPart(content={'discovered_tools': [{'name': 'get_forecast'}]}, tool_call_id='call_1'),
        v5='user',
        v6='tool',
        reason=_TOOL_RESPONSE_ROLE,
    ),
    LoadCapabilityReturnPart: Rendered(
        lambda: LoadCapabilityReturnPart(content={'message': "Capability 'refunds' loaded."}, tool_call_id='call_1'),
        v5='user',
        v6='tool',
        reason=_TOOL_RESPONSE_ROLE,
    ),
    ToolReturnPart: Rendered(
        lambda: ToolReturnPart(tool_name='get_weather', content='Sunny, 22 degrees.', tool_call_id='call_1'),
        v5='user',
        v6='tool',
        reason=_TOOL_RESPONSE_ROLE,
    ),
    RetryPromptPart: Rendered(  # pyright: ignore[reportDeprecated]
        lambda: legacy_retry_prompt_part(content='City not found.', tool_name='get_weather', tool_call_id='call_1'),
        v5='user',
        v6='tool',
        reason=(
            'The legacy retry, still in the stored history `pydantic_ai.all_messages` records. One naming a tool is a '
            '`tool_call_response`, which version 6 moves to the `tool` role.'
        ),
    ),
    RetryFeedbackPart: Rendered(
        lambda: RetryFeedbackPart(content='Please return text or call a tool.', retry=RetryInfo('no_output')),
        v5='user',
        v6='system',
        reason=(
            "Built with the fixed `'no_output'` text the agent loop writes, which version 6 moves to the system voice "
            'it reaches the model in. Feedback that can quote the model stays `user` at every version.'
        ),
    ),
    ToolAvailabilityDeltaPart: Rendered(
        lambda: ToolAvailabilityDeltaPart(tools_added=['get_forecast']),
        v5='system',
        v6='system',
        reason='No role means "the set of tools changed", and `system` is the least misleading one.',
    ),
    CapabilityLoadDeltaPart: Rendered(
        lambda: CapabilityLoadDeltaPart(loaded=['refunds']),
        v5='system',
        v6='system',
        reason='No role means "the loaded capabilities changed", and `system` is the least misleading one.',
    ),
    ModelSettingsDeltaPart: Rendered(
        lambda: ModelSettingsDeltaPart(settings={'thinking': 'high'}),
        v5='system',
        v6='system',
        reason="A configuration change rather than anyone's words, so a `system` message.",
    ),
    InstructionDeltaPart: Rendered(
        lambda: InstructionDeltaPart(
            id=InstructionId(source=CapabilityInstructionSource(id='refunds')),
            content='Issue refunds only for orders under $100.',
        ),
        v5='system',
        v6='system',
        reason="A change to the operator's instructions, in the system voice.",
    ),
    InstructionBaselinePart: Silent(
        lambda: InstructionBaselinePart(instructions=[InstructionPart('You are a weather assistant.')]),
        reason=(
            'Records the instructions a request was sent with, which the request span reports as '
            '`gen_ai.system_instructions`. Rendered as a message it was an empty `system` one: '
            'https://github.com/pydantic/pydantic-ai/pull/10911.'
        ),
    ),
    CompactionPart: Rendered(
        lambda: CompactionPart(content='The user asked for the weather in Paris, which was sunny.'),
        v5='user',
        v6='user',
        reason=(
            "A summary the application or a capability wrote, such as Pydantic AI Harness's `SummarizingCompaction`, "
            'which models receive in the user voice.'
        ),
    ),
    CapabilityPart: Rendered(
        lambda: RequestsLeftPart(content='Two model requests left.', source=OperatorSource()),
        v5='system',
        v6='system',
        reason=(
            "Its role follows its effective authority: an instruction part with `'operator'` authority is a `system` "
            'message, a downgraded instruction or a data part a `user` one.'
        ),
    ),
}
"""How traces render each `ModelRequestPart` member, and a part to check it with."""

_RESPONSE_PARTS: dict[type, Rendered[ModelResponsePart] | Silent[ModelResponsePart]] = {
    TextPart: Rendered(
        lambda: TextPart('It is sunny in Paris.'),
        v5='assistant',
        v6='assistant',
        reason='A text part.',
    ),
    ToolSearchCallPart: Rendered(
        lambda: ToolSearchCallPart(args={'queries': ['weather forecast']}, tool_call_id='call_1'),
        v5='assistant',
        v6='assistant',
        reason='A `tool_call` part.',
    ),
    LoadCapabilityCallPart: Rendered(
        lambda: LoadCapabilityCallPart(args={'id': 'refunds'}, tool_call_id='call_1'),
        v5='assistant',
        v6='assistant',
        reason='A `tool_call` part.',
    ),
    ToolCallPart: Rendered(
        lambda: ToolCallPart('get_weather', {'city': 'Paris'}, tool_call_id='call_1'),
        v5='assistant',
        v6='assistant',
        reason='A `tool_call` part.',
    ),
    NativeToolSearchCallPart: Rendered(
        lambda: NativeToolSearchCallPart(
            args={'queries': ['weather forecast']}, tool_call_id='srvtoolu_1', provider_name='anthropic'
        ),
        v5='assistant',
        v6='assistant',
        reason='A `tool_call` part marked `builtin`.',
    ),
    NativeToolCallPart: Rendered(
        lambda: NativeToolCallPart(
            'web_search', {'query': 'weather in Paris'}, tool_call_id='ws_1', provider_name='openai'
        ),
        v5='assistant',
        v6='assistant',
        reason='A `tool_call` part marked `builtin`.',
    ),
    NativeToolSearchReturnPart: Rendered(
        lambda: NativeToolSearchReturnPart(
            content={'discovered_tools': [{'name': 'get_forecast'}]},
            tool_call_id='srvtoolu_1',
            provider_name='anthropic',
        ),
        v5='assistant',
        v6='assistant',
        reason='A `tool_call_response` part marked `builtin`, in the response that ran the tool.',
    ),
    NativeToolReturnPart: Rendered(
        lambda: NativeToolReturnPart(
            'web_search', {'status': 'completed'}, tool_call_id='ws_1', provider_name='openai'
        ),
        v5='assistant',
        v6='assistant',
        reason='A `tool_call_response` part marked `builtin`, in the response that ran the tool.',
    ),
    ThinkingPart: Rendered(
        lambda: ThinkingPart('The user wants the current weather.'),
        v5='assistant',
        v6='assistant',
        reason='A `thinking` part.',
    ),
    CompactionPart: Silent(
        lambda: CompactionPart(
            id='cmp_1', provider_name='openai', provider_details={'encrypted_content': 'gAAAAB-compacted'}
        ),
        reason="A provider's compaction item, replayed only to that provider: no OTel GenAI part type represents it.",
    ),
    FilePart: Rendered(
        lambda: FilePart(content=BinaryContent(data=b'\x89PNG\r\n\x1a\n', media_type='image/png')),
        v5='assistant',
        v6='assistant',
        reason='A generated file, as a `blob` part.',
    ),
    SpeechPart: Rendered(
        lambda: SpeechPart(speaker='assistant', transcript='It is sunny in Paris.'),
        v5='assistant',
        v6='assistant',
        reason='A realtime reply: its transcript renders as text and its retained audio as a blob.',
    ),
}
"""How traces render each `ModelResponsePart` member, and a part to check it with."""


def test_every_part_type_has_an_instrumentation_entry() -> None:
    """A new member of either part union fails here until its entry says how traces render it.

    Unit test because it walks the part unions, which no recorded run covers.
    """
    assert_classified(union_members(ModelRequestPart), _REQUEST_PARTS, complete_walk=True, table_name='_REQUEST_PARTS')
    assert_classified(
        union_members(ModelResponsePart), _RESPONSE_PARTS, complete_walk=True, table_name='_RESPONSE_PARTS'
    )


def _assert_renders_as_classified(
    part_type: type,
    entry: Rendered[_PartT] | Silent[_PartT],
    members: set[type],
    wrap: Callable[[list[_PartT]], ModelMessage],
    neighbour: _PartT,
    version: Literal[5, 6],
) -> None:
    part = entry.build()
    # Several members subclass another, so a part checks its entry only if no member below `part_type` claims it.
    assert {member for member in members if isinstance(part, member) and issubclass(member, part_type)} == {
        part_type
    }, f'The `{part_type.__name__}` entry builds a `{type(part).__name__}`.'
    render = InstrumentationSettings(version=version).messages_to_otel_messages
    if isinstance(entry, Silent):
        assert render([wrap([neighbour, part, neighbour])]) == render([wrap([neighbour, neighbour])]), (
            f'`{part_type.__name__}` is classified silent, but changes the messages around it.'
        )
        assert render([wrap([part])]) == render([wrap([])]), (
            f'`{part_type.__name__}` is classified silent, but renders a message of its own.'
        )
    else:
        role = entry.v5 if version == 5 else entry.v6
        assert [(message['role'], bool(message['parts'])) for message in render([wrap([part])])] == [(role, True)], (
            f'`{part_type.__name__}` should render one `{role}` message with at least one part.'
        )


@pytest.mark.parametrize('version', [5, 6])
@pytest.mark.parametrize('part_type', list(_REQUEST_PARTS), ids=[t.__name__ for t in _REQUEST_PARTS])
def test_request_part_renders_as_classified(part_type: type, version: Literal[5, 6]) -> None:
    """A request part renders in its classified role, or leaves the user's turn around it whole.

    Unit test because no recorded run produces every part type, and a cassette holds the request, not the trace.
    """
    _assert_renders_as_classified(
        part_type,
        _REQUEST_PARTS[part_type],
        union_members(ModelRequestPart),
        lambda parts: ModelRequest(parts=parts),
        UserPromptPart('What is the weather in Paris?'),
        version,
    )


@pytest.mark.parametrize('version', [5, 6])
@pytest.mark.parametrize('part_type', list(_RESPONSE_PARTS), ids=[t.__name__ for t in _RESPONSE_PARTS])
def test_response_part_renders_as_classified(part_type: type, version: Literal[5, 6]) -> None:
    """A response part renders at least one OTel part, or leaves the response's other parts as they were.

    Unit test because no recorded run produces every part type, and a cassette holds the response, not the trace.
    """
    _assert_renders_as_classified(
        part_type,
        _RESPONSE_PARTS[part_type],
        union_members(ModelResponsePart),
        lambda parts: ModelResponse(parts=parts),
        TextPart('It is sunny in Paris.'),
        version,
    )
