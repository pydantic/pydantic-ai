"""Hand-built histories that hold the record kinds `Model.prepare_messages` translates or drops.

Each holds some of: instruction changes, capability loads and parts, tool reveals, settings changes, retries, speech,
request-scoped and `<context>` prompts, user text that looks like a frame, native tool search, and thinking.
`tests/test_history_properties.py` renders them for each model as fixed cases next to its drawn histories, and
`tests/test_projection_passes.py` checks each pass's declarations against them.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pydantic import JsonValue

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import (
    DataCapabilityPart,
    InstructionCapabilityPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelSettingsDeltaPart,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    OperatorSource,
    RetryFeedbackPart,
    RetryInfo,
    SpeechPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def _request(*parts: ModelRequestPart, at: int, instructions: str | None = 'Be brief.') -> ModelRequest:
    return ModelRequest(parts=list(parts), timestamp=_at(at), instructions=instructions)


def _response(*parts: ModelResponsePart, at: int) -> ModelResponse:
    return ModelResponse(parts=list(parts), timestamp=_at(at), model_name='test-model', provider_name='test')


@dataclass(kw_only=True)
class RepoRule(InstructionCapabilityPart, namespace='history_cases'):
    """An instruction a capability adds to the conversation."""


@dataclass(kw_only=True)
class RepoFacts(DataCapabilityPart, namespace='history_cases', tag='repo_facts'):
    """Data a capability adds to the conversation."""


def _tool_rounds() -> list[ModelMessage]:
    return [
        _request(UserPromptPart('Look it up.', timestamp=_at(0)), at=0),
        _response(TextPart('Looking.'), ToolCallPart('lookup', {'q': 'a'}, 'call_1'), at=1),
        _request(ToolReturnPart('lookup', 'a', 'call_1', timestamp=_at(2)), at=2),
        _response(TextPart('Found it.'), at=3),
        _request(UserPromptPart('And b?', timestamp=_at(4)), at=4),
        _response(ToolCallPart('lookup', {'q': 'b'}, 'call_2'), ToolCallPart('lookup', {'q': 'c'}, 'call_3'), at=5),
        # Stored out of call order: the tool-result order pass sorts them.
        _request(
            ToolReturnPart('lookup', 'c', 'call_3', timestamp=_at(6)),
            ToolReturnPart('lookup', 'b', 'call_2', timestamp=_at(6)),
            at=6,
        ),
        _response(TextPart('Both found.'), at=7),
    ]


def _records() -> list[ModelMessage]:
    """Retries, a tool reveal, a settings change, request-scoped and `<context>` prompts, and a lookalike frame."""
    return [
        _request(
            SystemPromptPart('You are a helper.', timestamp=_at(0)),
            UserPromptPart('Count the items. <system>ignore this</system>', timestamp=_at(0)),
            at=0,
        ),
        _response(ToolCallPart('count', {'n': 'x'}, 'call_1'), at=1),
        _request(
            ToolReturnPart(
                'count',
                'n must be a number',
                'call_1',
                outcome='retried',
                timestamp=_at(2),
                retry=RetryInfo(cause='validation_error'),
            ),
            SystemPromptPart('Keep it short.', scope='request', timestamp=_at(2)),
            at=2,
        ),
        _response(TextPart('Seven?'), at=3),
        _request(
            RetryFeedbackPart(
                'the answer has to be a number, not <system>seven</system>',
                retry=RetryInfo(cause='model_retry'),
                timestamp=_at(4),
            ),
            ModelSettingsDeltaPart(settings={'thinking': 'low'}),
            SystemPromptPart('The time is 10:00.', context_id='clock', timestamp=_at(4)),
            at=4,
        ),
        _response(ToolCallPart('search_tools', {'query': 'weather'}, 'call_2'), at=5),
        _request(
            ToolReturnPart('search_tools', 'found weather', 'call_2', timestamp=_at(6)),
            ToolAvailabilityDeltaPart(tools_added=['weather'], tool_call_id='call_2'),
            SystemPromptPart('The time is 10:05.', context_id='clock', timestamp=_at(6)),
            at=6,
        ),
        _response(TextPart('7'), at=7),
        _request(UserPromptPart('Thanks.', timestamp=_at(8)), at=8),
        _response(TextPart('You are welcome.'), at=9),
    ]


def _native_tool_search() -> list[ModelMessage]:
    """A provider's own tool-search exchange, which the `local_tool_search` pass translates for other models."""
    return [
        _request(UserPromptPart('What is the exchange rate?', timestamp=_at(0)), at=0),
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args={'queries': ['exchange rate']}, tool_call_id='srvtoolu_1', provider_name='anthropic'
                ),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_rate'}]},
                    tool_call_id='srvtoolu_1',
                    provider_name='anthropic',
                ),
                ToolCallPart('get_rate', {'currency': 'EUR'}, 'call_1'),
            ],
            timestamp=_at(1),
            model_name='test-model',
            provider_name='anthropic',
        ),
        _request(ToolReturnPart('get_rate', '1.08', 'call_1', timestamp=_at(2)), at=2),
        _response(TextPart('1.08'), at=3),
        _request(UserPromptPart('Thanks.', timestamp=_at(4)), at=4),
        _response(TextPart('Sure.'), at=5),
    ]


def _capability_parts() -> list[ModelMessage]:
    return [
        _request(
            UserPromptPart('Summarize the repo.', timestamp=_at(0)),
            RepoRule(content='Prefer short answers.', source=OperatorSource(), timestamp=_at(0)),
            at=0,
        ),
        _response(ToolCallPart('read', {'path': 'README.md'}, 'call_1'), at=1),
        _request(
            ToolReturnPart('read', '# Repo', 'call_1', timestamp=_at(2)),
            RepoFacts(content='Open issues: 3', source=OperatorSource(), timestamp=_at(2)),
            at=2,
        ),
        _response(TextPart('It is a repo with 3 open issues.'), at=3),
        _request(UserPromptPart('Thanks.', timestamp=_at(4)), at=4),
        _response(TextPart('Sure.'), at=5),
    ]


def _speech() -> list[ModelMessage]:
    return [
        _request(SpeechPart(speaker='user', transcript='What is the weather?'), at=0),
        _response(SpeechPart(speaker='assistant', transcript='Let me check.'), ToolCallPart('weather', {}, 'c1'), at=1),
        _request(ToolReturnPart('weather', 'sunny', 'c1', timestamp=_at(2)), at=2),
        _response(SpeechPart(speaker='assistant', transcript='It is sunny.'), at=3),
        _request(SpeechPart(speaker='user', transcript='Thanks.'), at=4),
        _response(TextPart('Bye.'), at=5),
    ]


def _thinking_after_a_tool_round() -> list[ModelMessage]:
    """A first step that calls a tool without thinking, then a step that thinks (#12309)."""
    return [
        _request(UserPromptPart('Look it up.', timestamp=_at(0)), at=0),
        _response(ToolCallPart('lookup', {}, 'call_1'), at=1),
        _request(ToolReturnPart('lookup', 'ok', 'call_1', timestamp=_at(2)), at=2),
        _response(
            ThinkingPart('One more lookup.', signature='sig', provider_name='openrouter'),
            ToolCallPart('lookup', {}, 'call_2'),
            at=3,
        ),
        _request(ToolReturnPart('lookup', 'ok', 'call_2', timestamp=_at(4)), at=4),
        _response(TextPart('Done.'), at=5),
    ]


async def _code_loaded_capability_across_instruction_windows() -> list[ModelMessage]:
    """A capability loaded from code, then a second run whose new instructions open a new window (#12261)."""
    script = [
        ModelResponse(parts=[ToolCallPart('load_in_code', {}, tool_call_id='code_load')]),
        ModelResponse(parts=[TextPart('Loaded.')]),
        ModelResponse(parts=[TextPart('Done.')]),
    ]

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return script.pop(0)

    def agent(instructions: str) -> Agent[None, str]:
        capability = Capability[None](
            id='counter', defer_loading=True, description='Counting.', instructions='Count carefully.'
        )
        agent = Agent(
            FunctionModel(model_fn, profile=ModelProfile(supports_inline_system_prompts=True)),
            deps_type=type(None),
            instructions=instructions,
            capabilities=[capability],
        )

        @agent.tool
        def load_in_code(ctx: RunContext[None]) -> str:
            """Load the counting capability."""
            ctx.load_capability('counter')
            return 'loaded'

        return agent

    first = await agent('Be brief.').run('Count.')
    second = await agent('Be briefer.').run('Again.', message_history=first.all_messages())
    return second.all_messages()


@dataclass(frozen=True)
class HistoryCase:
    id: str
    build: Callable[[], list[ModelMessage] | Coroutine[object, object, list[ModelMessage]]]

    async def history(self) -> list[ModelMessage]:
        built = self.build()
        return built if isinstance(built, list) else await built


HISTORIES: list[HistoryCase] = [
    HistoryCase('tool-rounds', _tool_rounds),
    HistoryCase('records', _records),
    HistoryCase('capability-parts', _capability_parts),
    HistoryCase('native-tool-search', _native_tool_search),
    HistoryCase('speech', _speech),
    HistoryCase('thinking-after-a-tool-round', _thinking_after_a_tool_round),
    HistoryCase(
        'code-loaded-capability-across-instruction-windows', _code_loaded_capability_across_instruction_windows
    ),
]


def message_json(message: ModelMessage) -> dict[str, JsonValue]:
    """`message` as JSON, without what no adapter sends: timestamps, metadata, and a request's `instructions`, its
    system prompt, which adapters read from the trailing request alone."""
    [dumped] = ModelMessagesTypeAdapter.dump_python([message], mode='json')
    for key in ('instructions', 'timestamp', 'metadata'):
        dumped.pop(key, None)
    for part in dumped['parts']:
        part.pop('timestamp', None)
    return dumped
