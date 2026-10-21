"""Round trips through the real Vercel AI and AG-UI clients.

A UI app keeps the conversation in the frontend and sends all of it back every turn, so what the model sees
on the next turn is what the client kept. These tests run scripted agent conversations through each adapter
and the protocol's real JavaScript client (see `conftest.py`), on two paths:

- **dump**: before every turn, the app hydrates the client with `dump_messages()` of the server's history, as
  on a page reload;
- **stream**: the client builds its messages from the event stream, as during a live chat.

Three properties must hold on each path:

1. **Load after dump** (`test_load`): `load_messages()` of what a hydrated client sends back projects
   identically to the history it was hydrated with.
2. **Load after stream** (`test_load`): `load_messages()` of what a streaming client sends back projects
   identically to the server's `all_messages()` for the turn.
3. **Wire request** (`test_wire_request`): every provider request the agent makes through the client equals
   the one a server keeping its own history makes. The agent runs on the real Anthropic and OpenAI Chat
   models, on mocked SDK clients, so each request is prepared and rendered for that provider as in production.

Properties 1 and 2 compare histories after the adapter's `sanitize_messages()`, so parts the server owns and
never trusts from a client (its system prompt, by default) count as kept. The projection drops only part fields
that never reach the model, `timestamp` and `metadata`, and the message-level ids and timestamps a reload re-mints.

Python ports of the client reducers are no substitute: one ignored an event the real AG-UI client handles,
and reported a loss that real clients don't have.
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import anyio
import pytest
from pydantic_core import to_jsonable_python

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability, Capability
from pydantic_ai.messages import (
    BinaryContent,
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
    _merge_consecutive_messages,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.settings import ModelSettings

from ..conftest import try_import
from .conftest import JSDriver

with try_import() as imports_successful:
    from anthropic.types.beta import BetaTextBlock, BetaUsage
    from openai.types.chat import ChatCompletionMessage

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.ui import UIAdapter
    from pydantic_ai.ui.ag_ui import AGUIAdapter
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter

    from ..models.mock_openai import (
        MockOpenAI,
        completion_message as openai_completion,
        get_mock_chat_completion_kwargs as openai_request_kwargs,
    )
    from ..models.test_anthropic import (
        MockAnthropic,
        completion_message as anthropic_completion,
        get_mock_chat_completion_kwargs as anthropic_request_kwargs,
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='UI, OpenAI or Anthropic extras not installed'),
    pytest.mark.ui_roundtrip,
    # A hydrated client sends back the system prompt `dump_messages()` wrote; the adapter strips it, says so,
    # and reinjects the agent's own.
    pytest.mark.filterwarnings('ignore:Parts carrying the system voice were stripped:UserWarning'),
]

Protocol = Literal['vercel', 'agui']
Path = Literal['dump', 'stream']
Property = Literal['load', 'wire']
PROTOCOLS: tuple[Protocol, ...] = ('vercel', 'agui')
PATHS: tuple[Path, ...] = ('dump', 'stream')


# --- Projection: what the model can see ---------------------------------------------------------------------------

# Never sent to the model: `metadata` is application data, and timestamps are re-minted on load.
_INVISIBLE_PART_FIELDS = frozenset({'timestamp', 'metadata', 'part_kind'})


def _project_part(part: ModelRequestPart | ModelResponsePart) -> dict[str, Any]:
    dumped: dict[str, Any] = to_jsonable_python(part, bytes_mode='base64')
    if isinstance(part, ToolCallPart):
        dumped['args'] = part.args_as_dict()
    return {'kind': part.part_kind} | {
        key: value for key, value in dumped.items() if key not in _INVISIBLE_PART_FIELDS and value is not None
    }


def project(messages: Sequence[ModelMessage]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Message boundaries and the model-visible fields of each part.

    Two differences that never reach the model are normalized away first, as the agent and the models do:
    consecutive requests are merged, as the agent does to any history it continues, and tool results are put
    in the order of the calls they answer, as every model sends them (a streaming AG-UI client keeps them in
    the order the tools finished).
    """
    projected: list[tuple[str, list[dict[str, Any]]]] = []
    call_order: dict[str, int] = {}
    for message in _merge_consecutive_messages(list(messages)):
        parts = list(message.parts)
        if isinstance(message, ModelResponse):
            call_order = {part.tool_call_id: i for i, part in enumerate(parts) if isinstance(part, ToolCallPart)}
        else:
            parts.sort(
                key=lambda part: call_order.get(
                    part.tool_call_id if isinstance(part, ToolReturnPart) else '', len(call_order)
                )
            )
        projected.append((message.kind, [_project_part(part) for part in parts]))
    return projected


# --- Scripted conversations ---------------------------------------------------------------------------------------


Script = Callable[[list[ModelMessage], AgentInfo], ModelResponse]


def _results_since_question(messages: list[ModelMessage]) -> int:
    """How many tool results, retries included, the model has seen since the user last asked a question.

    Enqueued user content isn't a question: the scenario prompts all start with `Question`.
    """
    count = 0
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            if any(
                isinstance(part, UserPromptPart)
                and isinstance(part.content, str)
                and part.content.startswith('Question')
                for part in message.parts
            ):
                break
            count += sum(isinstance(part, ToolReturnPart) for part in message.parts)
    return count


async def _stream_response(response: ModelResponse) -> AsyncIterator[str | DeltaToolCalls | DeltaThinkingCalls]:
    """Stream a scripted response in deltas, splitting text and tool arguments across chunks."""
    if not response.parts:
        yield ''
    for index, part in enumerate(response.parts):
        if isinstance(part, TextPart):
            middle = len(part.content) // 2
            yield part.content[:middle]
            yield part.content[middle:]
        elif isinstance(part, ThinkingPart):
            yield {index: DeltaThinkingPart(content=part.content)}
            yield {index: DeltaThinkingPart(signature=part.signature)}
        else:
            assert isinstance(part, ToolCallPart), f'`FunctionModel` cannot stream {part!r}'
            args = part.args_as_json_str()
            middle = len(args) // 2
            yield {index: DeltaToolCall(name=part.tool_name, json_args=args[:middle], tool_call_id=part.tool_call_id)}
            yield {index: DeltaToolCall(json_args=args[middle:])}


WireProvider = Literal['anthropic', 'openai']
WIRE_PROVIDERS: tuple[WireProvider, ...] = ('anthropic', 'openai')
GapKey = tuple[Property, Path, Protocol] | tuple[Property, Path, Protocol, WireProvider]


@dataclass
class ModelCall:
    """One model request, as the provider SDK would have received it."""

    body: str


class _ProviderWire(WrapperModel):
    """A real provider model on a mocked SDK client, answering with the scenario's script.

    The agent prepares each request for the provider model, as it would in production, and the provider
    model maps it onto its mocked client, which records the request body. The script's `FunctionModel`
    then streams the answer, so the conversation goes on exactly as the scripted one does.
    """

    def __init__(self, provider: WireProvider, script_model: FunctionModel, calls: list[ModelCall]) -> None:
        model: Model
        if provider == 'anthropic':
            anthropic_client = MockAnthropic.create_mock(
                anthropic_completion(
                    [BetaTextBlock(text='OK', type='text')], BetaUsage(input_tokens=1, output_tokens=1)
                )
            )
            model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=anthropic_client))
            self._sent: Callable[[], list[dict[str, Any]]] = lambda: anthropic_request_kwargs(anthropic_client)
            self._body_keys = ('system', 'messages', 'tools')
        else:
            openai_client = MockOpenAI.create_mock(
                openai_completion(ChatCompletionMessage(content='OK', role='assistant'))
            )
            model = OpenAIChatModel('gpt-5', provider=OpenAIProvider(openai_client=openai_client))
            self._sent = lambda: openai_request_kwargs(openai_client)
            self._body_keys = ('messages', 'tools')
        super().__init__(model)
        self._script_model = script_model
        self._calls = calls

    async def _record(
        self, messages: list[ModelMessage], settings: ModelSettings | None, params: ModelRequestParameters
    ) -> None:
        await self.wrapped.request(messages, settings, params)
        kwargs = self._sent()[-1]
        body = {key: kwargs.get(key) for key in self._body_keys}
        self._calls.append(ModelCall(json.dumps(body, default=str, indent=1)))

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:  # pragma: no cover
        raise AssertionError('UI adapters and the server baseline both stream')

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        await self._record(messages, model_settings, model_request_parameters)
        async with self._script_model.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as response:
            yield response


@dataclass(frozen=True)
class Scenario:
    id: str
    script: Script
    agent_kwargs: dict[str, Any] = field(default_factory=dict[str, Any])
    setup: Callable[[Agent[None, str]], None] = lambda agent: None
    adapter_kwargs: dict[str, Any] = field(default_factory=dict[str, Any])
    seed: list[ModelMessage] = field(default_factory=list[ModelMessage])
    """History the conversation continues, hydrated into the client on both paths.

    For parts `FunctionModel` can't produce: only provider models emit a `CompactionPart`.
    """
    prompts: tuple[str, ...] = ('Question 1', 'Question 2')
    gaps: dict[GapKey, str] = field(default_factory=dict[GapKey, str])
    """Known round-trip losses, as `(property, path, protocol)` -> a reason naming the issue or PR.

    A wire gap that only one provider's rendering shows adds that provider to the key.
    """

    def agent(self, provider: WireProvider | None = None) -> tuple[Agent[None, str], list[ModelCall]]:
        """A fresh agent answering with the script, and the list each of its provider requests is recorded in.

        With a `provider`, the agent runs on that provider's real model, so requests are prepared and rendered
        for it; without one, on the script's `FunctionModel`, and nothing is recorded.
        """
        calls: list[ModelCall] = []

        async def stream_function(messages: list[ModelMessage], info: AgentInfo):
            async for item in _stream_response(self.script(messages, info)):
                yield item

        script_model = FunctionModel(self.script, stream_function=stream_function)
        model: Model = script_model if provider is None else _ProviderWire(provider, script_model, calls)
        agent = Agent(model, **self.agent_kwargs)
        self.setup(agent)
        return agent, calls


def _text_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    turn = sum(isinstance(message, ModelResponse) for message in messages)
    return ModelResponse(parts=[TextPart(f'Answer number {turn}, streamed in deltas.')])


def _tool_call_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if _results_since_question(messages):
        return ModelResponse(parts=[TextPart('It is sunny in Paris.')])
    return ModelResponse(
        parts=[
            TextPart('Let me check.'),
            ToolCallPart('get_weather', {'city': 'Paris'}, tool_call_id=f'call-{len(messages)}'),
        ]
    )


def _parallel_tool_calls_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if _results_since_question(messages):
        return ModelResponse(parts=[TextPart('Sunny in both.')])
    return ModelResponse(
        parts=[
            TextPart('Checking both cities.'),
            ToolCallPart('get_weather', {'city': 'Paris'}, tool_call_id=f'paris-{len(messages)}'),
            ToolCallPart('get_weather', {'city': 'London'}, tool_call_id=f'london-{len(messages)}'),
        ]
    )


def _get_weather(agent: Agent[None, str]) -> None:
    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'Sunny in {city}'


def _get_weather_london_first(agent: Agent[None, str]) -> None:
    """Parallel calls that finish in the reverse of the order the model made them."""
    london_done = anyio.Event()

    @agent.tool_plain
    async def get_weather(city: str) -> str:
        nonlocal london_done
        if city == 'London':
            london_done.set()
        else:
            await london_done.wait()
            london_done = anyio.Event()
        return f'Sunny in {city}'


def _thinking_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[
            ThinkingPart('The user wants an answer.', signature=f'sig-{len(messages)}', provider_name='function'),
            TextPart('Here is an answer.'),
        ]
    )


def _reveal_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    results = _results_since_question(messages)
    if results == 0:
        return ModelResponse(parts=[ToolCallPart('unlock', {}, tool_call_id=f'unlock-{len(messages)}')])
    if results == 1:
        return ModelResponse(parts=[ToolCallPart('secret', {}, tool_call_id=f'secret-{len(messages)}')])
    return ModelResponse(parts=[TextPart('The secret is out.')])


def _reveal_tools(agent: Agent[None, str]) -> None:
    @agent.tool_plain
    def unlock() -> ToolReturn:
        return ToolReturn('unlocked', tools=['secret'])

    @agent.tool_plain(defer_loading=True)
    def secret() -> str:
        return 'the secret'


def _enqueue_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if _results_since_question(messages):
        return ModelResponse(parts=[TextPart('Noted the background result.')])
    return ModelResponse(parts=[ToolCallPart('start_job', {}, tool_call_id=f'job-{len(messages)}')])


def _enqueue_tools(agent: Agent[None, str]) -> None:
    @agent.tool
    def start_job(ctx: RunContext[None]) -> str:
        ctx.enqueue('Background result: 42')
        ctx.enqueue(ModelResponse(parts=[TextPart('Synthetic reply')]), 'Synthetic follow-up')
        return 'started'


def _enqueue_system_prompt(agent: Agent[None, str]) -> None:
    @agent.tool
    def start_job(ctx: RunContext[None]) -> str:
        ctx.enqueue(SystemPromptPart('The job runs in the background.'))
        return 'started'


def _retry_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    results = _results_since_question(messages)
    if results < 2:
        return ModelResponse(parts=[ToolCallPart('flaky', {'attempt': results}, tool_call_id=f'flaky-{len(messages)}')])
    return ModelResponse(parts=[TextPart('Worked on the second attempt.')])


def _retry_tools(agent: Agent[None, str]) -> None:
    @agent.tool_plain(retries=2)
    def flaky(attempt: int) -> str:
        if attempt == 0:
            raise ModelRetry('Try again with attempt=1.')
        return 'ok'


def _responses_since_question(messages: list[ModelMessage]) -> int:
    """How many answers the model has given since the user last asked a question."""
    count = 0
    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            count += 1
        elif any(
            isinstance(part, UserPromptPart) and isinstance(part.content, str) and part.content.startswith('Question')
            for part in message.parts
        ):
            break
    return count


def _draft_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('draft' if _responses_since_question(messages) == 0 else 'final answer')])


def _reject_drafts(agent: Agent[None, str]) -> None:
    @agent.output_validator
    def no_drafts(output: str) -> str:
        if output == 'draft':
            raise ModelRetry('That was a draft; answer for real.')
        return output


def _silent_first_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Answer nothing at first, so the framework asks again in its own voice (`no_output` feedback)."""
    if _responses_since_question(messages) == 0:
        return ModelResponse(parts=[])
    return ModelResponse(parts=[TextPart('Sorry, here is the answer.')])


def _questions_asked(ctx: RunContext[None]) -> int:
    return sum(
        isinstance(part, UserPromptPart) and isinstance(part.content, str) and part.content.startswith('Question')
        for message in ctx.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


_progress = Capability[None](id='progress')


@_progress.instructions(name='state', on_change='append')
def _progress_state(ctx: RunContext[None]) -> str:
    """Operator state that changes every turn, appended to history rather than rewriting the prefix."""
    return f'The user has asked {_questions_asked(ctx)} question(s) so far.'


class _RequestScopedReminder(AbstractCapability[None]):
    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        ctx.enqueue(SystemPromptPart('Reminder: keep it short.', scope='request'))
        return request_context


def _image_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if _results_since_question(messages):
        return ModelResponse(parts=[TextPart('That is a tiny image.')])
    return ModelResponse(parts=[ToolCallPart('take_picture', {}, tool_call_id=f'picture-{len(messages)}')])


_PNG = BinaryContent(
    data=bytes.fromhex(
        '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489'
        '0000000d49444154789c6360f8cf00000301010018dd8db00000000049454e44ae426082'
    ),
    media_type='image/png',
)


def _image_tools(agent: Agent[None, str]) -> None:
    @agent.tool_plain
    def take_picture() -> BinaryContent:
        return _PNG


_AGUI_ACTIVITY = (
    '`@ag-ui/client` leaves `activity` messages out of every request it sends (`prepareRunAgentInput`), '
    'so records the adapter carries as activities never come back; documented in #10569, carrier design pending'
)
_RETRY = (
    "the AI SDK stream has no carrier for a tool's outcome, so a streamed retry reloads as "
    '`outcome="failed"` instead of `"retried"`; #8185'
)
_AGUI_SYSTEM_ORDER = (
    "AG-UI `dump_messages()` writes a request's system prompts before its tool results, so an enqueued "
    'system prompt reloads ahead of the tool result it followed; no issue yet'
)
_APPEND_INSTRUCTIONS = (
    'append-mode instruction records (`InstructionBaselinePart`, `InstructionDeltaPart`) are system voice, '
    "which the adapter strips from client history in `manage_system_prompt='server'`, so every turn re-renders "
    'the baseline from current state and rewrites the prefix; no issue yet'
)
_FEEDBACK_NOT_STREAMED = (
    "retry feedback (`RetryFeedbackPart`) isn't streamed, so the client keeps the rejected and the accepted "
    'answer as one response; no issue yet'
)
_NO_OUTPUT_DEMOTED = (
    "`no_output` feedback is the framework's system voice, which a client can't claim, so it loads back as "
    'user-voice feedback and renders differently; no issue yet'
)
_AGUI_EMPTY_RESPONSE = (
    'AG-UI `dump_messages()` drops an empty response, so the feedback after it loads into the question '
    'before it; no issue yet'
)
_SERVER_SYSTEM_PROMPT = (
    'a system prompt enqueued in an earlier run is stripped from client history with the rest in '
    "`manage_system_prompt='server'`, and nothing restores it; #10389 kept only the ones of the current run"
)


def _gaps(reason: str, *keys: GapKey) -> dict[GapKey, str]:
    return dict.fromkeys(keys, reason)


_ALL_WIRE: tuple[tuple[Property, Path, Protocol], ...] = tuple(
    ('wire', path, protocol) for path in PATHS for protocol in PROTOCOLS
)

SCENARIOS = [
    Scenario(
        id='system_prompt_and_text_deltas',
        script=_text_script,
        agent_kwargs={'system_prompt': 'You are terse.', 'instructions': 'Answer in English.'},
    ),
    Scenario(
        id='client_owned_system_prompt',
        script=_enqueue_script,
        setup=_enqueue_system_prompt,
        adapter_kwargs={'manage_system_prompt': 'client'},
        gaps={
            **_gaps(_AGUI_ACTIVITY, ('load', 'stream', 'agui'), ('wire', 'stream', 'agui')),
            **_gaps(_AGUI_SYSTEM_ORDER, ('wire', 'dump', 'agui')),
        },
        # The frontend owns the system prompt, so it's the one sending it.
        seed=[
            ModelRequest(parts=[SystemPromptPart('You are terse.'), UserPromptPart('Question 0')]),
            ModelResponse(parts=[TextPart('Hello.')]),
        ],
    ),
    Scenario(
        id='enqueued_system_prompt',
        script=_enqueue_script,
        setup=_enqueue_system_prompt,
        gaps=_gaps(_SERVER_SYSTEM_PROMPT, *_ALL_WIRE),
    ),
    Scenario(id='tool_call', script=_tool_call_script, setup=_get_weather),
    Scenario(id='parallel_tool_calls', script=_parallel_tool_calls_script, setup=_get_weather_london_first),
    Scenario(id='thinking', script=_thinking_script),
    Scenario(
        id='tool_availability_delta',
        script=_reveal_script,
        setup=_reveal_tools,
        gaps=_gaps(
            _AGUI_ACTIVITY,
            ('load', 'dump', 'agui'),
            ('load', 'stream', 'agui'),
            ('wire', 'dump', 'agui'),
            ('wire', 'stream', 'agui'),
        ),
    ),
    Scenario(
        id='enqueued_messages',
        script=_enqueue_script,
        setup=_enqueue_tools,
        gaps=_gaps(_AGUI_ACTIVITY, ('load', 'stream', 'agui'), ('wire', 'stream', 'agui')),
    ),
    Scenario(
        id='tool_retry',
        script=_retry_script,
        setup=_retry_tools,
        gaps=_gaps(_RETRY, ('load', 'stream', 'vercel')),
    ),
    Scenario(
        id='output_validation_feedback',
        script=_draft_script,
        setup=_reject_drafts,
        gaps=_gaps(
            _FEEDBACK_NOT_STREAMED,
            ('load', 'stream', 'vercel'),
            ('load', 'stream', 'agui'),
            ('wire', 'stream', 'vercel'),
            ('wire', 'stream', 'agui'),
        ),
    ),
    Scenario(
        id='no_output_feedback',
        script=_silent_first_script,
        gaps={
            **_gaps(_AGUI_EMPTY_RESPONSE, ('load', 'dump', 'agui')),
            **_gaps(_NO_OUTPUT_DEMOTED, ('wire', 'dump', 'vercel'), ('wire', 'dump', 'agui')),
            **_gaps(
                _FEEDBACK_NOT_STREAMED,
                ('load', 'stream', 'vercel'),
                ('load', 'stream', 'agui'),
                ('wire', 'stream', 'vercel'),
                ('wire', 'stream', 'agui'),
            ),
        },
    ),
    Scenario(
        id='appended_instructions',
        script=_text_script,
        agent_kwargs={'capabilities': [_progress]},
        gaps=_gaps(_APPEND_INSTRUCTIONS, *_ALL_WIRE),
    ),
    Scenario(
        id='compaction_boundary',
        script=_reveal_script,
        setup=_reveal_tools,
        agent_kwargs={'capabilities': [_progress]},
        gaps={
            **_gaps(_AGUI_ACTIVITY, ('load', 'dump', 'agui'), ('load', 'stream', 'agui')),
            **_gaps(_APPEND_INSTRUCTIONS, *_ALL_WIRE),
        },
        # A reveal before the compaction, and new reveals and instruction changes after it.
        seed=[
            ModelRequest(parts=[UserPromptPart('Question 0')]),
            ModelResponse(parts=[ToolCallPart('unlock', {}, tool_call_id='unlock-0')]),
            ModelRequest(
                parts=[
                    ToolReturnPart('unlock', 'unlocked', tool_call_id='unlock-0'),
                    ToolAvailabilityDeltaPart(tools_added=['secret'], tool_call_id='unlock-0'),
                ]
            ),
            ModelResponse(
                parts=[
                    CompactionPart(content='Summary so far: the secret tool is unlocked.', provider_name='anthropic'),
                    TextPart('Continuing from the summary.'),
                ]
            ),
        ],
    ),
    Scenario(
        id='request_scoped_system_prompt',
        script=_text_script,
        agent_kwargs={'capabilities': [_RequestScopedReminder()]},
    ),
    Scenario(id='multimodal_tool_return', script=_image_script, setup=_image_tools),
    Scenario(
        id='compaction',
        script=_text_script,
        # OpenAI Chat never sends another provider's compaction, so only Anthropic's request shows the loss.
        gaps=_gaps(
            _AGUI_ACTIVITY,
            ('load', 'dump', 'agui'),
            ('load', 'stream', 'agui'),
            ('wire', 'dump', 'agui', 'anthropic'),
            ('wire', 'stream', 'agui', 'anthropic'),
        ),
        seed=[
            ModelRequest(parts=[UserPromptPart('Question 0')]),
            ModelResponse(
                parts=[
                    CompactionPart(content='Summary so far.', provider_name='anthropic'),
                    TextPart('Continuing from the summary.'),
                ]
            ),
        ],
    ),
]


# --- Clients ------------------------------------------------------------------------------------------------------


def _sse_data(encoded: Sequence[str]) -> list[dict[str, Any]]:
    """The JSON payloads of a server-sent event stream."""
    return [
        json.loads(line.removeprefix('data: '))
        for line in ''.join(encoded).splitlines()
        if line.startswith('data: ') and line != 'data: [DONE]'
    ]


@dataclass
class Turn:
    """One request from the client, as the server received it, and what the server did with it."""

    sent: list[ModelMessage]
    """The history the client sent, loaded and sanitized, without the new prompt."""
    expected: list[ModelMessage]
    """What the client should have sent, sanitized the same way: the history it was hydrated with or streamed."""
    history: list[ModelMessage]
    """The server's `all_messages()` once the turn finished."""
    calls: list[ModelCall]


class Client(ABC):
    """A conversation through one protocol's real JavaScript client."""

    protocol: Protocol

    def __init__(self, driver: JSDriver, session: str, adapter_kwargs: dict[str, Any]) -> None:
        self.driver = driver
        self.session = session
        self.adapter_kwargs = adapter_kwargs

    def submit(self, prompt: str, hydrate: list[ModelMessage] | None) -> bytes:
        """Send `prompt` from the client, first hydrating it with `dump_messages(hydrate)` if given."""
        params: dict[str, Any] = {'message': self.user_message(prompt)}
        if hydrate is not None:
            dumped = self.adapter_type.dump_messages(hydrate)
            params['messages'] = to_jsonable_python(dumped, by_alias=True, exclude_none=True)
        body = self.driver.request(f'{self.protocol}.submit', session=self.session, **params)
        return json.dumps(body).encode()

    def receive(self, encoded: list[str]) -> None:
        """Feed the server's stream to the client's pending request."""
        result = self.driver.request(f'{self.protocol}.receive', session=self.session, **self.stream(encoded))
        assert result.get('error') is None, result['error']

    def sanitize(self, adapter: UIAdapter[Any, Any, Any, Any, Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Parts carrying the system voice were stripped', UserWarning)
            return adapter.sanitize_messages(messages)

    async def turn(
        self, agent: Agent[None, str], calls: list[ModelCall], body: bytes, expected: list[ModelMessage]
    ) -> tuple[Turn, list[str]]:
        """Run the agent on the client's request, as the app's endpoint would."""
        adapter = self.adapter(agent, body)
        results: list[AgentRunResult[Any]] = []
        start = len(calls)
        encoded = [chunk async for chunk in adapter.encode_stream(adapter.run_stream(on_complete=results.append))]
        [result] = results
        turn = self.received(adapter, expected)
        turn.history, turn.calls = result.all_messages(), calls[start:]
        return turn, encoded

    def received(self, adapter: UIAdapter[Any, Any, Any, Any, Any], expected: list[ModelMessage]) -> Turn:
        """What the client sent, next to what it should have, both sanitized as the adapter does."""
        sent = self.adapter_type.load_messages(adapter.run_input.messages[:-1])
        return Turn(sent=self.sanitize(adapter, sent), expected=self.sanitize(adapter, expected), history=[], calls=[])

    @property
    @abstractmethod
    def adapter_type(self) -> type[UIAdapter[Any, Any, Any, Any, Any]]: ...

    @abstractmethod
    def user_message(self, prompt: str) -> dict[str, Any]:
        """The message the user sends, in the shape the client's send call takes."""

    @abstractmethod
    def adapter(self, agent: Agent[None, str], body: bytes) -> UIAdapter[Any, Any, Any, Any, Any]:
        """The adapter the app's endpoint builds for the client's request body."""

    @abstractmethod
    def stream(self, encoded: list[str]) -> dict[str, Any]:
        """The server's encoded stream, as the driver's `receive` takes it."""


class VercelClient(Client):
    protocol = 'vercel'

    @property
    def adapter_type(self) -> type[UIAdapter[Any, Any, Any, Any, Any]]:
        return VercelAIAdapter

    def user_message(self, prompt: str) -> dict[str, Any]:
        return {'text': prompt}

    def adapter(self, agent: Agent[None, str], body: bytes) -> UIAdapter[Any, Any, Any, Any, Any]:
        return VercelAIAdapter(agent, VercelAIAdapter.build_run_input(body), sdk_version=6, **self.adapter_kwargs)

    def stream(self, encoded: list[str]) -> dict[str, Any]:
        return {'chunks': _sse_data(encoded)}


class AGUIClient(Client):
    protocol = 'agui'

    @property
    def adapter_type(self) -> type[UIAdapter[Any, Any, Any, Any, Any]]:
        return AGUIAdapter

    def __init__(self, driver: JSDriver, session: str, adapter_kwargs: dict[str, Any]) -> None:
        super().__init__(driver, session, adapter_kwargs)
        self.prompts = 0

    def user_message(self, prompt: str) -> dict[str, Any]:
        self.prompts += 1
        return {'id': f'{self.session}-user-{self.prompts}', 'role': 'user', 'content': prompt}

    def adapter(self, agent: Agent[None, str], body: bytes) -> UIAdapter[Any, Any, Any, Any, Any]:
        return AGUIAdapter(agent, AGUIAdapter.build_run_input(body), **self.adapter_kwargs)

    def stream(self, encoded: list[str]) -> dict[str, Any]:
        return {'events': _sse_data(encoded)}


CLIENTS: dict[Protocol, type[Client]] = {'vercel': VercelClient, 'agui': AGUIClient}


# --- Running conversations ----------------------------------------------------------------------------------------


async def run_server(scenario: Scenario, provider: WireProvider) -> list[list[ModelCall]]:
    """The provider requests of each turn when the server keeps its own history, with no UI in between."""
    agent, calls = scenario.agent(provider)
    turns: list[list[ModelCall]] = []
    history = scenario.seed
    for prompt in scenario.prompts:
        start = len(calls)
        result: AgentRunResult[str] | None = None
        async with agent.run_stream_events(prompt, message_history=history) as events:
            async for event in events:
                if isinstance(event, AgentRunResultEvent):
                    result = event.result
        assert result is not None
        turns.append(calls[start:])
        history = result.all_messages()
    return turns


async def run_client(
    scenario: Scenario, client: Client, path: Path, provider: WireProvider | None = None
) -> list[Turn]:
    """The conversation through a client, plus one more prompt to see what the client sends back last."""
    agent, calls = scenario.agent(provider)
    turns: list[Turn] = []
    history = scenario.seed
    for index, prompt in enumerate(scenario.prompts):
        # A streaming client starts from the seed too, as when an app loads a stored conversation.
        hydrate = history if path == 'dump' or index == 0 else None
        turn, encoded = await client.turn(agent, calls, client.submit(prompt, hydrate), history)
        turns.append(turn)
        history = turn.history
        if path == 'stream':
            client.receive(encoded)
    body = client.submit(f'Question {len(scenario.prompts) + 1}', history if path == 'dump' else None)
    turns.append(client.received(client.adapter(agent, body), history))
    return turns


def _params(prop: Property) -> list[Any]:
    params: list[Any] = []
    providers: tuple[WireProvider | None, ...] = WIRE_PROVIDERS if prop == 'wire' else (None,)
    for scenario in SCENARIOS:
        for path in PATHS:
            for protocol in PROTOCOLS:
                for provider in providers:
                    gap = scenario.gaps.get((prop, path, protocol))
                    if provider is not None:
                        gap = scenario.gaps.get((prop, path, protocol, provider), gap)
                    values = (scenario, path, protocol) if provider is None else (scenario, path, protocol, provider)
                    params.append(
                        pytest.param(
                            *values,
                            id='-'.join([scenario.id, path, protocol, *([provider] if provider else [])]),
                            marks=[pytest.mark.xfail(reason=gap, strict=True)] if gap else [],
                        )
                    )
    return params


# --- Properties ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(('scenario', 'path', 'protocol'), _params('load'))
async def test_load(scenario: Scenario, path: Path, protocol: Protocol, js_driver: JSDriver) -> None:
    """`load_messages()` of what the client sends back projects identically to the history it should hold.

    That's the history it was hydrated with on the dump path, and the server's `all_messages()` on the stream path.
    """
    client = CLIENTS[protocol](js_driver, f'{scenario.id}-{path}', scenario.adapter_kwargs)
    for turn in await run_client(scenario, client, path):
        assert project(turn.sent) == project(turn.expected)


@pytest.mark.parametrize(('scenario', 'path', 'protocol', 'provider'), _params('wire'))
async def test_wire_request(
    scenario: Scenario,
    path: Path,
    protocol: Protocol,
    provider: WireProvider,
    js_driver: JSDriver,
    allow_model_requests: None,
) -> None:
    """Every provider request through the client equals the one a server keeping its own history makes."""
    client = CLIENTS[protocol](js_driver, f'{scenario.id}-{path}-{provider}-wire', scenario.adapter_kwargs)
    server_turns = await run_server(scenario, provider)
    # The last turn only records what the client would send next.
    client_turns = (await run_client(scenario, client, path, provider))[:-1]
    for server_calls, client_turn in zip(server_turns, client_turns, strict=True):
        assert server_calls
        assert [call.body for call in client_turn.calls] == [call.body for call in server_calls]
