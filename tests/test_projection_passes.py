"""Guards on the projection passes `Model.prepare_messages` runs, over representative histories.

Each history holds the record kinds the passes translate or drop: instruction changes, capability loads and parts,
tool reveals, settings changes, retries, speech, request-scoped and `<context>` prompts, user text that looks like a
frame, and thinking. Each is rendered for several adapters, so both `prepare_messages` and the adapters' own mapping
are covered. Three properties are checked (#11030):

- **Earlier messages are stable.** Every request a history leads to starts with what the previous request settled:
  every message `prepare_messages` returned for it, the trailing request's request-scoped prompts aside, and the
  provider body up to its last assistant turn.
- **Round trip.** A history dumped and loaded with `ModelMessagesTypeAdapter` renders the same request bytes.
- **No pass drops a kind a later pass reads.** Each pass declares the part kinds it reads, the ones it consumes
  (renders as other parts) and the ones it drops (leaves out with nothing in their place). The order is checked from
  the declarations, and the declarations are checked against what each pass does.

A property that fails for a history here on the base branch is a known finding, pinned by a strict expected failure on
that history, as `tests/test_history_properties.py` does.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import cache

import pydantic_core
import pytest
from _pytest.mark import ParameterSet
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
    repair_messages,
)
from pydantic_ai.models import (
    _PROJECTION_PASSES,  # pyright: ignore[reportPrivateUsage]
    Model,
    ModelRequestParameters,
    _PartKind,  # pyright: ignore[reportPrivateUsage]
    _Projection,  # pyright: ignore[reportPrivateUsage]
    _ProjectionPass,  # pyright: ignore[reportPrivateUsage]
    _request_scoped_tail_texts,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from .conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.groq import GroqProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='model provider packages not installed')

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def _request(*parts: ModelRequestPart, at: int, instructions: str | None = 'Be brief.') -> ModelRequest:
    return ModelRequest(parts=list(parts), timestamp=_at(at), instructions=instructions)


def _response(*parts: ModelResponsePart, at: int) -> ModelResponse:
    return ModelResponse(parts=list(parts), timestamp=_at(at), model_name='test-model', provider_name='test')


@dataclass(kw_only=True)
class RepoRule(InstructionCapabilityPart, namespace='test_projection_passes'):
    """An instruction a capability adds to the conversation."""


@dataclass(kw_only=True)
class RepoFacts(DataCapabilityPart, namespace='test_projection_passes', tag='repo_facts'):
    """Data a capability adds to the conversation."""


# --- Representative histories -------------------------------------------------------------------------------------


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
            RetryFeedbackPart('the answer has to be a number', retry=RetryInfo(cause='model_retry'), timestamp=_at(4)),
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
    HistoryCase('speech', _speech),
    HistoryCase('thinking-after-a-tool-round', _thinking_after_a_tool_round),
    HistoryCase(
        'code-loaded-capability-across-instruction-windows', _code_loaded_capability_across_instruction_windows
    ),
]


# --- Render targets ---------------------------------------------------------------------------------------------


def _prepare(model: Model, history: list[ModelMessage]) -> tuple[ModelRequestParameters, list[ModelMessage]]:
    """What an agent run does to a history before the model maps it, as in `tests/test_history_properties.py`."""
    run_parameters = ModelRequestParameters()
    messages = repair_messages(model.prepare_messages(repair_messages(history), run_parameters))
    _, parameters = model.prepare_request(None, run_parameters)
    return parameters, messages


def _jsonable(value: object) -> JsonValue:
    return pydantic_core.to_jsonable_python(value, bytes_mode='base64')


@cache
def _openai_chat_model() -> OpenAIChatModel:
    return OpenAIChatModel('gpt-5.5', provider=OpenAIProvider(api_key='test-key'))


async def _openai_chat_body(messages: list[ModelMessage], parameters: ModelRequestParameters) -> list[JsonValue]:
    mapped = await _openai_chat_model()._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, model_request_parameters=parameters, model_settings=OpenAIChatModelSettings()
    )
    return [_jsonable(message) for message in mapped]


@cache
def _openrouter_model() -> OpenRouterModel:
    """Sends thinking back in a `reasoning` field (`openai_chat_send_back_thinking_parts='field'`)."""
    return OpenRouterModel('anthropic/claude-haiku-4.5', provider=OpenRouterProvider(api_key='test-key'))


async def _openrouter_body(messages: list[ModelMessage], parameters: ModelRequestParameters) -> list[JsonValue]:
    mapped = await _openrouter_model()._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, model_request_parameters=parameters, model_settings=OpenAIChatModelSettings()
    )
    return [_jsonable(message) for message in mapped]


@cache
def _openai_responses_model() -> OpenAIResponsesModel:
    return OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(api_key='test-key'))


async def _openai_responses_body(messages: list[ModelMessage], parameters: ModelRequestParameters) -> list[JsonValue]:
    _, items = await _openai_responses_model()._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, model_settings=OpenAIResponsesModelSettings(), model_request_parameters=parameters
    )
    return [_jsonable(item) for item in items]


@cache
def _anthropic_model() -> AnthropicModel:
    return AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key='test-key'))


async def _anthropic_body(messages: list[ModelMessage], parameters: ModelRequestParameters) -> list[JsonValue]:
    _, mapped = await _anthropic_model()._map_message(  # pyright: ignore[reportPrivateUsage]
        messages, model_request_parameters=parameters, model_settings=AnthropicModelSettings()
    )
    return [_jsonable(message) for message in mapped]


@cache
def _google_model() -> GoogleModel:
    return GoogleModel('gemini-3.5-flash', provider=GoogleProvider(api_key='test-key'))


async def _google_body(messages: list[ModelMessage], parameters: ModelRequestParameters) -> list[JsonValue]:
    _, contents = await _google_model()._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, model_request_parameters=parameters
    )
    return [_jsonable(content) for content in contents]


@cache
def _groq_model() -> GroqModel:
    """A model without mid-conversation system messages, so the `system_voice` pass wraps them."""
    return GroqModel('openai/gpt-oss-120b', provider=GroqProvider(api_key='test-key'))


async def _groq_body(messages: list[ModelMessage], parameters: ModelRequestParameters) -> list[JsonValue]:
    mapped = await _groq_model()._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, model_request_parameters=parameters
    )
    return [_jsonable(message) for message in mapped]


def _is_user_side(item: JsonValue) -> bool:
    """Whether a body item is the request's side of the conversation, rather than a turn the model produced."""
    assert isinstance(item, dict)
    role = item.get('role')
    if role is not None:
        return role not in ('assistant', 'model')
    # OpenAI Responses items without a role: everything but a tool result is the model's.
    return item.get('type') == 'function_call_output'


@dataclass(frozen=True)
class Target:
    id: str
    model: Callable[[], Model]
    body: Callable[[list[ModelMessage], ModelRequestParameters], Coroutine[object, object, list[JsonValue]]]


TARGETS: list[Target] = [
    Target('openai-chat', _openai_chat_model, _openai_chat_body),
    Target('openrouter', _openrouter_model, _openrouter_body),
    Target('openai-responses', _openai_responses_model, _openai_responses_body),
    Target('anthropic', _anthropic_model, _anthropic_body),
    Target('google', _google_model, _google_body),
    Target('groq', _groq_model, _groq_body),
]


@dataclass(frozen=True)
class Rendered:
    """One request a history leads to: what `prepare_messages` returned, and the provider body's messages."""

    messages: list[JsonValue]
    request_scoped_tail: int
    """How many of the trailing request's last parts are request-scoped prompts, which later requests drop."""
    body: list[JsonValue]


async def _render(target: Target, history: list[ModelMessage]) -> Rendered:
    parameters, messages = _prepare(target.model(), history)
    return Rendered(
        messages=[_message_json(message) for message in messages],
        request_scoped_tail=len(_request_scoped_tail_texts(messages)),
        body=await target.body(messages, parameters),
    )


def _message_json(message: ModelMessage) -> JsonValue:
    """`message` as JSON, without what no adapter sends: timestamps, metadata, and a request's `instructions`, its
    system prompt, which adapters read from the trailing request alone."""
    [dumped] = ModelMessagesTypeAdapter.dump_python([message], mode='json')
    for key in ('instructions', 'timestamp', 'metadata'):
        dumped.pop(key, None)
    for part in dumped['parts']:
        part.pop('timestamp', None)
    return dumped


def _assert_settled_messages_kept(earlier: Rendered, later: Rendered) -> None:
    """`later` renders every message `earlier` rendered the same, and the trailing request's parts as the start of
    its own, so parts a later turn adds to it (such as a note ahead of carried reasoning) only extend it. The trailing
    request's request-scoped prompts are left out: later requests drop them."""
    *settled, trailing = earlier.messages
    assert later.messages[: len(settled)] == settled
    assert isinstance(trailing, dict) and isinstance(trailing['parts'], list)
    parts = trailing['parts'][: len(trailing['parts']) - earlier.request_scoped_tail]
    later_trailing = later.messages[len(settled)]
    assert isinstance(later_trailing, dict) and isinstance(later_trailing['parts'], list)
    assert {**later_trailing, 'parts': later_trailing['parts'][: len(parts)]} == {**trailing, 'parts': parts}


def _conversation(body: list[JsonValue]) -> list[JsonValue]:
    """The body's items after the leading system ones: those carry the request's instructions, which an instruction
    change rewrites by design, and which the cache-prefix tests cover."""
    start = next(
        (
            i
            for i, item in enumerate(body)
            if not (isinstance(item, dict) and item.get('role') in ('system', 'developer'))
        ),
        len(body),
    )
    return body[start:]


def _settled_body(body: list[JsonValue]) -> list[JsonValue]:
    """The conversation items up to and including the last turn the model produced."""
    conversation = _conversation(body)
    last = max((i for i, item in enumerate(conversation) if not _is_user_side(item)), default=-1)
    return conversation[: last + 1]


# --- Known findings ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownFinding:
    history: str
    target: str | None
    """`None` when every target fails, because the pass list itself is wrong."""
    tracked_by: str


KNOWN_STABILITY_FINDINGS = [
    # Strict expected failures: #12304 (the fix for #12261) and #12309 flip them, and each entry goes when its fix
    # reaches this branch.
    # `drop_capability_loads` runs before `instruction_state`, which folds the loads: once the second run opens a new
    # instruction window, the code-loaded capability's statement in the first run's turns stops rendering.
    KnownFinding(
        'code-loaded-capability-across-instruction-windows',
        None,
        'https://github.com/pydantic/pydantic-ai/issues/12261',
    ),
    # `OpenAIChatModel._map_messages` backfills `reasoning: ''` on every tool-call turn once any response thinks,
    # including turns earlier requests sent without it. The adapter does this after `prepare_messages`.
    KnownFinding('thinking-after-a-tool-round', 'openrouter', 'https://github.com/pydantic/pydantic-ai/pull/12309'),
]


def _cases(findings: Sequence[KnownFinding] = ()) -> list[ParameterSet]:
    params: list[ParameterSet] = []
    for history in HISTORIES:
        for target in TARGETS:
            finding = next(
                (
                    finding
                    for finding in findings
                    if finding.history == history.id and finding.target in (None, target.id)
                ),
                None,
            )
            marks = (
                (pytest.mark.xfail(strict=True, reason=f'known finding, {finding.tracked_by}'),)
                if finding is not None
                else ()
            )
            params.append(pytest.param(history, target, id=f'{history.id}-{target.id}', marks=marks))
    return params


# --- (a) Earlier messages are stable ------------------------------------------------------------------------------


@pytest.mark.parametrize(('history_case', 'target'), _cases(KNOWN_STABILITY_FINDINGS))
async def test_earlier_messages_render_the_same_in_every_later_request(history_case: HistoryCase, target: Target):
    """Each request starts with what the previous request settled: the prepared messages (see
    `_assert_settled_messages_kept`) and the body up to its last assistant turn. Rendering a turn depends only on that
    turn and what came before it, never on later ones."""
    history = await history_case.history()
    request_ends = [i + 1 for i, message in enumerate(history) if isinstance(message, ModelRequest)]
    renderings = [await _render(target, history[:end]) for end in request_ends]
    for earlier, later in zip(renderings, renderings[1:]):
        _assert_settled_messages_kept(earlier, later)
        settled_body = _settled_body(earlier.body)
        assert _conversation(later.body)[: len(settled_body)] == settled_body


# --- (b) Round trip ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(('history_case', 'target'), _cases())
async def test_round_trip_renders_the_same_request(history_case: HistoryCase, target: Target):
    """A history dumped and loaded with `ModelMessagesTypeAdapter` renders the same request bytes."""
    history = await history_case.history()
    loaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
    assert await _render(target, loaded) == await _render(target, history)


# --- (c) No pass drops a kind a later pass reads ------------------------------------------------------------------

REMOVED_KINDS: frozenset[_PartKind] = frozenset(
    kind for projection_pass in _PROJECTION_PASSES for kind in projection_pass.consumes | projection_pass.drops
)
"""The part kinds some pass consumes or drops: the ones whose readers the declarations have to name."""

KNOWN_ORDER_FINDINGS = [
    # #12304, the fix for #12261, empties this list: drop the entry when it reaches this branch.
    # https://github.com/pydantic/pydantic-ai/issues/12261: the instruction state folds code loads to keep their blocks
    # rendered across instruction windows, but the records are dropped before it runs.
    ('drop_capability_loads', 'instruction_state', 'CapabilityLoadDeltaPart'),
]


def _reads(projection_pass: _ProjectionPass) -> frozenset[_PartKind]:
    return projection_pass.reads | projection_pass.consumes


def test_no_pass_drops_a_kind_a_later_pass_reads():
    violations = [
        (earlier.name, later.name, kind.__name__)
        for index, earlier in enumerate(_PROJECTION_PASSES)
        for later in _PROJECTION_PASSES[index + 1 :]
        for kind in sorted(earlier.drops & _reads(later), key=lambda kind: kind.__name__)
    ]
    assert violations == KNOWN_ORDER_FINDINGS


def test_pass_names_are_unique():
    names = [projection_pass.name for projection_pass in _PROJECTION_PASSES]
    assert len(names) == len(set(names))


def _part_kinds(messages: list[ModelMessage]) -> set[type]:
    return {type(part) for message in messages for part in message.parts}


def _without(messages: list[ModelMessage], kind: _PartKind) -> list[ModelMessage]:
    """`messages` without their parts of `kind`, keeping a message left with none."""
    return [
        replace(message, parts=[part for part in message.parts if not isinstance(part, kind)]) for message in messages
    ]


def _dump(messages: list[ModelMessage]) -> list[JsonValue]:
    return [_message_json(message) for message in messages]


@pytest.mark.parametrize(('history_case', 'target'), _cases())
async def test_passes_drop_only_what_they_declare(history_case: HistoryCase, target: Target):
    """No pass leaves a part kind out of the history that it doesn't declare it consumes or drops."""
    model = target.model()
    projection = _Projection(repair_messages(await history_case.history()), ModelRequestParameters())
    for projection_pass in _PROJECTION_PASSES:
        projected = projection_pass.run(model, projection)
        removed = _part_kinds(projection.messages) - _part_kinds(projected.messages)
        declared = tuple(projection_pass.consumes | projection_pass.drops)
        assert [kind.__name__ for kind in removed if not issubclass(kind, declared)] == [], projection_pass.name
        projection = projected


@pytest.mark.parametrize(('history_case', 'target'), _cases())
async def test_passes_read_only_what_they_declare(history_case: HistoryCase, target: Target):
    """A pass's output doesn't depend on the parts of a kind some pass consumes or drops, unless it declares reading it.

    Each pass runs on the stored history, which still holds every kind, so a pass that reads a kind an earlier pass
    drops shows up here even though, run in order, it never sees one.
    """
    model = target.model()
    history = repair_messages(await history_case.history())
    present = _part_kinds(history)
    for projection_pass in _PROJECTION_PASSES:
        for kind in sorted(REMOVED_KINDS - _reads(projection_pass), key=lambda kind: kind.__name__):
            if not any(issubclass(present_kind, kind) for present_kind in present):
                continue
            stripped = _without(history, kind)
            full = projection_pass.run(model, _Projection(history, ModelRequestParameters(), authored=history))
            without = projection_pass.run(model, _Projection(stripped, ModelRequestParameters(), authored=stripped))
            assert _dump(_without(full.messages, kind)) == _dump(without.messages), (
                projection_pass.name,
                kind.__name__,
            )
