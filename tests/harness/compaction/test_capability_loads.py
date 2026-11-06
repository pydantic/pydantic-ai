"""A deferred capability loaded before a compaction stays loaded after it, or the model is told it isn't (#11264, #11770)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    CapabilityLoadDeltaPart,
    CompactionPart,
    InstructionBaselinePart,
    InstructionDeltaPart,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    RelevanceCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    compact_now,
)
from pydantic_ai_harness.compaction._capability_loads import keep_capability_loads
from tests.function_model_prefix import COMPACTION_REWRITES_HISTORY

INSTRUCTIONS = 'Call get_forecast with a city name.'
STATED_INSTRUCTIONS = f'<system><context id="capability:weather">\n{INSTRUCTIONS}\n</context></system>'
"""The capability's instructions in the system voice, as `FunctionModel` receives them."""
STYLE_INSTRUCTIONS = 'Answer in one sentence.'
STATED_STYLE_INSTRUCTIONS = f'<system><context id="capability:style">\n{STYLE_INSTRUCTIONS}\n</context></system>'
SUMMARY = 'Athens was sunny.'
"""Short enough that the compacted history, the unloaded-capability note included, is smaller than the original."""
UNLOADED_NOTE = (
    "Compacting the conversation unloaded these capabilities: 'weather'. Their instructions and tools are "
    'no longer available. Call `load_capability` for any you still need before using its tools.'
)


class Weather(AbstractCapability[None]):
    def __init__(self, *, defer_loading: bool = True) -> None:
        self.id = 'weather'
        self.description = 'Weather forecasts for any city.'
        self.defer_loading = defer_loading

    def get_instructions(self) -> str:
        return INSTRUCTIONS

    def get_toolset(self) -> FunctionToolset[None]:
        toolset = FunctionToolset[None]()

        @toolset.tool_plain
        def get_forecast(city: str) -> str:
            return f'Sunny in {city}'

        return toolset


class Style(AbstractCapability[None]):
    """A deferred capability with instructions and no tools."""

    def __init__(self) -> None:
        self.id = 'style'
        self.description = 'How to phrase answers.'
        self.defer_loading = True

    def get_instructions(self) -> str:
        return STYLE_INSTRUCTIONS


def _agent(model: Model, *capabilities: AbstractCapability[None], from_code: bool = False) -> Agent[None, str]:
    return Agent(
        model, deps_type=type(None), capabilities=list(capabilities), tools=[open_weather] if from_code else []
    )


def open_weather(ctx: RunContext[None]) -> str:
    """Loads the weather capability from code, recording a `CapabilityLoadDeltaPart` rather than an exchange."""
    ctx.load_capability('weather')
    return 'Opened.'


def _first_turn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    step = sum(isinstance(message, ModelResponse) for message in messages)
    if step == 0:
        # The model loads the capability itself, or calls a tool that loads it from code.
        loader = 'open_weather' if 'open_weather' in {tool.name for tool in info.function_tools} else 'load_capability'
        return ModelResponse(parts=[ToolCallPart(loader, {'id': 'weather'} if loader == 'load_capability' else {})])
    if step == 1:
        return ModelResponse(parts=[ToolCallPart('get_forecast', {'city': 'Athens'})])
    return ModelResponse(parts=[TextPart('Sunny in Athens.')])


@dataclass
class _LaterTurn:
    """Calls `get_forecast` for *city*, recording what each request offered and said."""

    city: str = 'Oslo'
    tools: list[list[str]] = field(default_factory=list[list[str]])
    requests: list[list[ModelMessage]] = field(default_factory=list[list[ModelMessage]])

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.tools.append(sorted(tool.name for tool in info.function_tools))
        self.requests.append(messages)
        returns = [
            part for part in messages[-1].parts if isinstance(part, ToolReturnPart) and part.tool_name == 'get_forecast'
        ]
        if returns:
            return ModelResponse(parts=[TextPart(str(returns[0].content))])
        return ModelResponse(parts=[ToolCallPart('get_forecast', {'city': self.city})])


@dataclass
class _Summarizer:
    """Writes `SUMMARY`, recording each transcript it was asked to summarize."""

    prompts: list[str] = field(default_factory=list[str])

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.prompts.append('\n'.join(_user_texts(messages)))
        return ModelResponse(parts=[TextPart(SUMMARY)])


def _summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(SUMMARY)])


def _judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Judges every step unneeded, so each one that can be is dropped or summarized."""
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'needed_verbatim': 0, 'still_matters': 0})])


def _user_texts(messages: Sequence[ModelMessage]) -> list[str]:
    return [
        item.content if isinstance(item, TextContent) else str(item)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
        for item in ([part.content] if isinstance(part.content, str) else part.content)
    ]


def _load_deltas(messages: Sequence[ModelMessage]) -> list[CapabilityLoadDeltaPart]:
    return [part for message in messages for part in message.parts if isinstance(part, CapabilityLoadDeltaPart)]


def _shape(messages: Sequence[ModelMessage]) -> list[str]:
    """Each message as one line naming its parts, so a snapshot pins where records go and what they hold."""

    def describe(part: object) -> str:
        match part:
            case CapabilityLoadDeltaPart():
                return f'load {part.loaded}'
            case ToolAvailabilityDeltaPart():
                return f'reveal {part.tools_added}'
            case ToolCallPart():
                return f'{part.tool_name}({part.args})'
            case ToolReturnPart():
                return f'{part.tool_name} return'
            case UserPromptPart():
                return f'user {" ".join(_user_texts([ModelRequest(parts=[part])]))!r}'
            case TextPart():
                return f'text {part.content!r}'
            case CompactionPart():
                return 'compaction' + (f' ({part.provider_name})' if part.provider_name else '')
            case InstructionDeltaPart():
                return f'instructions {part.id}'
            case InstructionBaselinePart():
                return 'baseline'
            case _:  # pragma: no cover
                return type(part).__name__

    return [f'{message.kind}: {", ".join(describe(part) for part in message.parts)}' for message in messages]


async def _first_turn_history(*, from_code: bool = False) -> list[ModelMessage]:
    # The first turn doesn't compact, so the load it made is in the history the next turn compacts.
    first = await _agent(FunctionModel(_first_turn), Weather(), from_code=from_code).run('Weather in Athens?')
    return first.all_messages()


async def _run_second_turn(
    compaction: AbstractCapability[None], *, from_code: bool = False
) -> tuple[_LaterTurn, list[ModelMessage]]:
    second_turn = _LaterTurn()
    second = await _agent(FunctionModel(second_turn), Weather(), compaction, from_code=from_code).run(
        'And in Oslo?', message_history=await _first_turn_history(from_code=from_code)
    )
    return second_turn, second.all_messages()


STRATEGIES = [
    pytest.param(
        'summarizing',
        partial(SummarizingCompaction[None], max_messages=6, keep_messages=1, model=FunctionModel(_summarize)),
        id='summarizing',
    ),
    pytest.param('sliding', partial(SlidingWindowCompaction[None], max_messages=6, keep_messages=1), id='sliding'),
    pytest.param(
        'relevance-dropping',
        partial(RelevanceCompaction[None], FunctionModel(_judge), max_tokens=1, keep_tokens=1),
        id='relevance-dropping',
        marks=pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY),
    ),
    pytest.param(
        'relevance-summarizing',
        partial(
            RelevanceCompaction[None],
            FunctionModel(_judge),
            summarizer=SummarizingCompaction(model=FunctionModel(_summarize), max_messages=1),
            max_tokens=1,
            keep_tokens=1,
        ),
        id='relevance-summarizing',
    ),
]


SHAPES = snapshot(
    {
        'summarizing': [
            "request: user 'Weather in Athens?', baseline",
            'request: compaction',
            "request: load ['weather'], reveal ['get_forecast']",
            "request: user 'And in Oslo?', baseline, instructions capability:weather",
            "response: get_forecast({'city': 'Oslo'})",
            'request: get_forecast return',
            "response: text 'Sunny in Oslo'",
        ],
        'sliding': [
            "request: user 'Weather in Athens?', baseline",
            "request: load ['weather'], reveal ['get_forecast']",
            "request: user 'And in Oslo?', instructions capability:weather",
            "response: get_forecast({'city': 'Oslo'})",
            'request: get_forecast return',
            "response: text 'Sunny in Oslo'",
        ],
        'relevance-dropping': [
            "request: user 'Weather in Athens?', baseline",
            "request: load ['weather'], reveal ['get_forecast']",
            "response: text 'Sunny in Athens.'",
            "request: user 'And in Oslo?', instructions capability:weather",
            "response: get_forecast({'city': 'Oslo'})",
            'request: get_forecast return',
            "response: text 'Sunny in Oslo'",
        ],
        'relevance-summarizing': [
            "request: user 'Weather in Athens?', baseline",
            "request: load ['weather'], reveal ['get_forecast']",
            'request: compaction',
            "response: get_forecast({'city': 'Oslo'})",
            'request: get_forecast return, baseline, instructions capability:weather',
            "response: text 'Sunny in Oslo'",
        ],
    }
)
CODE_LOAD_SHAPES = snapshot(
    {
        'summarizing': [
            "request: user 'Weather in Athens?', baseline",
            'request: compaction',
            "request: load ['weather'], reveal ['get_forecast']",
            "request: user 'And in Oslo?', baseline, instructions capability:weather",
            "response: get_forecast({'city': 'Oslo'})",
            'request: get_forecast return',
            "response: text 'Sunny in Oslo'",
        ],
        'sliding': [
            "request: user 'Weather in Athens?', baseline",
            "request: load ['weather'], reveal ['get_forecast']",
            "request: user 'And in Oslo?', instructions capability:weather",
            "response: get_forecast({'city': 'Oslo'})",
            'request: get_forecast return',
            "response: text 'Sunny in Oslo'",
        ],
        'relevance-dropping': [
            "request: user 'Weather in Athens?', baseline",
            "request: load ['weather'], reveal ['get_forecast']",
            "response: text 'Sunny in Athens.'",
            "request: user 'And in Oslo?', instructions capability:weather",
            "response: get_forecast({'city': 'Oslo'})",
            'request: get_forecast return',
            "response: text 'Sunny in Oslo'",
        ],
        'relevance-summarizing': [
            "request: user 'Weather in Athens?', baseline",
            "request: load ['weather'], reveal ['get_forecast']",
            'request: compaction',
            "response: get_forecast({'city': 'Oslo'})",
            'request: get_forecast return, baseline, instructions capability:weather',
            "response: text 'Sunny in Oslo'",
        ],
    }
)
OPT_OUT_SHAPES = snapshot(
    {
        'summarizing': [
            "request: user 'Weather in Athens?', baseline",
            'request: compaction',
            'request: user "Compacting the conversation unloaded these capabilities: \'weather\'. Their instructions and tools are no longer available. Call `load_capability` for any you still need before using its tools."',
            "request: user 'And in Oslo?', baseline",
            "response: get_forecast({'city': 'Oslo'})",
            'request: get_forecast return',
            "response: text \"Tool 'get_forecast' is not available yet: it belongs to capability 'weather'. Call `load_capability` for it first, then call the tool again once you've read the capability's instructions.\"",
        ],
        'sliding': [
            "request: user 'Weather in Athens?', baseline",
            'request: user "Compacting the conversation unloaded these capabilities: \'weather\'. Their instructions and tools are no longer available. Call `load_capability` for any you still need before using its tools."',
            "request: user 'And in Oslo?'",
            "response: get_forecast({'city': 'Oslo'})",
            'request: get_forecast return',
            "response: text \"Tool 'get_forecast' is not available yet: it belongs to capability 'weather'. Call `load_capability` for it first, then call the tool again once you've read the capability's instructions.\"",
        ],
        'relevance-dropping': [
            "request: user 'Weather in Athens?', baseline",
            'request: user "Compacting the conversation unloaded these capabilities: \'weather\'. Their instructions and tools are no longer available. Call `load_capability` for any you still need before using its tools."',
            "response: text 'Sunny in Athens.'",
            "request: user 'And in Oslo?'",
            "response: get_forecast({'city': 'Oslo'})",
            'request: get_forecast return',
            "response: text \"Tool 'get_forecast' is not available yet: it belongs to capability 'weather'. Call `load_capability` for it first, then call the tool again once you've read the capability's instructions.\"",
        ],
        'relevance-summarizing': [
            "request: user 'Weather in Athens?', baseline",
            'request: user "Compacting the conversation unloaded these capabilities: \'weather\'. Their instructions and tools are no longer available. Call `load_capability` for any you still need before using its tools."',
            'request: compaction',
            "response: get_forecast({'city': 'Oslo'})",
            'request: get_forecast return, baseline',
            "response: text \"Tool 'get_forecast' is not available yet: it belongs to capability 'weather'. Call `load_capability` for it first, then call the tool again once you've read the capability's instructions.\"",
        ],
    }
)


class TestCapabilityLoadsAcrossCompaction:
    @pytest.mark.parametrize(('name', 'make'), STRATEGIES)
    async def test_a_loaded_capability_stays_loaded(
        self, name: str, make: Callable[..., AbstractCapability[None]]
    ) -> None:
        second_turn, messages = await _run_second_turn(make())

        # The first request after the compaction offers the capability's tools and states its instructions.
        assert second_turn.tools[0] == ['get_forecast', 'load_capability']
        assert STATED_INSTRUCTIONS in _user_texts(second_turn.requests[0])
        # And the model's call to one of them runs.
        assert messages[-1].parts == [TextPart('Sunny in Oslo')]
        # The load is recorded again as a delta and its reveal, never as a `load_capability` exchange the model didn't
        # make. They sit right after the summary, or ahead of the kept messages when nothing replaced the removed ones.
        assert _shape(messages) == SHAPES[name]
        assert not any(isinstance(part, LoadCapabilityCallPart) for message in messages for part in message.parts)

    @pytest.mark.parametrize(('name', 'make'), STRATEGIES)
    async def test_a_capability_loaded_from_code_stays_loaded(
        self, name: str, make: Callable[..., AbstractCapability[None]]
    ) -> None:
        second_turn, messages = await _run_second_turn(make(), from_code=True)

        assert second_turn.tools[0] == ['get_forecast', 'load_capability', 'open_weather']
        assert STATED_INSTRUCTIONS in _user_texts(second_turn.requests[0])
        assert messages[-1].parts == [TextPart('Sunny in Oslo')]
        # Recorded again the way it was recorded: a delta, and the reveal of its tools.
        assert _shape(messages) == CODE_LOAD_SHAPES[name]

    @pytest.mark.parametrize(('name', 'make'), STRATEGIES)
    async def test_the_opt_out_unloads_and_says_so(
        self, name: str, make: Callable[..., AbstractCapability[None]]
    ) -> None:
        second_turn, messages = await _run_second_turn(make(preserve_loaded_capabilities=False))

        assert second_turn.tools[0] == ['load_capability']
        assert STATED_INSTRUCTIONS not in _user_texts(second_turn.requests[0])
        assert UNLOADED_NOTE in _user_texts(second_turn.requests[0])
        assert _shape(messages) == OPT_OUT_SHAPES[name]

    @pytest.mark.parametrize(
        'compaction',
        [
            pytest.param(
                ClearToolResults[None](max_messages=1, keep_pairs=1, clear_tool_inputs=True),
                id='clear',
                marks=pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY),
            ),
            pytest.param(
                ClampOversizedMessages[None](max_part_chars=10, keep_head_chars=1, keep_tail_chars=1), id='clamp'
            ),
        ],
    )
    async def test_strategies_that_keep_every_message_leave_loads_alone(
        self, compaction: AbstractCapability[None]
    ) -> None:
        second_turn, messages = await _run_second_turn(compaction)

        assert second_turn.tools[0] == ['get_forecast', 'load_capability']
        assert messages[-1].parts == [TextPart('Sunny in Oslo')]
        assert _load_deltas(messages) == []

    async def test_a_recorded_load_is_recorded_again_by_the_next_compaction_and_never_summarized(self) -> None:
        summarizer = _Summarizer()
        compaction = SummarizingCompaction[None](max_messages=6, keep_messages=1, model=FunctionModel(summarizer))

        def load_both(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart('load_capability', {'id': 'weather'}),
                        ToolCallPart('load_capability', {'id': 'style'}),
                    ]
                )
            return ModelResponse(parts=[TextPart('Loaded.')])

        first = await _agent(FunctionModel(load_both), Weather(), Style()).run('Weather in Athens?')
        history = first.all_messages()
        for city in ['Oslo', 'Paris']:
            later_turn = _LaterTurn(city)
            result = await _agent(FunctionModel(later_turn), Weather(), Style(), compaction).run(
                f'And in {city}?', message_history=history
            )
            history = result.all_messages()
            # Each turn's first request compacts, and both capabilities are still loaded, in the order they loaded.
            assert later_turn.tools[0] == ['get_forecast', 'load_capability']
            texts = _user_texts(later_turn.requests[0])
            assert texts.index(STATED_INSTRUCTIONS) < texts.index(STATED_STYLE_INSTRUCTIONS)
            # Each compaction summarized the earlier records away and recorded the loads anew, right after its summary.
            assert [delta.loaded for delta in _load_deltas(history)] == [['weather'], ['style']]
            summary_at = max(
                index
                for index, message in enumerate(history)
                if any(isinstance(part, CompactionPart) for part in message.parts)
            )
            assert _shape(history[summary_at + 1 : summary_at + 2]) == [
                "request: load ['weather'], reveal ['get_forecast'], load ['style']"
            ]

        # Load state is never summarized, the model's own loads included: the records carry it.
        assert len(summarizer.prompts) > 1
        assert not any('load_capability' in prompt for prompt in summarizer.prompts)

    @pytest.mark.parametrize(
        'capabilities',
        [
            pytest.param([Weather(defer_loading=False)], id='now-always-on'),
            pytest.param([], id='no-longer-registered'),
        ],
    )
    async def test_a_load_the_run_has_no_use_for_is_not_recorded(
        self, capabilities: list[AbstractCapability[None]]
    ) -> None:
        compaction = SummarizingCompaction[None](max_messages=6, keep_messages=1, model=FunctionModel(_summarize))
        model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('ok')]))
        result = await _agent(model, *capabilities, compaction).run('Hi', message_history=await _first_turn_history())

        messages = result.all_messages()
        assert not any(isinstance(part, LoadCapabilityCallPart) for message in messages for part in message.parts)
        assert UNLOADED_NOTE not in _user_texts(messages)

    async def test_only_a_loads_own_reveal_is_recorded_with_it(self) -> None:
        # Tools revealed by a search or a tool's `ToolReturn(tools=...)` aren't the load's: they're kept, or not, apart.
        original: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Weather in Athens?')]),
            ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'weather'}, tool_call_id='load')]),
            ModelRequest(
                parts=[
                    LoadCapabilityReturnPart(content={'message': 'loaded'}, tool_call_id='load'),
                    ToolAvailabilityDeltaPart(tools_added=['found'], tool_call_id='search'),
                    ToolAvailabilityDeltaPart(tools_added=['get_forecast'], tool_call_id='load'),
                    ToolAvailabilityDeltaPart(tools_added=['revealed']),
                ]
            ),
            ModelRequest(parts=[UserPromptPart('And in Oslo?')]),
        ]
        compacted = [original[0], original[-1]]

        result = keep_capability_loads(original, compacted, kept_from=1, deferred_ids={'weather'})

        assert _shape(result) == snapshot(
            [
                "request: user 'Weather in Athens?'",
                "request: load ['weather'], reveal ['get_forecast']",
                "request: user 'And in Oslo?'",
            ]
        )

    @pytest.mark.parametrize('load_call_id', [pytest.param(None, id='before-request'), pytest.param('tool', id='tool')])
    async def test_a_reveal_after_a_toolless_load_from_code_is_not_the_loads(self, load_call_id: str | None) -> None:
        # A load from code without tools leaves only its delta, so the reveal right after it is someone else's.
        original: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Weather in Athens?')]),
            ModelResponse(parts=[TextPart('Sunny.')]),
            ModelRequest(
                parts=[
                    CapabilityLoadDeltaPart(loaded=['style'], tool_call_id=load_call_id),
                    ToolAvailabilityDeltaPart(tools_added=['found'], tool_call_id='search'),
                ]
            ),
            ModelRequest(parts=[UserPromptPart('And in Oslo?')]),
        ]
        compacted = [original[0], original[-1]]

        result = keep_capability_loads(original, compacted, kept_from=1, deferred_ids={'style'})

        assert _shape(result) == snapshot(
            [
                "request: user 'Weather in Athens?'",
                "request: load ['style']",
                "request: user 'And in Oslo?'",
            ]
        )

    async def test_compact_now_records_loads_without_a_run(self) -> None:
        history = await _first_turn_history()
        next_prompt = ModelRequest(parts=[UserPromptPart('And in Oslo?')])

        compacted = await compact_now(
            SummarizingCompaction[None](max_messages=1, keep_messages=1),
            [*history, next_prompt],
            model=FunctionModel(_summarize),
        )

        # There's no registry to check outside a run; the next run ignores a load of a capability it doesn't register.
        assert [delta.loaded for delta in _load_deltas(compacted)] == [['weather']]
        assert compacted[-1] is next_prompt

    async def test_a_history_holding_only_recorded_loads_and_a_summary_is_not_summarized_again(self) -> None:
        history = await _first_turn_history()
        compacted = await compact_now(
            SummarizingCompaction[None](max_messages=1, keep_messages=1),
            [*history, ModelRequest(parts=[UserPromptPart('And in Oslo?')])],
            model=FunctionModel(_summarize),
        )

        def must_not_summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise AssertionError('summarized again')  # pragma: no cover

        # Everything ahead of the newest message is the kept first prompt, the summary and the recorded load.
        again = await compact_now(
            SummarizingCompaction[None](max_messages=1, keep_tokens=1),
            compacted,
            model=FunctionModel(must_not_summarize),
        )
        assert again == compacted

    async def test_a_load_after_a_kept_provider_compaction_is_recorded_after_its_tool_results(self) -> None:
        def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            needed = 1 if any('KEEP' in text for text in _user_texts(messages)) else 0
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {'needed_verbatim': needed, 'still_matters': 0})]
            )

        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Weather in Athens?')]),
            # A provider's compaction, in a response that also calls a tool: the judge keeps this step.
            ModelResponse(
                parts=[
                    CompactionPart(content='provider summary', provider_name='anthropic'),
                    ToolCallPart('get_forecast', {'city': 'Athens'}, tool_call_id='forecast'),
                ],
                provider_name='anthropic',
            ),
            ModelRequest(parts=[ToolReturnPart('get_forecast', 'Sunny KEEP', tool_call_id='forecast')]),
            # The load after it is too large for the target, so it's dropped.
            ModelResponse(
                parts=[TextPart('x' * 4_000), LoadCapabilityCallPart(args={'id': 'weather'}, tool_call_id='load')]
            ),
            ModelRequest(parts=[LoadCapabilityReturnPart(content={'message': 'loaded'}, tool_call_id='load')]),
            ModelResponse(parts=[TextPart('Sunny in Athens.')]),
            ModelRequest(parts=[UserPromptPart('And in Oslo?')]),
        ]

        compacted = await compact_now(
            RelevanceCompaction[None](FunctionModel(judge), max_tokens=1_000, target_tokens=500, keep_tokens=1),
            history,
            model=FunctionModel(_summarize),
        )

        # The recorded load goes after the kept step's tool result, so the call and its result stay together.
        assert compacted[:3] == history[:3]
        assert compacted[3].parts == [CapabilityLoadDeltaPart(loaded=['weather'])]
        assert compacted[4:] == history[5:]

    async def test_a_load_after_a_signed_block_opening_the_history_is_recorded_after_it(self) -> None:
        # Anthropic's on-demand compaction leaves a signed block that must stay first in the history.
        def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {'needed_verbatim': 0, 'still_matters': 0})]
            )

        history: list[ModelMessage] = [
            ModelResponse(
                parts=[CompactionPart(content='signed summary', provider_name='anthropic')], provider_name='anthropic'
            ),
            ModelRequest(parts=[UserPromptPart('Weather in Athens?')]),
            ModelResponse(
                parts=[TextPart('x' * 4_000), LoadCapabilityCallPart(args={'id': 'weather'}, tool_call_id='load')]
            ),
            ModelRequest(parts=[LoadCapabilityReturnPart(content={'message': 'loaded'}, tool_call_id='load')]),
            ModelResponse(parts=[TextPart('Sunny in Athens.')]),
            ModelRequest(parts=[UserPromptPart('And in Oslo?')]),
        ]

        compacted = await compact_now(
            RelevanceCompaction[None](FunctionModel(judge), max_tokens=1_000, target_tokens=500, keep_tokens=1),
            history,
            model=FunctionModel(_summarize),
        )

        assert _shape(compacted) == snapshot(
            [
                'response: compaction (anthropic)',
                "request: user 'Weather in Athens?'",
                "request: load ['weather']",
                "response: text 'Sunny in Athens.'",
                "request: user 'And in Oslo?'",
            ]
        )

    @pytest.mark.parametrize('preserve', [True, False])
    async def test_recording_again_on_its_own_result_changes_nothing(self, preserve: bool) -> None:
        # Core's compaction step (#11454) calls this after whichever compaction ran, a strategy's included.
        original = [*await _first_turn_history(), ModelRequest(parts=[UserPromptPart('And in Oslo?')])]
        compacted = await compact_now(
            SummarizingCompaction[None](max_messages=1, keep_messages=1, preserve_loaded_capabilities=preserve),
            original,
            model=FunctionModel(_summarize),
        )
        # The compaction recorded the load, or noted it was unloaded, itself.
        assert any("load ['weather']" in line or 'unloaded these capabilities' in line for line in _shape(compacted))

        again = keep_capability_loads(
            original, compacted, kept_from=len(compacted) - 1, deferred_ids={'weather'}, preserve=preserve
        )
        assert again == compacted
