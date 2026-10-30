"""A tool discovered before a compaction stays discovered after it (#11493).

Core counts a deferred tool as discovered while the record of its discovery is in the
[`post_compaction_window`][pydantic_ai.messages.post_compaction_window]: a local `search_tools` exchange, a
provider's native search result, or a reveal from a tool that returned `ToolReturn(tools=...)`. A compaction
that removes that record would make the tool undiscovered again, so the shipped strategies record the
discoveries again after it as a `ToolAvailabilityDeltaPart`, not as a search the model never made.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from functools import partial

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, Tool, ToolReturn
from pydantic_ai.capabilities import AbstractCapability, Capability, ToolSearch, Toolset
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchReturnContent,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.compaction import (
    RelevanceCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    compact_now,
)
from pydantic_ai_harness.compaction._capability_loads import keep_tool_discoveries
from tests.function_model_prefix import COMPACTION_REWRITES_HISTORY

SUMMARY = 'Athens was sunny.'


def get_forecast(city: str) -> str:
    """Get the weather forecast for a city."""
    return f'Sunny in {city}'


def find_weather_tools() -> ToolReturn:
    """Find the tools that answer weather questions."""
    return ToolReturn('Found the forecast tool.', tools=['get_forecast'])


def _capabilities() -> list[AbstractCapability[None]]:
    return [
        Toolset(FunctionToolset([Tool(get_forecast, defer_loading=True), Tool(find_weather_tools)])),
        ToolSearch(strategy='keywords'),
    ]


def _discovering_turn(discovery: ToolCallPart) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
    def turn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step == 0:
            return ModelResponse(parts=[discovery])
        if step == 1:
            return ModelResponse(parts=[ToolCallPart('get_forecast', {'city': 'Athens'})])
        return ModelResponse(parts=[TextPart('Sunny in Athens.')])

    return turn


async def _discovered_in_a_run(discovery: ToolCallPart) -> list[ModelMessage]:
    result = await Agent(
        FunctionModel(_discovering_turn(discovery)), deps_type=type(None), capabilities=_capabilities()
    ).run('Weather in Athens?')
    return result.all_messages()


async def _searched() -> list[ModelMessage]:
    return await _discovered_in_a_run(ToolCallPart('search_tools', {'queries': ['forecast']}))


async def _returned() -> list[ModelMessage]:
    return await _discovered_in_a_run(ToolCallPart('find_weather_tools', {}))


async def _searched_natively() -> list[ModelMessage]:
    # `FunctionModel` has no native tool search, so the provider's turn is written out as Anthropic returns it.
    return [
        ModelRequest(parts=[UserPromptPart('Weather in Athens?')]),
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args={'queries': ['forecast']}, tool_call_id='search', provider_name='anthropic'
                ),
                NativeToolSearchReturnPart(
                    content=ToolSearchReturnContent(discovered_tools=[{'name': 'get_forecast'}]),
                    tool_call_id='search',
                    provider_name='anthropic',
                ),
                ToolCallPart('get_forecast', {'city': 'Athens'}, tool_call_id='forecast'),
            ],
            provider_name='anthropic',
        ),
        ModelRequest(parts=[ToolReturnPart('get_forecast', 'Sunny in Athens', tool_call_id='forecast')]),
        ModelResponse(parts=[TextPart('Sunny in Athens.')]),
        ModelRequest(parts=[UserPromptPart('Thanks!')]),
        ModelResponse(parts=[TextPart('Any time.')]),
    ]


Discover = Callable[[], Awaitable[list[ModelMessage]]]

DISCOVERIES = [
    pytest.param(_searched, id='search_tools'),
    pytest.param(_returned, id='ToolReturn.tools'),
    pytest.param(_searched_natively, id='native'),
]


def _summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(SUMMARY)])


def _judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Judges every step unneeded, so each one that can be is dropped or summarized."""
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'needed_verbatim': 0, 'still_matters': 0})])


STRATEGIES = [
    pytest.param(
        partial(SummarizingCompaction[None], max_messages=6, keep_messages=1, model=FunctionModel(_summarize)),
        id='summarizing',
    ),
    pytest.param(partial(SlidingWindowCompaction[None], max_messages=6, keep_messages=1), id='sliding'),
    pytest.param(
        partial(RelevanceCompaction[None], FunctionModel(_judge), max_tokens=1, keep_tokens=1),
        id='relevance-dropping',
        marks=pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY),
    ),
    pytest.param(
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


@dataclass
class _LaterTurn:
    """Calls `get_forecast` straight away, recording the tools each request offered."""

    city: str = 'Oslo'
    tools: list[list[str]] = field(default_factory=list[list[str]])

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.tools.append(sorted(tool.name for tool in info.function_tools))
        returns = [
            part for part in messages[-1].parts if isinstance(part, ToolReturnPart) and part.tool_name == 'get_forecast'
        ]
        if returns:
            return ModelResponse(parts=[TextPart(str(returns[0].content))])
        return ModelResponse(parts=[ToolCallPart('get_forecast', {'city': self.city})])


def _reveals(messages: Sequence[ModelMessage]) -> list[ToolAvailabilityDeltaPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolAvailabilityDeltaPart)]


def _shape(messages: Sequence[ModelMessage]) -> list[str]:
    """Each message as one line naming its parts, so a snapshot pins where the record goes and what it holds."""

    def describe(part: object) -> str:
        match part:
            case ToolAvailabilityDeltaPart():
                return f'reveal {part.tools_added}'
            case ToolCallPart():
                return f'{part.tool_name}({part.args})'
            case ToolReturnPart():
                return f'{part.tool_name} return'
            case UserPromptPart():
                return f'user {part.content!r}'
            case TextPart():
                return f'text {part.content!r}'
            case CompactionPart():
                return 'compaction'
            case _:  # pragma: no cover
                return type(part).__name__

    return [f'{message.kind}: {", ".join(describe(part) for part in message.parts)}' for message in messages]


class TestToolDiscoveriesAcrossCompaction:
    @pytest.mark.parametrize('make', STRATEGIES)
    @pytest.mark.parametrize('discover', DISCOVERIES)
    async def test_a_discovered_tool_stays_discovered(
        self,
        discover: Discover,
        make: Callable[[], AbstractCapability[None]],
    ) -> None:
        history = await discover()
        later_turn = _LaterTurn()
        result = await Agent(
            FunctionModel(later_turn), deps_type=type(None), capabilities=[*_capabilities(), make()]
        ).run('And in Oslo?', message_history=history)

        # The first request after the compaction still offers the discovered tool, and the model's call to it runs.
        assert 'get_forecast' in later_turn.tools[0]
        assert result.output == 'Sunny in Oslo'
        # The discovery is recorded again as a reveal of its own, not as a search the model never made.
        recorded = [part for part in _reveals(result.all_messages()) if part.tool_call_id is None]
        assert recorded == [ToolAvailabilityDeltaPart(tools_added=['get_forecast'])]

    async def test_the_record_goes_right_after_the_summary(self) -> None:
        result = await Agent(
            FunctionModel(_LaterTurn()),
            deps_type=type(None),
            capabilities=[
                *_capabilities(),
                SummarizingCompaction[None](max_messages=6, keep_messages=1, model=FunctionModel(_summarize)),
            ],
        ).run('And in Oslo?', message_history=await _searched())

        assert _shape(result.all_messages()) == snapshot(
            [
                "request: user 'Weather in Athens?'",
                'request: compaction',
                "request: reveal ['get_forecast']",
                "request: user 'And in Oslo?'",
                "response: get_forecast({'city': 'Oslo'})",
                'request: get_forecast return',
                "response: text 'Sunny in Oslo'",
            ]
        )

    async def test_a_recorded_discovery_is_recorded_again_by_the_next_compaction(self) -> None:
        compaction = SummarizingCompaction[None](max_messages=6, keep_messages=1, model=FunctionModel(_summarize))
        history = await _searched()
        for city in ['Oslo', 'Paris']:
            later_turn = _LaterTurn(city)
            result = await Agent(
                FunctionModel(later_turn), deps_type=type(None), capabilities=[*_capabilities(), compaction]
            ).run(f'And in {city}?', message_history=history)
            history = result.all_messages()
            assert 'get_forecast' in later_turn.tools[0]
            assert result.output == f'Sunny in {city}'
            # Each compaction summarized the previous record away and recorded the discovery once more.
            assert _reveals(history) == [ToolAvailabilityDeltaPart(tools_added=['get_forecast'])]

    async def test_a_kept_discovery_is_not_recorded_again(self) -> None:
        history = await _searched()
        # Keeping everything after the first prompt keeps the search, so nothing is lost.
        compacted = await compact_now(
            SlidingWindowCompaction[None](max_messages=1, keep_messages=len(history)),
            [*history, ModelRequest(parts=[UserPromptPart('And in Oslo?')])],
            model=FunctionModel(_summarize),
        )
        assert _reveals(compacted) == []

    async def test_a_tool_a_capability_load_revealed_is_left_to_the_load(self) -> None:
        def weather() -> Capability[None]:
            return Capability(id='weather', description='Weather.', tools=[Tool(get_forecast)], defer_loading=True)

        def load_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'weather'})])
            return ModelResponse(parts=[TextPart('Loaded.')])

        first = await Agent(FunctionModel(load_then_answer), deps_type=type(None), capabilities=[weather()]).run(
            'Weather in Athens?'
        )
        next_prompt = ModelRequest(parts=[UserPromptPart('And in Oslo?')])
        compacted = await compact_now(
            SummarizingCompaction[None](max_messages=1, keep_messages=1, preserve_loaded_capabilities=False),
            [*first.all_messages(), next_prompt],
            model=FunctionModel(_summarize),
        )

        # The opt-out unloads the capability, so its tool isn't revealed again as if it had been discovered.
        assert _reveals(compacted) == []

    @pytest.mark.parametrize('discover', DISCOVERIES)
    async def test_recording_again_on_its_own_result_changes_nothing(self, discover: Discover) -> None:
        original = [*await discover(), ModelRequest(parts=[UserPromptPart('And in Oslo?')])]
        compacted = await compact_now(
            SummarizingCompaction[None](max_messages=1, keep_messages=1), original, model=FunctionModel(_summarize)
        )
        assert ToolAvailabilityDeltaPart(tools_added=['get_forecast']) in _reveals(compacted)

        assert keep_tool_discoveries(original, compacted, kept_from=len(compacted) - 1) == compacted
