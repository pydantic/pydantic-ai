"""Tools revealed during a realtime session: by loading a deferred capability, by tool search, and by `ToolReturn.tools`."""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace

import anyio
import pytest

from pydantic_ai import Agent, ToolReturn
from pydantic_ai._deferred_capabilities import LoadCapabilityCallPart, LoadCapabilityReturnPart
from pydantic_ai.capabilities import AbstractCapability, Capability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RealtimeTurnCompleteEvent,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.realtime import RealtimeEvent, RealtimeModelProfile, RealtimeModelSettings, RealtimeSession
from pydantic_ai.realtime.codec import (
    OutputTranscript,
    RealtimeCodecEvent,
    ResponseDone,
    ToolCall,
    ToolResult,
    UpdateTools,
    merge_realtime_profile,
)
from pydantic_ai.settings import ToolOrOutput
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from .test_session import (
    FakeRealtimeConnection,
    FakeRealtimeModel,
    _profile,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.anyio

_UPDATES = merge_realtime_profile(_profile(), RealtimeModelProfile(supports_tool_updates=True))
_NO_UPDATES = merge_realtime_profile(_profile(), RealtimeModelProfile(supports_tool_updates=False))


class _ScriptedConnection(FakeRealtimeConnection):
    """A connection whose events the test pushes as the conversation goes, the way a provider answers what it's sent."""

    def __init__(self) -> None:
        super().__init__([])
        self._queue: asyncio.Queue[RealtimeCodecEvent] = asyncio.Queue()

    def push(self, *events: RealtimeCodecEvent) -> None:
        for event in events:
            self._queue.put_nowait(event)

    async def codec_events(self) -> AsyncIterator[RealtimeCodecEvent]:
        while True:
            yield await self._queue.get()


def _call(tool_name: str, args: str = '{}', *, call_id: str) -> list[RealtimeCodecEvent]:
    return [ToolCall(tool_call_id=call_id, tool_name=tool_name, args=args), ResponseDone()]


_ANSWER: list[RealtimeCodecEvent] = [OutputTranscript(text='Done.', is_final=True), ResponseDone()]


async def _converse(
    session: RealtimeSession, connection: _ScriptedConnection, turns: Sequence[list[RealtimeCodecEvent]]
) -> list[RealtimeEvent]:
    """Push each turn once the tool results of the one before are in, and stop after the last turn completes."""
    pending = list(turns)
    connection.push(*pending.pop(0))
    events: list[RealtimeEvent] = []
    with anyio.fail_after(5):
        async for event in session:  # pragma: no branch
            events.append(event)
            if isinstance(event, FunctionToolResultEvent) and pending:
                connection.push(*pending.pop(0))
            elif isinstance(event, RealtimeTurnCompleteEvent) and not pending:
                break
    return events


def _forecast_capability() -> Capability[None]:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    async def get_forecast(city: str) -> str:
        """Look up the weather forecast for a city."""
        return f'Sunny in {city}.'

    return Capability(id='weather', description='Weather forecasts.', toolsets=[toolset], defer_loading=True)


def _sent_groups(connection: FakeRealtimeConnection) -> list[type]:
    return [type(content) for content in connection.sent]


async def test_loading_a_capability_reveals_its_tools() -> None:
    """The capability's tools go out ahead of the load's result, in the same send, and the model can call them."""
    connection = _ScriptedConnection()
    model = FakeRealtimeModel(connection, profile=_UPDATES)
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[_forecast_capability()])

    async with agent.realtime(model).session() as session:
        events = await _converse(
            session,
            connection,
            [
                _call('load_capability', '{"id": "weather"}', call_id='load'),
                _call('get_forecast', '{"city": "Paris"}', call_id='forecast'),
                _ANSWER,
            ],
        )

    assert model.last_tools is not None
    assert [tool.name for tool in model.last_tools] == ['load_capability']
    update, load_result, forecast_result = connection.sent
    assert isinstance(update, UpdateTools)
    assert [tool.name for tool in update.tools] == ['load_capability', 'get_forecast']
    assert isinstance(load_result, ToolResult) and load_result.tool_call_id == 'load'
    assert isinstance(forecast_result, ToolResult) and forecast_result.output == 'Sunny in Paris.'
    results = [event.part for event in events if isinstance(event, FunctionToolResultEvent)]
    assert [part.outcome for part in results] == ['success', 'success']

    messages = session.all_messages()
    load_call = messages[0].parts[0]
    assert isinstance(load_call, LoadCapabilityCallPart)
    assert load_call.capability_id == 'weather'
    load_request = messages[1]
    assert isinstance(load_request, ModelRequest)
    assert isinstance(load_request.parts[0], LoadCapabilityReturnPart)
    assert load_request.parts[1] == ToolAvailabilityDeltaPart(tools_added=['get_forecast'], tool_call_id='load')


async def test_a_loaded_capabilitys_prepare_tools_shapes_the_update() -> None:
    """The update carries the tools as the loaded capability's own `prepare_tools` hook prepares them, as a run's next request does."""
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    async def get_forecast(city: str) -> str:  # pragma: no cover
        """Look up the weather forecast for a city."""
        return f'Sunny in {city}.'

    class Weather(AbstractCapability[None]):
        id = 'weather'
        defer_loading = True

        def get_description(self) -> str:
            return 'Weather forecasts.'

        def get_toolset(self) -> FunctionToolset[None]:
            return toolset

        async def prepare_tools(self, ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [replace(tool_def, description='Prepared description.') for tool_def in tool_defs]

    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[Weather()])
    async with agent.realtime(FakeRealtimeModel(connection, profile=_UPDATES)).session() as session:
        await _converse(session, connection, [_call('load_capability', '{"id": "weather"}', call_id='load'), _ANSWER])

    update = connection.sent[0]
    assert isinstance(update, UpdateTools)
    assert [(tool.name, tool.description) for tool in update.tools][1:] == [('get_forecast', 'Prepared description.')]


async def test_a_standard_run_continues_a_session_that_revealed_tools() -> None:
    """History from a session that loaded a capability is history a standard run treats as loaded too."""
    connection = _ScriptedConnection()
    capability = _forecast_capability()
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[capability])
    async with agent.realtime(FakeRealtimeModel(connection, profile=_UPDATES)).session() as session:
        await _converse(session, connection, [_call('load_capability', '{"id": "weather"}', call_id='load'), _ANSWER])

    seen: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[TextPart('Sunny.')])

    result = await agent.run('And tomorrow?', model=FunctionModel(respond), message_history=session.all_messages())
    assert result.output == 'Sunny.'
    assert seen == ['load_capability', 'get_forecast']


async def test_a_seeded_load_advertises_its_tools_at_connect_on_any_model() -> None:
    """A capability the history already loaded needs no update: its tools go out when the session opens."""
    capability = _forecast_capability()

    def load_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'weather'})])
        return ModelResponse(parts=[TextPart('Loaded.')])

    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[capability])
    history = (await agent.run('Hi.', model=FunctionModel(load_then_answer))).all_messages()

    connection = _ScriptedConnection()
    model = FakeRealtimeModel(connection, profile=_NO_UPDATES)
    async with agent.realtime(model, message_history=history).session() as session:
        events = await _converse(
            session, connection, [_call('get_forecast', '{"city": "Oslo"}', call_id='forecast'), _ANSWER]
        )

    assert model.last_tools is not None
    assert [tool.name for tool in model.last_tools] == ['load_capability', 'get_forecast']
    (result,) = [event.part for event in events if isinstance(event, FunctionToolResultEvent)]
    assert result.content == 'Sunny in Oslo.'
    assert not any(isinstance(content, UpdateTools) for content in connection.sent)


async def test_loading_an_already_loaded_capability_is_refused() -> None:
    """A session knows what it loaded, so a second load is refused as in a standard run, and reveals nothing again."""
    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[_forecast_capability()])
    async with agent.realtime(FakeRealtimeModel(connection, profile=_UPDATES)).session() as session:
        events = await _converse(
            session,
            connection,
            [
                _call('load_capability', '{"id": "weather"}', call_id='first'),
                _call('load_capability', '{"id": "weather"}', call_id='second'),
                _ANSWER,
            ],
        )

    results = [event.part for event in events if isinstance(event, FunctionToolResultEvent)]
    assert [part.outcome for part in results] == ['success', 'retried']
    assert _sent_groups(connection) == [UpdateTools, ToolResult, ToolResult]


async def test_tool_search_reveals_a_deferred_tool() -> None:
    """A `defer_loading=True` tool is found by `search_tools`, and goes out with its result."""
    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None))

    @agent.tool_plain(defer_loading=True)
    async def get_stock_price(ticker: str) -> str:
        """Look up the current stock price for a ticker symbol."""
        return f'{ticker} is at 41.'

    model = FakeRealtimeModel(connection, profile=_UPDATES)
    async with agent.realtime(model).session() as session:
        await _converse(
            session,
            connection,
            [
                _call('search_tools', '{"queries": ["stock price"]}', call_id='search'),
                _call('get_stock_price', '{"ticker": "ACME"}', call_id='price'),
                _ANSWER,
            ],
        )

    assert model.last_tools is not None
    assert [tool.name for tool in model.last_tools] == ['search_tools']
    update, _, price = connection.sent
    assert isinstance(update, UpdateTools)
    assert [tool.name for tool in update.tools] == ['search_tools', 'get_stock_price']
    assert isinstance(price, ToolResult) and price.output == 'ACME is at 41.'
    # A search's result records its own matches, so no `ToolAvailabilityDeltaPart` is added for it.
    search_request = session.all_messages()[1]
    assert isinstance(search_request, ModelRequest)
    assert not any(isinstance(part, ToolAvailabilityDeltaPart) for part in search_request.parts)


async def test_a_tool_return_reveals_a_deferred_tool_once() -> None:
    """`ToolReturn.tools` reveals a tool as in a standard run, and a second reveal of it changes nothing."""
    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None))

    @agent.tool_plain
    async def unlock() -> ToolReturn[str]:
        return ToolReturn(return_value='Unlocked.', tools=['secret', 'secret', 'no_such_tool'])

    @agent.tool_plain(defer_loading=True)
    async def secret() -> str:
        return 'The secret.'

    async with agent.realtime(FakeRealtimeModel(connection, profile=_UPDATES)).session() as session:
        await _converse(
            session,
            connection,
            [
                _call('unlock', call_id='first'),
                _call('unlock', call_id='second'),
                _call('secret', call_id='call'),
                _ANSWER,
            ],
        )

    assert _sent_groups(connection) == [UpdateTools, ToolResult, ToolResult, ToolResult]
    update = connection.sent[0]
    assert isinstance(update, UpdateTools)
    assert [tool.name for tool in update.tools] == ['unlock', 'search_tools', 'secret']
    reveals = [
        part
        for message in session.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart)
    ]
    # As the graph records it: once, with the name that resolves to nothing kept.
    assert reveals == [ToolAvailabilityDeltaPart(tools_added=['secret', 'no_such_tool'], tool_call_id='first')]
    secret_result = connection.sent[-1]
    assert isinstance(secret_result, ToolResult) and secret_result.output == 'The secret.'


async def test_a_revealed_tool_outside_the_tool_choice_stays_hidden() -> None:
    """The update is narrowed by `tool_choice`, as the tools the session opened with were."""
    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[_forecast_capability()])

    @agent.tool_plain
    async def other() -> str:  # pragma: no cover
        return 'other'

    settings = RealtimeModelSettings(tool_choice=ToolOrOutput(function_tools=['load_capability', 'other']))
    model = FakeRealtimeModel(connection, profile=_UPDATES, settings=settings)
    async with agent.realtime(model).session() as session:
        await _converse(session, connection, [_call('load_capability', '{"id": "weather"}', call_id='load'), _ANSWER])

    update = connection.sent[0]
    assert isinstance(update, UpdateTools)
    assert [tool.name for tool in update.tools] == ['load_capability', 'other']


async def test_revealing_an_unloaded_capabilitys_tool_is_refused() -> None:
    """A tool another capability owns is revealed by loading that capability, as in a standard run."""
    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[_forecast_capability()])

    @agent.tool_plain
    async def peek() -> ToolReturn[str]:
        return ToolReturn(return_value='Peeked.', tools=['get_forecast'])

    with pytest.raises(UserError, match=r"`ToolReturn\.tools` cannot reveal 'get_forecast'"):
        async with agent.realtime(FakeRealtimeModel(connection, profile=_UPDATES)).session() as session:
            await _converse(session, connection, [_call('peek', call_id='peek'), _ANSWER])


async def test_a_reveal_a_model_cant_take_warns_once() -> None:
    """On a model that keeps the tools it opened with, a reveal the open couldn't foresee is recorded, and warned about."""
    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None))
    revealable = {'ready': False}

    @agent.toolset(per_run_step=True)
    async def late_tools(ctx: RunContext[None]) -> FunctionToolset[None] | None:
        if not revealable['ready']:
            return None
        toolset = FunctionToolset[None]()

        @toolset.tool_plain(defer_loading=True)
        async def late() -> str:  # pragma: no cover
            return 'late'

        return toolset

    @agent.tool_plain
    async def reveal() -> ToolReturn[str]:
        return ToolReturn(return_value='Revealed.', tools=['late'])

    async with agent.realtime(FakeRealtimeModel(connection, profile=_NO_UPDATES)).session() as session:
        revealable['ready'] = True
        with pytest.warns(
            UserWarning, match=r"A tool call revealed 'late'.*`supports_tool_updates` is `False`"
        ) as record:
            await _converse(
                session,
                connection,
                [
                    # Ahead of the step the tool joins in: a name that resolves to nothing yet changes nothing.
                    _call('reveal', call_id='first'),
                    _call('reveal', call_id='second'),
                    _call('reveal', call_id='third'),
                    _ANSWER,
                ],
            )

    assert len([warning for warning in record if 'revealed' in str(warning.message)]) == 1
    assert _sent_groups(connection) == [ToolResult, ToolResult, ToolResult]
    first_request = session.all_messages()[1]
    assert isinstance(first_request, ModelRequest)
    assert first_request.parts[1] == ToolAvailabilityDeltaPart(tools_added=['late'], tool_call_id='first')


async def test_an_instruction_only_capability_needs_no_update() -> None:
    """Loading a capability with nothing to reveal sends no update, on any model."""

    class Pirate(AbstractCapability[None]):
        id = 'pirate'
        defer_loading = True

        def get_instructions(self) -> str:
            return 'Speak like a pirate.'

    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[Pirate()])
    async with agent.realtime(FakeRealtimeModel(connection, profile=_NO_UPDATES)).session() as session:
        await _converse(session, connection, [_call('load_capability', '{"id": "pirate"}', call_id='load'), _ANSWER])

    assert _sent_groups(connection) == [ToolResult]
    request = session.all_messages()[1]
    assert isinstance(request, ModelRequest)
    assert [type(part) for part in request.parts] == [LoadCapabilityReturnPart]


async def test_tools_still_unrevealed_on_a_model_without_updates_are_refused_at_open() -> None:
    """Of the deferred tools, only those history didn't reveal make a session on a model without updates refuse to open."""
    agent: Agent[None, str] = Agent(deps_type=type(None))

    @agent.tool_plain(defer_loading=True)
    async def revealed() -> str:  # pragma: no cover
        return 'revealed'

    @agent.tool_plain(defer_loading=True)
    async def unrevealed() -> str:  # pragma: no cover
        return 'unrevealed'

    history: list[ModelMessage] = [ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['revealed'])])]
    model = FakeRealtimeModel(FakeRealtimeConnection([]), profile=_NO_UPDATES)
    with pytest.raises(UserError, match=r"`defer_loading=True` are not supported; remove it from: 'unrevealed'\."):
        async with agent.realtime(model, message_history=history).session():
            pass  # pragma: no cover
