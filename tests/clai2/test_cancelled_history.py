"""Interrupted requests remain available to the next clarification."""

import signal

import anyio
import pytest
from pydantic_ai import Agent, AgentStreamEvent, PartDeltaEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel

from pydantic_clai2 import Session
from pydantic_clai2.interrupts import Interrupts


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.parametrize('stage', ['before_run', 'before_model', 'stream'])
async def test_sigint_retains_request_and_partial_response(stage: str) -> None:
    class Pause(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            if stage == 'before_run':
                signal.raise_signal(signal.SIGINT)
                await anyio.sleep_forever()

    async def instructions(ctx: RunContext[None]) -> str:
        if stage == 'before_model':
            signal.raise_signal(signal.SIGINT)
            await anyio.sleep_forever()
        return ''

    async def observe(event: AgentStreamEvent) -> None:
        if stage == 'stream' and isinstance(event, PartDeltaEvent):
            signal.raise_signal(signal.SIGINT)
            await anyio.sleep_forever()

    agent = Agent(TestModel(custom_output_text='partial answer'), deps_type=type(None), instructions=instructions)
    prior = [ModelRequest(parts=[UserPromptPart('earlier')]), ModelResponse(parts=[TextPart('hello')])]
    session = Session(agent, deps=None, plugins=[Pause()], message_history=prior, on_stream_event=observe)

    async def turn() -> None:
        await session.prompt('make a personality plugin')

    assert not await Interrupts().run(turn())
    assert session.messages[:2] == prior
    assert any(
        isinstance(part, UserPromptPart) and part.content == 'make a personality plugin'
        for message in session.messages
        for part in message.parts
    )
    if stage == 'stream':
        response = session.messages[-1]
        assert isinstance(response, ModelResponse)
        assert response.state == 'interrupted'
        assert any(isinstance(part, TextPart) and part.content for part in response.parts)

    session.plugins = []
    session.on_stream_event = None
    with agent.override(instructions=''):
        result = await session.prompt('use playful plus pedantic')
    assert [
        part.content for message in result.all_messages() for part in message.parts if isinstance(part, UserPromptPart)
    ] == ['earlier', 'make a personality plugin', 'use playful plus pedantic']


async def test_failed_turn_does_not_replace_history() -> None:
    class Fail(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            raise ValueError('failed')

    session = Session(Agent(TestModel()), deps=None)
    await session.prompt('earlier')
    prior = session.messages
    session.plugins = [Fail()]
    with pytest.raises(ValueError, match='failed'):
        await session.prompt('failure')
    assert session.messages == prior
    session.plugins = []
    await session.prompt('retry')
