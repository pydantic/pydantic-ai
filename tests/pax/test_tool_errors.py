"""An exception a plugin's tool raises reaches the stock agent's model as a failed tool result.

Driven through `Session`, which binds plugins the way the shell does: into a rebuilt stock agent, or
per run on a supplied one.
"""

from collections.abc import AsyncIterator

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_pax import Session
from pydantic_ai_pax.app import create_stock_agent


def _returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


def _calls_once() -> FunctionModel:
    """Calls `crash` once, then answers with what it got back."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if returned := _returns(messages):
            yield f'The tool failed: {returned[0].content}'
        else:
            yield {0: DeltaToolCall(name='crash', json_args='{}', tool_call_id='crash')}

    return FunctionModel(stream_function=stream)


def _crashing_plugin() -> Capability[None]:
    plugin = Capability[None]()

    @plugin.tool_plain
    def crash() -> str:
        raise RuntimeError('connection to /var/run/secret.sock lost')

    return plugin


async def test_a_crashing_plugin_tool_is_reported_and_the_turn_answers() -> None:
    session = Session(create_stock_agent(_calls_once()), deps=None, plugins=[_crashing_plugin()])

    result = await session.prompt('go')

    assert result.output == 'The tool failed: `crash` raised RuntimeError'
    [failed] = _returns(session.messages)
    assert failed.outcome == 'failed'
    assert 'secret.sock' not in str(failed.content), 'the exception message stays out of what the model sees'


async def test_a_supplied_agent_still_fails_the_turn() -> None:
    """`--agent` and `chat(agent)` keep the agent's own error handling."""
    session = Session(Agent(_calls_once()), deps=None, plugins=[_crashing_plugin()])

    with pytest.raises(RuntimeError, match='connection to'):
        await session.prompt('go')


async def test_cancelling_a_turn_inside_a_tool_still_cancels_it() -> None:
    started = anyio.Event()
    plugin = Capability[None]()

    @plugin.tool_plain
    async def crash() -> None:
        started.set()
        await anyio.sleep_forever()

    session = Session(create_stock_agent(_calls_once()), deps=None, plugins=[plugin])
    cancelled = False

    async def prompt() -> None:
        nonlocal cancelled
        try:
            await session.prompt('go')
        except anyio.get_cancelled_exc_class():
            cancelled = True
            raise

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(prompt)
        await started.wait()
        tasks.cancel_scope.cancel()

    assert cancelled
    assert not [part for part in _returns(session.messages) if part.outcome == 'failed']
