"""Speculative results obey the same sandbox boundary as cold tool calls."""

from decimal import Decimal

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from pydantic_ai_harness.code_mode import CodeMode

from .test_speculation import ToolLog, build_agent, padded


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.anyio
@pytest.mark.parametrize('eager', [False, True])
async def test_claimed_result_serializes_mapping_keys(eager: bool) -> None:
    capability = CodeMode[None](speculate=['prices'], eager=eager)
    agent = build_agent(
        log=ToolLog(),
        code=padded("prices_by_amount = await prices()\nprint(prices_by_amount['1.50'])"),
        capability=capability,
    )
    calls = 0

    @agent.tool_plain
    def prices() -> dict[Decimal, str]:
        nonlocal calls
        calls += 1
        return {Decimal('1.50'): 'USD'}

    result = await agent.run('go')
    returns = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]
    assert len(returns) == 1
    assert returns[0].content == {'output': 'USD\n', 'result': 'ok'}
    assert calls == 1
    assert capability.speculation_stats is not None
    assert capability.speculation_stats.adopted == 1
