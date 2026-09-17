"""`/usage`, `/cost`, and the footer cost, derived only from retained messages."""

import io
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Generic, TypeVar

import pytest
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage
from rich.console import Console

from pydantic_clai2 import Session, chat
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.status import Status
from pydantic_clai2.usage_report import cost_line, format_cost, session_usage, usage_command

PromptT = TypeVar('PromptT')


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def priced_history() -> list[ModelMessage]:
    """Two turns as core would retain them: a tool round trip, then a plain answer, then an unpriced one."""
    return [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(
            parts=[ToolCallPart('shell', {})],
            model_name='gpt-4o',
            usage=RequestUsage(input_tokens=1000, cache_read_tokens=100, output_tokens=50, cost=Decimal('0.0123')),
        ),
        ModelRequest(parts=[ToolReturnPart('shell', 'done', tool_call_id='1')]),
        ModelResponse(
            parts=[TextPart('answer')],
            model_name='gpt-4o',
            usage=RequestUsage(input_tokens=2000, cache_write_tokens=200, output_tokens=100, cost=Decimal('0.00001')),
        ),
        ModelRequest(parts=[UserPromptPart('second')]),
        ModelResponse(parts=[TextPart('cheap')], usage=RequestUsage(input_tokens=10, output_tokens=1)),
        ModelRequest(parts=[UserPromptPart('cancelled before any response')]),
    ]


def test_turns_group_tool_round_trips_and_skip_unpriced_responses() -> None:
    usage = session_usage(priced_history())
    assert [turn.requests for turn in usage.turns] == [2, 1]
    assert usage.turns[0].input_tokens == 3000
    assert usage.turns[0].cost == Decimal('0.01231')
    assert usage.turns[1].cost is None
    assert usage.total.requests == 3
    assert usage.total.cost == Decimal('0.01231')
    assert usage.unpriced == ('unknown',)
    assert cost_line(usage) == (
        'Retained history cost $0.0123: 3,161 tokens (3,010 in, 151 out) over 3 requests.'
        ' No price data for unknown; those responses are not counted in costs.'
    )


def test_prompt_cancelled_before_a_response_is_not_a_turn() -> None:
    usage = session_usage([ModelRequest(parts=[UserPromptPart('cancelled')])])
    assert usage.turns == []
    assert cost_line(usage) == 'No usage yet.'


def test_response_before_any_prompt_starts_a_turn() -> None:
    response = ModelResponse(parts=[TextPart('seeded')], model_name='m', usage=RequestUsage(input_tokens=5))
    usage = session_usage([response])
    assert len(usage.turns) == 1
    assert usage.turns[0].requests == 1
    assert usage.unpriced == ('m',)


def test_usage_table_shows_cache_columns_and_sub_cent_costs() -> None:
    output = io.StringIO()
    line = usage_command(priced_history(), console=Console(file=output, width=120))
    table = output.getvalue()
    assert 'Cache read' in table and 'Cache write' in table
    assert '$0.0123' in table
    assert 'unknown' in table
    assert 'Total' in table
    assert '3,010' in table
    assert line.startswith('Retained history cost $0.0123')
    assert format_cost(Decimal('0.00001')) == '<$0.0001'
    assert format_cost(Decimal(0)) == '$0.0000'


def test_empty_session_prints_no_table() -> None:
    output = io.StringIO()
    assert usage_command([], console=Console(file=output)) == 'No usage yet.'
    assert output.getvalue() == ''


async def test_unpriced_test_model_and_clear() -> None:
    session = Session(Agent(TestModel(model_name='no-such-model')), deps=None)
    await session.prompt('one')
    await session.prompt('two')
    usage = session_usage(session.messages)
    assert len(usage.turns) == 2
    assert usage.total.input_tokens > 0 and usage.total.output_tokens > 0
    assert usage.total.cost is None
    assert usage.unpriced == ('no-such-model',)
    output = io.StringIO()
    line = usage_command(session.messages, console=Console(file=output, width=120))
    assert 'Cache' not in output.getvalue()
    assert output.getvalue().count('unknown') == 3
    assert line.startswith('Retained history cost unknown:')
    assert 'over 2 requests' in line
    assert 'No price data for no-such-model' in line
    session.clear()
    assert cost_line(session_usage(session.messages)) == 'No usage yet.'


def test_footer_hides_cost_without_price_data() -> None:
    status = Status(model='test')
    assert '$' not in status.text()
    status.cost = Decimal('0.0123')
    assert '| $0.0123 |' in status.text()


class Priced(AbstractCapability[None]):
    """Stand in for a provider genai-prices knows: core keeps a cost that is already set."""

    async def after_model_request(
        self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        response.usage.cost = Decimal('0.0123')
        return response


async def test_shell_commands_and_footer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    values = ['hello', '/usage', '/cost', '/new', '/cost', '/exit']
    footers: list[str] = []

    class Prompt(Generic[PromptT]):
        def __init__(self, **kwargs: object) -> None:
            toolbar = kwargs['bottom_toolbar']
            assert callable(toolbar)
            self.toolbar: Callable[[], object] = toolbar

        async def prompt_async(self, label: str) -> str:
            footers.append(str(self.toolbar()))
            return values.pop(0)

    monkeypatch.setattr('pydantic_clai2._app.PromptSession', Prompt)
    output = io.StringIO()
    await chat(
        Agent(TestModel(custom_output_text='hi')),
        deps=None,
        plugins=[Priced()],
        console=Console(file=output, width=120),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    text = output.getvalue()
    assert 'Total' in text
    assert text.count('Retained history cost $0.0123: ') == 2
    assert 'No usage yet.' in text
    assert '$' not in footers[0]
    assert all('$0.0123' in footer for footer in footers[1:4])
    assert all('$' not in footer for footer in footers[4:])
