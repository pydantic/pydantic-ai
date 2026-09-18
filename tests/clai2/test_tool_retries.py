"""Tool retry settings reach native agent execution without changing tool overrides."""

import io
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from menu_script import make_context
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from rich.console import Console
from test_app_edges import inputs

from pydantic_clai2 import Session, chat
from pydantic_clai2.config import Settings
from pydantic_clai2.set_menu import SettingsSource
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def test_menu_validates_persists_and_resets_tool_retries(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    source = SettingsSource(context)
    row = next(row for row in source.rows() if row.key == 'run.tool_retries')
    assert row.default == source.current(row) == '3'
    for invalid in ('-1', '1.5', 'true', 'null', 'bad'):
        assert source.problem(row, invalid) is not None
    assert source.problem(row, '0') is None
    source.apply(row, '0')
    assert context.store.load().tool_retries == 0
    assert applied == ['run.tool_retries']
    source.reset(row)
    assert source.current(row) == '3'
    assert context.store.load().tool_retries == 3


@pytest.mark.parametrize('saved', [None, 5])
@pytest.mark.parametrize('explicit', [None, 1])
async def test_chat_applies_retry_budget_between_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: int | None, explicit: int | None
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    context, _ = make_context(tmp_path)
    if saved is not None:
        context.set_setting(['run.tool_retries', str(saved)])
    budgets: list[int] = []
    agent = Agent(TestModel(custom_output_text='done'), deps_type=type(None), retries={'tools': 8, 'output': 7})

    @agent.tool(retries=explicit)
    def retrying(ctx: RunContext[None]) -> str:
        budgets.append(ctx.retry)
        raise ModelRetry('Try again')

    inputs(
        monkeypatch,
        ['first', '/set run.tool_retries 0', '/new', 'second', '/set run.tool_retries 3', '/new', 'third', '/exit'],
    )
    await chat(
        agent,
        deps=None,
        settings=Settings(model=None, tool_retries=store.load().tool_retries, session_namer=False),
        store=store,
        console=Console(file=io.StringIO(), width=120),
    )
    limits = [saved if saved is not None else 3, 0, 3]
    expected = [retry for limit in limits for retry in range((explicit if explicit is not None else limit) + 1)]
    assert budgets == expected
    assert store.load().tool_retries == 3


async def test_tool_budget_does_not_override_output_validation() -> None:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'done'

    agent = Agent(FunctionModel(stream_function=respond), deps_type=type(None), retries={'output': 4})
    attempts: list[int] = []

    @agent.output_validator
    def validate(ctx: RunContext[None], output: str) -> str:
        attempts.append(ctx.retry)
        if ctx.retry < 4:
            raise ModelRetry('Try again')
        return output

    session = Session(agent, deps=None)
    session.tool_retries = 0
    assert (await session.prompt('run')).output == 'done'
    assert attempts == [0, 1, 2, 3, 4]
