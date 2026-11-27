"""`run.request_limit` ends a turn with an answer: the model is told what is left and answers on the last request."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.report_remaining_budget import RemainingBudgetPart
from pydantic_ai_harness.wrap_up_at_limits import DEFAULT_WRAP_UP_PROMPT, WrapUpPart, wrapped_up_by
from pydantic_ai_pax.app import create_stock_agent
from pydantic_ai_pax.cli import headless
from pydantic_ai_pax.config import Settings
from pydantic_ai_pax.config.project_settings import ProjectSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.runtime.session import Session
from tests.function_model_prefix import REQUEST_SCOPED_PROMPT_DROPPED


def _notes(message: ModelMessage) -> list[str]:
    """The request-scoped system notes `FunctionModel` was sent in `message`, as it renders them."""
    assert isinstance(message, ModelRequest)
    return [
        part.content.removeprefix('<system>').removesuffix('</system>')
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str) and part.content.startswith('<system>')
    ]


def _busy_model(sent: list[list[str]], tool: str) -> FunctionModel:
    """A model that keeps calling `tool` until it may not, recording the notes each of its requests carried."""

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        # Request-scoped notes are in the request just sent only.
        sent.append(_notes(messages[-1]))
        if (info.model_settings or {}).get('tool_choice') == 'none':
            yield 'what I found so far'
        else:
            yield {0: DeltaToolCall(name=tool, json_args='{"path": "notes.txt"}')}

    return FunctionModel(stream_function=respond)


def _busy_agent(sent: list[list[str]]) -> Agent[None, str]:
    """Your own agent (`--agent`), with a tool of its own."""
    agent = Agent(_busy_model(sent, 'look'), deps_type=type(None))

    @agent.tool_plain
    def look(path: str) -> str:
        return f'more to look at in {path}'

    return agent


@pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
@pytest.mark.parametrize('supplied', [False, True])
async def test_headless_run_at_the_request_limit_ends_with_an_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], supplied: bool
) -> None:
    """`pax -p` on the stock agent and on your own `--agent`, which Pax runs under the same limit."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'notes.txt').write_text('Notes.')
    sent: list[list[str]] = []
    monkeypatch.setattr(headless, 'create_agent', lambda: create_stock_agent(_busy_model(sent, 'read_file')))
    assert (
        await headless.run_headless(
            text='look around',
            settings=Settings(model=None, request_limit=3, code_mode=False),
            store=SettingsStore(tmp_path / 'config.db'),
            project=ProjectSettings(),
            agent=_busy_agent(sent) if supplied else None,
        )
        == 0
    )
    assert capsys.readouterr().out == 'what I found so far\n'
    assert sent == [
        ['Budget remaining:\n- this run: 3 requests including this one'],
        ['Budget remaining:\n- this run: 2 requests including this one'],
        ['Budget remaining:\n- this run: 1 request including this one', DEFAULT_WRAP_UP_PROMPT],
    ]


@pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
async def test_session_follows_a_changed_request_limit() -> None:
    sent: list[list[str]] = []
    session = Session(_busy_agent(sent), deps=None, usage_limits=UsageLimits(request_limit=2))
    result = await session.prompt('look around')
    assert result.output == 'what I found so far'
    assert wrapped_up_by(result.new_messages()) == 'request_limit'
    assert len(sent) == 2
    # What `/set run.request_limit` does between turns.
    session.usage_limits = UsageLimits(request_limit=1)
    result = await session.prompt('once more')
    assert result.output == 'what I found so far'
    assert sent[2:] == [['Budget remaining:\n- this run: 1 request including this one', DEFAULT_WRAP_UP_PROMPT]]


async def test_no_usage_limits_add_nothing() -> None:
    """A host that passes no `usage_limits`, whose runs get core's default limits, as before."""

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        assert _notes(messages[-1]) == []
        assert (info.model_settings or {}).get('tool_choice') is None
        yield 'done'

    session = Session(Agent(FunctionModel(stream_function=respond)), deps=None)
    result = await session.prompt('hi')
    assert result.output == 'done'
    parts = [part for message in result.all_messages() for part in message.parts]
    assert not any(isinstance(part, RemainingBudgetPart | WrapUpPart) for part in parts)
