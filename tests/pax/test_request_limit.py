"""`run.request_limit` ends a turn with an answer: the last request the limit allows is made to answer."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.wrap_up_at_limits import DEFAULT_WRAP_UP_PROMPT, wrapped_up_by
from pydantic_ai_pax.app import create_stock_agent
from pydantic_ai_pax.cli import headless
from pydantic_ai_pax.config import Settings
from pydantic_ai_pax.config.project_settings import ProjectSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.runtime.session import Session
from tests.function_model_prefix import REQUEST_SCOPED_PROMPT_DROPPED


@dataclass
class Sent:
    """What one model request carried: its system notes, its `tool_choice`, and its prompt and tool results."""

    notes: list[str]
    tool_choice: object
    parts: list[tuple[str, str]]


def _sent(message: ModelMessage, info: AgentInfo) -> Sent:
    assert isinstance(message, ModelRequest)
    texts = [
        (part.part_kind, str(part.content))
        for part in message.parts
        if isinstance(part, UserPromptPart | ToolReturnPart)
    ]
    return Sent(
        # How `FunctionModel` renders a request-scoped system prompt, such as the wrap-up.
        notes=[
            text.removeprefix('<system>').removesuffix('</system>') for _, text in texts if text.startswith('<system>')
        ],
        tool_choice=(info.model_settings or {}).get('tool_choice'),
        # Without what differs between runs, such as timestamps and tool call IDs.
        parts=[(kind, text) for kind, text in texts if not text.startswith('<system>')],
    )


def _busy_model(sent: list[Sent], tool: str, *, calls: int = 100) -> FunctionModel:
    """A model that calls `tool` `calls` times, then answers; or answers as soon as tools are unavailable."""

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        # Request-scoped prompts are in the request just sent only.
        sent.append(_sent(messages[-1], info))
        if sent[-1].tool_choice == 'none' or len(sent) > calls:
            yield 'what I found so far'
        else:
            yield {0: DeltaToolCall(name=tool, json_args='{"path": "notes.txt"}')}

    return FunctionModel(stream_function=respond)


def _busy_agent(sent: list[Sent], *, calls: int = 100) -> Agent[None, str]:
    """Your own agent (`--agent`), with a tool of its own."""
    agent = Agent(_busy_model(sent, 'look', calls=calls), deps_type=type(None))

    @agent.tool_plain
    def look(path: str) -> str:
        return f'more to look at in {path}'

    return agent


@pytest.mark.parametrize('supplied', [False, True])
async def test_headless_run_at_the_request_limit_ends_with_an_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], supplied: bool
) -> None:
    """`pax -p` on the stock agent and on your own `--agent`, which Pax runs under the same limit."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'notes.txt').write_text('Notes.')
    sent: list[Sent] = []
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
    assert [(request.notes, request.tool_choice) for request in sent] == [
        ([], None),
        ([], None),
        ([DEFAULT_WRAP_UP_PROMPT], 'none'),
    ]


async def test_only_the_last_request_changes() -> None:
    """The requests before the last are what they are without a limit; the last answers."""
    unlimited: list[Sent] = []
    result = await Session(_busy_agent(unlimited, calls=3), deps=None).prompt('look around')
    assert result.output == 'what I found so far'
    limited: list[Sent] = []
    session = Session(_busy_agent(limited, calls=3), deps=None, usage_limits=UsageLimits(request_limit=3))
    result = await session.prompt('look around')
    assert result.output == 'what I found so far'
    assert wrapped_up_by(result.new_messages()) == 'request_limit'
    assert len(unlimited) == 4
    assert limited[:2] == unlimited[:2]
    assert limited[2:] == [Sent(notes=[DEFAULT_WRAP_UP_PROMPT], tool_choice='none', parts=unlimited[2].parts)]


# The next turn's requests leave out the wrap-up prompt the turn before ended on.
@pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
async def test_session_follows_a_changed_request_limit() -> None:
    sent: list[Sent] = []
    session = Session(_busy_agent(sent), deps=None, usage_limits=UsageLimits(request_limit=2))
    result = await session.prompt('look around')
    assert result.output == 'what I found so far'
    assert len(sent) == 2
    # What `/set run.request_limit` does between turns.
    session.usage_limits = UsageLimits(request_limit=1)
    result = await session.prompt('once more')
    assert result.output == 'what I found so far'
    assert wrapped_up_by(result.new_messages()) == 'request_limit'
    assert [(request.notes, request.tool_choice) for request in sent[2:]] == [([DEFAULT_WRAP_UP_PROMPT], 'none')]


async def test_no_usage_limits_add_nothing() -> None:
    """A host that passes no `usage_limits` still has its turn stopped by core's default limit, as before."""
    sent: list[Sent] = []
    with pytest.raises(UsageLimitExceeded):
        await Session(_busy_agent(sent), deps=None).prompt('look around')
    assert len(sent) == UsageLimits().request_limit
    assert all(request.notes == [] and request.tool_choice is None for request in sent)
