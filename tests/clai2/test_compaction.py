"""The built-in `compaction` plugin, loaded with `load_plugin` and driven through one `chat()` run."""

import io
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError
from rich.console import Console

from pydantic_ai import Agent, ModelHTTPError, capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.compaction import FallbackCompaction
from pydantic_clai2 import Session, chat
from pydantic_clai2.builtin_plugins.compaction import CompactionPlugin
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import LoadedPlugin, PluginHost, SessionEnd, Transcript, load_plugin
from tests.clai2.test_app_edges import inputs


def make_plugin(
    conversation: Transcript | Session[None, str] | None = None, **settings: JsonValue
) -> LoadedPlugin[None]:
    host = PluginHost[None](
        name='compaction', console=Console(file=io.StringIO()), settings=dict(settings), conversation=conversation
    )
    return load_plugin(CompactionPlugin, host)


def summary_prompt(summary_run: Sequence[ModelMessage]) -> str:
    """The user turn the summariser received, as `capture_run_messages` recorded it."""
    request = summary_run[0]
    assert isinstance(request, ModelRequest)
    [prompt] = request.parts
    assert isinstance(prompt, UserPromptPart) and isinstance(prompt.content, str)
    return prompt.content


def two_turns() -> list[ModelMessage]:
    """Two request/response pairs; the chain always keeps the newest pair intact."""
    return [
        ModelRequest.user_text_prompt('hello there'),
        ModelResponse(parts=[TextPart('hi')]),
        ModelRequest.user_text_prompt('and again'),
        ModelResponse(parts=[TextPart('yo')]),
    ]


async def test_compact_sends_the_history_and_focus_to_the_summariser() -> None:
    transcript = Transcript(messages=two_turns(), model=TestModel(custom_output_text='the gist'))
    plugin = make_plugin(transcript, protected_tokens=0)
    plugin.host.status.context_alert = True
    with capture_run_messages() as summary_run:
        notice = await plugin.commands.execute_async('/compact the auth work')
    prompt = summary_prompt(summary_run)
    assert 'User: hello there\nAssistant: hi\nUser: and again' in prompt
    assert prompt.endswith('Give particular weight to: the auth work')
    assert notice.startswith('Compacted 4 messages down to 3; about ') and notice.endswith(' tokens saved.')
    summary, first_request, last_response = transcript.messages
    assert isinstance(summary, ModelRequest) and isinstance(first_request, ModelRequest)
    [summary_part], [request_part] = summary.parts, first_request.parts
    assert isinstance(summary_part, SystemPromptPart) and isinstance(request_part, UserPromptPart)
    assert summary_part.content == 'Summary of previous conversation:\n\nthe gist'
    assert request_part.content == 'hello there', 'harness keeps the first user message verbatim'
    assert isinstance(last_response, ModelResponse)
    assert plugin.host.status.context_alert, 'the colour follows the figure: both wait for the next reading'


async def test_compact_says_when_there_is_nothing_to_do() -> None:
    assert await make_plugin().commands.execute_async('/compact') == 'Nothing to compact: the conversation is empty.'
    short = Transcript(messages=[ModelRequest.user_text_prompt('hi')], model='test')
    plugin = make_plugin(short)
    with capture_run_messages() as summary_run:
        notice = await plugin.commands.execute_async('/compact')
    assert notice == 'Nothing to compact: the last 50,000 tokens are always kept.' and not summary_run
    with pytest.raises(ValueError, match='Choose a model first'):
        await make_plugin(
            Transcript(messages=[ModelRequest.user_text_prompt('hi')]), protected_tokens=0
        ).commands.execute_async('/compact')


def assert_truncated_without_a_summary(transcript: Transcript) -> None:
    request, response = transcript.messages
    assert isinstance(request, ModelRequest) and isinstance(response, ModelResponse)
    assert all(isinstance(part, UserPromptPart) for part in request.parts), 'no SystemPromptPart summary'


async def test_truncation_strategy_drops_older_messages_without_a_summary() -> None:
    transcript = Transcript(messages=two_turns(), model=TestModel())
    plugin = make_plugin(transcript, strategy='truncation', protected_tokens=0)
    with capture_run_messages() as summary_run:
        notice = await plugin.commands.execute_async('/compact')
    assert notice.startswith('Compacted 4 messages down to 2;') and not summary_run
    assert_truncated_without_a_summary(transcript)


async def test_summariser_failure_falls_back_to_truncation() -> None:
    def refuse(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=503, model_name='down', body=None)

    transcript = Transcript(messages=two_turns(), model=FunctionModel(refuse))
    plugin = make_plugin(transcript, protected_tokens=0)
    notice = await plugin.commands.execute_async('/compact')
    assert notice.startswith('Compacted 4 messages down to 2;')
    assert_truncated_without_a_summary(transcript)


async def test_settings_are_validated_on_activation() -> None:
    with pytest.raises(ValidationError):
        make_plugin(threshold=0)
    with pytest.raises(ValidationError):
        make_plugin(strategy='magic')
    with pytest.raises(ValidationError):
        make_plugin(compact_at=0.5)


@pytest.mark.parametrize('strategy', ['summarization', 'truncation'])
async def test_direct_fallback_capability_compacts_before_gauging(strategy: str) -> None:
    session = Session(Agent(TestModel(custom_output_text='gist')), deps=None)
    plugin = make_plugin(session, strategy=strategy, context_window=1000, protected_tokens=0)
    assert isinstance(plugin.capabilities[0], FallbackCompaction)
    session.plugins = plugin.capabilities
    session.replace_messages(
        [
            ModelRequest.user_text_prompt('first'),
            ModelResponse(parts=[TextPart('reply')]),
            ModelRequest.user_text_prompt('old ' * 1000),
            ModelResponse(parts=[TextPart('reply')]),
        ]
    )
    await session.prompt('new')
    assert not any(
        isinstance(part, UserPromptPart) and part.content == 'old ' * 1000
        for message in session.messages
        for part in message.parts
    )
    assert any(isinstance(part, SystemPromptPart) for message in session.messages for part in message.parts) == (
        strategy == 'summarization'
    )
    assert not plugin.host.status.context_alert, 'the gauge measures the compacted request'
    assert plugin.host.status.context_tokens is not None and plugin.host.status.context_tokens < 850
    assert plugin.host.status.context_window == 1000
    cramped = make_plugin(session, strategy=strategy, context_window=1, protected_tokens=50_000)
    session.plugins = cramped.capabilities
    await session.prompt('again')
    assert cramped.host.status.context_alert, 'a protected tail can still exceed the threshold'


@pytest.mark.parametrize(
    ('model_window', 'override', 'expected'),
    [(1_000_000, None, 1_000_000), (1_000_000, 200_000, 200_000), (None, None, None)],
)
async def test_gauge_uses_known_window_not_fallback(
    model_window: int | None, override: int | None, expected: int | None
) -> None:
    model = TestModel(profile={'context_window': model_window})
    session = Session(Agent(model), deps=None)
    plugin = make_plugin(session, context_window=override)
    plugin.host.status.context_window = 123
    session.plugins = plugin.capabilities
    await session.prompt('hello')
    assert plugin.host.status.context_window == expected
    assert plugin.host.status.context_tokens is not None


async def test_unloading_clears_the_context_window_and_alert() -> None:
    plugin = make_plugin()
    plugin.host.status.context_alert = True
    plugin.host.status.context_tokens = 123
    plugin.host.status.context_window = 1_000_000
    await plugin.dispatch(SessionEnd(reason='exit'))
    assert not plugin.host.status.context_alert
    assert plugin.host.status.context_window is None
    assert plugin.host.status.context_tokens == 123


async def test_shell_loads_the_plugin_and_compacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['first', 'second', '/compact', '/plugins list', '/exit'])
    output = io.StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        console=Console(file=output, width=200),
        settings=Settings(model='test'),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(
            PluginSettings(
                id='compaction', factory='pydantic_clai2.builtin_plugins.compaction', settings={'protected_tokens': 0}
            ),
        ),
    )
    text = output.getvalue()
    assert 'Compacted 4 messages down to 3' in text
    assert 'compaction' in text and 'pydantic_clai2.builtin_plugins.compaction (built-in)' in text
