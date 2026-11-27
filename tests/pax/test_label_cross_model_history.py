"""The built-in `label_cross_model_history` plugin tells the model which earlier turns another model wrote."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.label_cross_model_history import DEFAULT_CROSS_MODEL_NOTICE
from pydantic_ai_pax import DEFAULT_PLUGINS
from pydantic_ai_pax.app import create_stock_agent
from pydantic_ai_pax.config import Settings
from pydantic_ai_pax.config.project_settings import ProjectSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import TurnStart
from pydantic_ai_pax.runtime import claude_code_sessions
from pydantic_ai_pax.runtime.session_runtime import SessionRuntime, create_shell

[DECLARATION] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'label_cross_model_history']
VOLATILE = {'timestamp', 'run_id', 'conversation_id'}


@dataclass
class Requests:
    """Stands in for every model Pax resolves, keeping each request's messages as the model got them."""

    sent: list[str] = field(default_factory=list[str])

    def resolve(self, name: str) -> Model:
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            dumped = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
            for message in dumped:
                for key in VOLATILE:
                    message.pop(key, None)
                for part in message['parts']:
                    part.pop('timestamp', None)
            self.sent.append(json.dumps(dumped))
            yield f'{name} answers'

        return FunctionModel(stream_function=stream, model_name=name)

    def notices(self) -> list[int]:
        """How many times each request states the notice."""
        return [sent.count(json.dumps(DEFAULT_CROSS_MODEL_NOTICE)[1:-1]) for sent in self.sent]


async def session(
    tmp_path: Path, *, enabled: bool = True, model: str = 'openai:gpt-6', history: list[ModelMessage] | None = None
) -> tuple[SessionRuntime[None, str], Requests]:
    store = SettingsStore(tmp_path / 'config.db')
    shell = create_shell(
        create_stock_agent(),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=Settings(model=model),
        store=store,
        builtin_plugins=[DECLARATION.model_copy(update={'enabled': enabled})],
        project=ProjectSettings(),
        message_history=history or [],
        headless=True,
    )
    requests = Requests()
    shell.session.resolve_model = requests.resolve
    await shell.loader.load_all()
    return shell, requests


async def turn(shell: SessionRuntime[None, str], text: str) -> None:
    ended = await shell.run_turn(TurnStart(text=text), headless=True)
    assert ended.outcome == 'completed', ended.error


async def test_default_on_and_labels_the_turns_before_a_model_switch(tmp_path: Path) -> None:
    assert DECLARATION.enabled
    shell, requests = await session(tmp_path)
    try:
        await turn(shell, 'Draft a plan')
        await shell.commands.execute_async('/model anthropic:claude-sonnet-4-6')
        await turn(shell, 'Review it')
        await turn(shell, 'Go on')
    finally:
        await shell.loader.close('exit')
    # Stated once, where it starts to apply; the request after it carries that same message.
    assert requests.notices() == [0, 1, 1]
    assert requests.sent[2].startswith(requests.sent[1][:-1])


async def test_one_model_sends_what_it_would_without_the_plugin(tmp_path: Path) -> None:
    sent: list[list[str]] = []
    for enabled in (True, False):
        shell, requests = await session(tmp_path / str(enabled), enabled=enabled)
        try:
            await turn(shell, 'Draft a plan')
            await turn(shell, 'Review it')
        finally:
            await shell.loader.close('exit')
        sent.append(requests.sent)
    with_plugin, without = sent
    assert len(with_plugin) == 2
    assert with_plugin == without


async def test_the_same_model_on_another_account_is_not_another_model(tmp_path: Path) -> None:
    """With `accounts.pool`, a request falls back to another account of the model, which its response records."""
    # An answer saved before responses recorded their model counts as nobody's.
    saved = [ModelRequest.user_text_prompt('Earlier'), ModelResponse(parts=[TextPart('An earlier answer')])]
    shell, requests = await session(tmp_path, model='openai-codex@default:gpt-6', history=saved)
    try:
        await turn(shell, 'Draft a plan')
        shell.session.model = 'openai-codex@work:gpt-6'
        await turn(shell, 'Review it')
        shell.session.model = 'openai-codex:gpt-6'
        await turn(shell, 'Go on')
        shell.session.model = 'openai-codex@work:gpt-6-luna'
        await turn(shell, 'Check it')
    finally:
        await shell.loader.close('exit')
    assert requests.notices() == [0, 0, 0, 1]


async def test_a_supplied_agents_own_model_is_its_own(tmp_path: Path) -> None:
    requests = Requests()
    shell = create_shell(
        Agent(requests.resolve('function:own')),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=[DECLARATION],
        project=ProjectSettings(),
        headless=True,
    )
    await shell.loader.load_all()
    try:
        await turn(shell, 'Draft a plan')
        await turn(shell, 'Review it')
    finally:
        await shell.loader.close('exit')
    assert requests.notices() == [0, 0]


async def test_an_imported_session_is_labeled(tmp_path: Path) -> None:
    transcript = tmp_path / 'claude.jsonl'
    entries = [
        {
            'type': 'user',
            'uuid': 'u1',
            'parentUuid': None,
            'cwd': str(tmp_path),
            'message': {'role': 'user', 'content': 'Fix the parser'},
        },
        {
            'type': 'assistant',
            'uuid': 'a1',
            'parentUuid': 'u1',
            'message': {
                'role': 'assistant',
                'model': 'claude-sonnet-4-6',
                'content': [{'type': 'text', 'text': 'Fixed'}],
            },
        },
    ]
    transcript.write_text(''.join(json.dumps(entry) + '\n' for entry in entries), encoding='utf-8')
    imported = claude_code_sessions.messages(transcript)
    # Continuing on the model Claude Code ran on, under the name Pax selects it by.
    shell, requests = await session(tmp_path, model='anthropic:claude-sonnet-4-6', history=imported)
    try:
        await turn(shell, 'Now add a test')
    finally:
        await shell.loader.close('exit')
    assert requests.notices() == [1]


async def test_disabling_the_plugin_stops_the_label(tmp_path: Path) -> None:
    shell, requests = await session(tmp_path)
    try:
        await shell.commands.execute_async('/plugins disable label_cross_model_history')
        await turn(shell, 'Draft a plan')
        await shell.commands.execute_async('/model anthropic:claude-sonnet-4-6')
        await turn(shell, 'Review it')
    finally:
        await shell.loader.close('exit')
    assert requests.notices() == [0, 0]
