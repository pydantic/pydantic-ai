"""Outside `/permissions full`, a change to Pax's own settings always asks the user, and is denied with no one to ask."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.approval_reviewer import ApprovalReviewer
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.permissions import PermissionDecision, PermissionRequest
from pydantic_ai_pax.builtin_plugins.web.live import BrowserAnswerer
from pydantic_ai_pax.config import PermissionMode
from pydantic_ai_pax.runtime.permissions import (
    PROTECTED,
    UNANSWERED,
    approval_handler,
    approval_question,
    session_permissions,
)
from tests.pax.test_permissions import runtime
from tests.pax.test_speculative_mode import streamed


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The user's Pax folder, inside the workspace so the file tools can reach it."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg'))
    return tmp_path / 'xdg' / 'pydantic-ai-pax'


def calling(*calls: tuple[str, Mapping[str, object]]) -> FunctionModel:
    """A model that makes `calls` one per step, then repeats the last tool result it got."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step < len(calls):
            name, args = calls[step]
            return ModelResponse(parts=[ToolCallPart(name, dict(args), tool_call_id=f'call{step}')])
        returned = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        return ModelResponse(parts=[TextPart(f'Tool said: {returned[0].content}' if returned else 'done')])

    return streamed(respond)


class Person:
    """Denies every call, keeping what it was asked."""

    def __init__(self) -> None:
        self.asked: list[PermissionRequest] = []

    async def __call__(self, request: PermissionRequest) -> PermissionDecision:
        self.asked.append(request)
        return PermissionDecision(approved=False)


RAN: list[str] = []


def run_command(command: str) -> str:
    """A shell tool declared as harness `Shell` declares its own, that records the command instead of running it."""
    RAN.append(command)
    return f'ran {command}'


async def run(
    workspace: Path,
    *calls: tuple[str, Mapping[str, object]],
    mode: PermissionMode = 'auto',
    database: Path | None = None,
    extra: Sequence[AbstractCapability[None]] = (),
) -> Person:
    RAN.clear()
    person = Person()
    capabilities = [
        LocalWorkspace(workspace),
        FileSystem[None](),
        *session_permissions(lambda: mode, database=database),
        *extra,
        approval_handler(person),
    ]
    shell = Tool(run_command, metadata={'code_arg_name': 'command', 'code_arg_language': 'shell'})
    await Agent(calling(*calls), deps_type=type(None), tools=[shell], capabilities=capabilities).run('go')
    return person


def write(path: str) -> tuple[str, Mapping[str, object]]:
    return ('write_file', {'path': path, 'content': '{}'})


@pytest.mark.parametrize(
    'path',
    [
        pytest.param('xdg/pydantic-ai-pax/config.db', id='settings-database'),
        pytest.param('xdg/pydantic-ai-pax/credentials-default.json', id='credentials'),
        pytest.param('xdg/pydantic-ai-pax/plugins/mine.py', id='plugins-folder'),
        pytest.param('.pax/settings.json', id='project-folder'),
        pytest.param('sub/.clai/settings.json', id='legacy-project-folder'),
        pytest.param('.git/hooks/pre-commit', id='git-hooks'),
        pytest.param('db/config.db-wal', id='database-elsewhere'),
        pytest.param('db/plugins/mine.py', id='plugins-next-to-database-elsewhere'),
    ],
)
async def test_auto_asks_before_a_write_to_paxs_settings(tmp_path: Path, config_home: Path, path: str) -> None:
    (tmp_path / path).parent.mkdir(parents=True)
    person = await run(tmp_path, write(path), database=tmp_path / 'db' / 'config.db')
    [asked] = person.asked
    assert asked.protected and asked.subject == path
    assert approval_question(asked).questions[0].question == f'{PROTECTED} Write {path}?'
    assert not (tmp_path / path).exists()


async def test_a_reviewer_that_says_safe_still_asks(tmp_path: Path, config_home: Path) -> None:
    def safe(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'response': 'no'})])

    (tmp_path / '.pax').mkdir()
    reviewer = ApprovalReviewer[None](FunctionModel(safe), question='Could this call do harm?')
    person = await run(tmp_path, write('.pax/settings.json'), extra=[reviewer])
    assert [asked.subject for asked in person.asked] == ['.pax/settings.json']
    assert not (tmp_path / '.pax' / 'settings.json').exists()


@pytest.mark.parametrize(
    'command',
    [
        'pax config set model openai:gpt-5.5',
        'uv run pax plugins enable hooks',
        'echo x > .pax/settings.json',
    ],
)
async def test_auto_asks_before_a_command_that_changes_paxs_settings(
    tmp_path: Path, config_home: Path, command: str
) -> None:
    (tmp_path / '.pax').mkdir()
    person = await run(tmp_path, ('run_command', {'command': command}))
    [asked] = person.asked
    assert asked.protected and asked.subject == command
    assert approval_question(asked).questions[0].question == f'{PROTECTED} Run this command?\n{command}'
    assert RAN == []


async def test_reads_and_other_writes_run(tmp_path: Path, config_home: Path) -> None:
    (tmp_path / '.pax').mkdir()
    (tmp_path / '.pax' / 'settings.json').write_text('{}')
    person = await run(
        tmp_path,
        ('read_file', {'path': '.pax/settings.json'}),
        ('run_command', {'command': 'cat .pax/settings.json'}),
        ('run_command', {'command': 'pax config show'}),
        write('notes.md'),
    )
    assert person.asked == []
    assert RAN == ['cat .pax/settings.json', 'pax config show']
    assert (tmp_path / 'notes.md').read_text() == '{}'


async def test_full_lets_a_change_to_paxs_settings_run(tmp_path: Path, config_home: Path) -> None:
    """`/permissions full` is the user's choice to let everything run, Pax's own settings included."""
    (tmp_path / '.pax').mkdir()
    person = await run(tmp_path, write('.pax/settings.json'), mode='full')
    assert person.asked == []
    assert (tmp_path / '.pax' / 'settings.json').read_text() == '{}'


async def test_a_session_no_one_can_be_asked_in_denies_a_change_to_paxs_settings(
    tmp_path: Path, config_home: Path
) -> None:
    """Headless, the write to the plugins folder beside the session's settings database is denied."""
    (tmp_path / 'plugins').mkdir()
    agent = Agent(calling(write('plugins/mine.py')), deps_type=type(None), capabilities=[FileSystem[None]()])
    shell, output = runtime(tmp_path, agent)
    shell.send_input('/permissions auto')
    shell.send_input('add a plugin')
    shell.send_input('/exit')
    assert await shell.run() == 'exit'
    assert f'Tool said: {UNANSWERED}' in output.getvalue()
    assert not (tmp_path / 'plugins' / 'mine.py').exists()


async def test_the_browser_card_gives_the_same_reason(tmp_path: Path, config_home: Path) -> None:
    (tmp_path / '.pax').mkdir()
    [held] = (await run(tmp_path, write('.pax/settings.json'))).asked
    answerer = BrowserAnswerer(publish=lambda message, _: None, terminal=None)
    async with anyio.create_task_group() as group:
        group.start_soon(answerer.approve, held)
        await anyio.wait_all_tasks_blocked()
        [questions] = answerer.questions
        assert questions.approval is not None and questions.approval.reason == PROTECTED
        answerer.reply(questions.id, None)
