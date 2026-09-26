"""`!command` input runs in the system shell and never starts an agent turn."""

import io
import time
from pathlib import Path

import pytest
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from rich.console import Console
from test_app_edges import inputs

from pydantic_clai2 import chat
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.shell_passthrough import shell_command


@pytest.mark.parametrize(
    ('text', 'command'),
    [
        ('!ls -lh', 'ls -lh'),
        ('  !git status  ', 'git status'),
        ('!  echo hi', 'echo hi'),
        ('!', None),
        ('!   ', None),
        ('  !  ', None),
        ('ls !', None),
        ('/help', None),
        ('hello', None),
    ],
)
def test_shell_command_detection(text: str, command: str | None) -> None:
    assert shell_command(text) == command


async def shell_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, values: list[str | BaseException], *, agent_turns: int = 0
) -> str:
    """Run the interactive loop, checking how many inputs reached the model."""
    requests: list[ModelRequestContext] = []

    class CountRequests(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            requests.append(request_context)
            return request_context

    monkeypatch.chdir(tmp_path)
    inputs(monkeypatch, values)
    output = io.StringIO()
    await chat(
        Agent(TestModel(custom_output_text='agent reply'), deps_type=type(None), capabilities=[CountRequests()]),
        deps=None,
        console=Console(file=output, width=120),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    assert len(requests) == agent_turns
    return output.getvalue()


class TestShellPassthrough:
    async def test_runs_in_cwd_without_agent_turn(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await shell_session(tmp_path, monkeypatch, ['  !printf hi > marker.txt  ', '/exit'])
        assert (tmp_path / 'marker.txt').read_text() == 'hi'
        assert '$ printf hi > marker.txt' in text
        assert 'Shell passthrough, not sent to the agent' in text
        assert 'Done (' in text
        assert 'agent reply' not in text

    async def test_reports_exit_code(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await shell_session(tmp_path, monkeypatch, ['!exit 3', '/exit'])
        assert 'Exit code 3 (' in text

    async def test_ctrl_c_interrupts_command_not_clai(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The shell signals CLAI as the terminal would on Ctrl-C; the exec'd child never sees it, so it is killed.
        started = time.monotonic()
        text = await shell_session(
            tmp_path, monkeypatch, ['!kill -INT $PPID; exec sleep 30', '!printf after > marker.txt', '/exit']
        )
        assert time.monotonic() - started < 10
        assert 'Interrupted (' in text
        assert (tmp_path / 'marker.txt').read_text() == 'after'

    async def test_second_ctrl_c_exits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await shell_session(tmp_path, monkeypatch, [KeyboardInterrupt(), '!kill -INT $PPID; exec sleep 30'])
        assert 'Input cleared' in text
        assert 'Interrupted (' in text

    async def test_spawn_failure_is_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        async def unavailable(command: str) -> None:
            raise FileNotFoundError('no shell')

        monkeypatch.setattr('pydantic_clai2.shell_passthrough.asyncio.create_subprocess_shell', unavailable)
        text = await shell_session(tmp_path, monkeypatch, ['!ls', '/exit'])
        assert 'Shell error: no shell' in text

    async def test_nul_byte_is_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await shell_session(tmp_path, monkeypatch, ['!echo a\x00b', '/exit'])
        assert 'Shell error: embedded null byte' in text

    async def test_bare_bang_is_a_prompt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await shell_session(tmp_path, monkeypatch, ['!', '/exit'], agent_turns=1)
        assert 'agent reply' in text

    async def test_help_mentions_passthrough(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await shell_session(tmp_path, monkeypatch, ['/help', '/exit'])
        assert '!COMMAND: Run COMMAND with the system shell' in text
