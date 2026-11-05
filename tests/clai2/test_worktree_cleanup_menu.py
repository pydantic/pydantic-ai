"""The real exit picker, with terminal keys supplied through a pipe."""

import io

import pytest
from prompt_toolkit.input import create_pipe_input
from rich.console import Console

from pydantic_clai2.builtin_plugins.ask_user_menu import QuestionMenu
from pydantic_clai2.ui.menus import worktree_cleanup
from pydantic_clai2.ui.rendering import theme


@pytest.mark.parametrize(
    ('keys', 'expected'),
    [
        ('\r', 'keep'),
        ('\x1b', 'keep'),
        ('\x03', 'keep'),
        ('1', 'remove'),
        ('2', 'keep'),
        ('3', 'never'),
        ('\x1b[A\r', 'remove'),
        ('\x1b[B\r', 'never'),
        ('\x1b[B\x1b[B\r', 'remove'),
        ('4\r', 'keep'),
    ],
)
async def test_exit_picker(keys: str, expected: str, monkeypatch: pytest.MonkeyPatch) -> None:
    output = io.StringIO()
    console = Console(file=output, width=100, height=30)
    monkeypatch.setattr(worktree_cleanup, 'Console', lambda: console)
    original_frame = QuestionMenu.frame
    frames: list[tuple[str, ...]] = []

    def frame(menu: QuestionMenu, *, width: int, height: int) -> tuple[str, ...]:
        rendered = original_frame(menu, width=width, height=height)
        frames.append(rendered)
        return rendered

    monkeypatch.setattr(QuestionMenu, 'frame', frame)
    with create_pipe_input() as pipe:
        monkeypatch.setattr('pydantic_clai2.ui.prompt.question_input.create_input', lambda: pipe)
        pipe.send_text(keys)
        assert await worktree_cleanup.ask_removal('/tmp/checkout\nThe branch will be kept.') == expected

    initial = frames[0]
    assert 'Remove this worktree?' in initial[0]
    assert initial[1] == theme.sgr(theme.MUTED) + '/tmp/checkout\x1b[0m'
    assert initial[2] == theme.sgr(theme.MUTED) + 'The branch will be kept.\x1b[0m'
    assert initial[3] == '  1. Yes'
    assert '> 2. No' in initial[4]
    assert initial[5] == "  3. No, don't ask again for this worktree"
    assert 'Other' not in output.getvalue()


async def test_exit_picker_accepts_long_details(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local cleanup details are not subject to the model's question-length limit."""
    output = io.StringIO()
    monkeypatch.setattr(worktree_cleanup, 'Console', lambda: Console(file=output))
    with create_pipe_input() as pipe:
        monkeypatch.setattr('pydantic_clai2.ui.prompt.question_input.create_input', lambda: pipe)
        pipe.send_text('\r')
        assert await worktree_cleanup.ask_removal('/tmp/checkout\n' + 'argument ' * 80 + '\x1b[2J') == 'keep'
    assert '\\x1b[2J' in output.getvalue()
