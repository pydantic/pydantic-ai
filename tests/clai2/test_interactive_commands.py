"""Which `!` commands get the terminal instead of captured output."""

import pytest

from pydantic_clai2.cli.interactive_commands import needs_terminal
from pydantic_clai2.cli.shell_passthrough import terminal_command


@pytest.mark.parametrize(
    'command',
    [
        'nano .env',
        'vim',
        'nvim src/app.py',
        'emacs -nw notes.txt',
        'less README.md',
        'man ls',
        'htop',
        'top -o cpu',
        'ssh host uptime',
        '/usr/bin/vim file',
        'vim.exe file',
        'git log | less',
        'echo hi; nano notes.txt',
        '(cd src && top)',
        'less < notes.txt',
        'FOO=1 nano x',
        'sudo -E vim /etc/hosts',
        'sudo -u alice nano file',
        'sudo --user alice -E vim file',
        'nice -n 10 vim file',
        'nice -10 top',
        'env -u HOME EDITOR=vim vim x',
        'time -o log.txt less file',
        'command -p vi',
        'python',
        'python3',
        'python3.12',
        '/usr/bin/python3',
        'ipython',
        'node',
        'bash',
        'sqlite3',
        'git commit',
        'git commit --amend',
        'git commit -v',
        'git -c core.editor=vim commit',
        'git -C repo commit -a',
        'git rebase -i HEAD~3',
        'git rebase --interactive main',
        'git add -p',
        'git add --patch src',
        'git checkout -p',
        'git stash -p',
    ],
)
def test_needs_terminal(command: str) -> None:
    assert needs_terminal(command)


@pytest.mark.parametrize(
    'command',
    [
        '',
        'ls -la',
        'echo vim',
        'echo hi > vim',
        'npm test',
        'python script.py',
        'python -c "print(1)"',
        'sh -c "echo hi"',
        'node index.js',
        'env',
        'FOO=1',
        'sudo -E',
        'sudo -u alice',
        'sudo -u alice ls',
        'nice -n 10 make',
        'git',
        'git --version',
        'git status',
        'git log',
        'git commit -m fix',
        'git commit -am fix',
        'git commit --message=fix',
        'git commit --amend --no-edit',
        'git commit -F msg.txt',
        'git -C repo commit --fixup=HEAD',
        'git rebase main',
        'git add .',
        'git add -A',
        'git checkout -b feature',
        "vim 'unclosed",
    ],
)
def test_captured(command: str) -> None:
    assert not needs_terminal(command)


@pytest.mark.parametrize(
    ('command', 'interactive'),
    [
        ('!ls', 'ls'),
        ('! echo hi', 'echo hi'),
        ('nano .env', 'nano .env'),
        ('ls', None),
    ],
)
def test_terminal_command(command: str, interactive: str | None) -> None:
    """A leading `!`, typed as `!!command`, forces the terminal for any command."""
    assert terminal_command(command) == interactive
