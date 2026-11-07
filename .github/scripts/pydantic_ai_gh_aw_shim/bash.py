"""Claude's `Bash` tool -- run a shell command in the workspace.

Backed by pydantic-ai-harness's `ShellToolset`, which runs the command in the
run's workspace with the per-command timeout. The Claude `Bash` signature
(`command`, optional `timeout` in seconds), the sandbox PATH augmentation, and
the tail-keeping output cap are preserved by the adapter.
"""

import shlex
from collections.abc import Awaitable, Callable
from typing import TypeAlias

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import WorkspaceError

from ._backends import BASH_DEFAULT_TIMEOUT, BASH_MAX_TIMEOUT, run_shell, timed_out

# Token prefixes of the `Bash(...)` entries in `--allowed-tools`; a command must start with one of them.
BashScopes: TypeAlias = tuple[tuple[str, ...], ...]
# The pipeline and list operators that join simple commands. Each joined command is checked on its own.
_COMMAND_SEPARATORS = frozenset({'|', '||', '&&', ';'})


async def bash(ctx: RunContext[object], command: str, timeout: int | None = None) -> str:
    """Run a shell command in the repository workspace.

    Returns the command's labeled stdout/stderr (truncated). `timeout` is in
    seconds (default 120, capped at 600).
    """
    secs = BASH_DEFAULT_TIMEOUT if not timeout or timeout <= 0 else min(int(timeout), BASH_MAX_TIMEOUT)
    try:
        out = await run_shell(ctx, command, timeout_seconds=float(secs))
    except (ModelRetry, OSError, WorkspaceError) as exc:
        # ModelRetry: the harness blocked the command; the shim's tools have always
        # surfaced such conditions as a returned error string rather than a
        # model-facing retry. OSError / WorkspaceError: the workspace could not start
        # the command (e.g. its working directory does not exist) -- the harness
        # doesn't convert those, so catch them here rather than abort the whole run.
        return f'error: {exc}'
    # On timeout the harness *returns* output ending in a `[Command timed out
    # after Ns]` line rather than raising. The old tool surfaced timeouts as an `error:` string
    # (and `Grep` already wraps the same sentinel), so do the same here instead
    # of handing the model an unprefixed result it might read as success.
    if timed_out(out):
        return f'error: {out}'
    return out


def _simple_commands(command: str) -> list[list[str]] | None:
    """The words of each simple command joined in `command`, or `None` where the shell would read other syntax.

    Quotes and backslashes are followed the way the shell follows them, so a quoted `|` stays an argument. Refused:
    redirection, subshells, background jobs, command substitution, `$'...'` strings and unterminated quotes.
    """
    segments: list[str] = []
    start = index = 0
    quote = ''
    while index < len(command):
        char = command[index]
        substitution = char == '`' or command.startswith('$(', index)
        if char == '\\' and quote != "'":
            index += 2
            continue
        if quote:
            if char == quote:
                quote = ''
            elif quote == '"' and substitution:
                return None
        elif char in '\'"':
            quote = char
        elif substitution or command.startswith("$'", index) or char in '()<>':
            return None
        elif char in '|&;':
            end = index
            while end < len(command) and command[end] in '|&;':
                end += 1
            if command[index:end] not in _COMMAND_SEPARATORS:
                return None
            segments.append(command[start:index])
            start = index = end
            continue
        index += 1
    if quote:
        return None
    try:
        return [shlex.split(segment) for segment in [*segments, command[start:]]]
    except ValueError:
        return None


def scoped_bash(scopes: BashScopes) -> Callable[[RunContext[object], str, int | None], Awaitable[str]]:
    """`bash` limited to commands whose every simple command starts with the tokens of one of `scopes`.

    Pipelines and `;`, `&&` and `||` lists are checked command by command. Whatever a prefix cannot scope is
    refused: newlines, redirection, background jobs, subshells, command substitution and `$'...'` strings.
    """
    allowed = ', '.join(f'`{" ".join(scope)}`' for scope in scopes) or 'none'
    refusal = (
        f'error: this workflow allows only Bash commands that start with {allowed}, alone or joined by `|`, `&&`, '
        "`||` or `;`. Redirection, substitution, subshells, background jobs, `$'...'` strings and multi-line commands "
        'are not allowed.'
    )

    async def bash_in_scope(ctx: RunContext[object], command: str, timeout: int | None = None) -> str:
        commands = None if '\n' in command or '\r' in command else _simple_commands(command)
        if commands is None or not all(
            any(tuple(words[: len(scope)]) == scope for scope in scopes) for words in commands
        ):
            return refusal
        return await bash(ctx, command, timeout)

    return bash_in_scope
