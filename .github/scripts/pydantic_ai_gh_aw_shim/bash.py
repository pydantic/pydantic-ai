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


def scoped_bash(scopes: BashScopes) -> Callable[[RunContext[object], str, int | None], Awaitable[str]]:
    """`bash` limited to commands whose every simple command starts with the tokens of one of `scopes`.

    Pipelines and `;`, `&&` and `||` lists are checked command by command. Whatever a prefix cannot scope is
    refused: newlines, redirection, background jobs, subshells, and command substitution.
    """
    allowed = ', '.join(f'`{" ".join(scope)}`' for scope in scopes) or 'none'
    refusal = (
        f'error: this workflow allows only Bash commands that start with {allowed}, alone or joined by `|`, `&&`, '
        '`||` or `;`. Redirection, substitution, subshells, background jobs and multi-line commands are not allowed.'
    )

    async def bash_in_scope(ctx: RunContext[object], command: str, timeout: int | None = None) -> str:
        if '\n' in command or '\r' in command:
            return refusal
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        # A `#` mid-word would end the parse, while the shell still runs what follows it.
        lexer.commenters = ''
        try:
            tokens = list(lexer)
        except ValueError:
            return refusal
        segment: list[str] = []
        for token in [*tokens, ';']:
            if token in _COMMAND_SEPARATORS:
                if not any(tuple(segment[: len(scope)]) == scope for scope in scopes):
                    return refusal
                segment = []
            elif (token and token[0] in '();<>|&') or '`' in token or '$(' in token:
                return refusal
            else:
                segment.append(token)
        return await bash(ctx, command, timeout)

    return bash_in_scope
