"""Claude's `Bash` tool -- run a shell command in the workspace.

Backed by pydantic-ai-harness's `ShellToolset`, which runs the command in the
run's workspace with the per-command timeout. The Claude `Bash` signature
(`command`, optional `timeout` in seconds), the sandbox PATH augmentation, and
the tail-keeping output cap are preserved by the adapter.
"""

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import WorkspaceError

from ._backends import BASH_DEFAULT_TIMEOUT, BASH_MAX_TIMEOUT, run_shell, timed_out


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
