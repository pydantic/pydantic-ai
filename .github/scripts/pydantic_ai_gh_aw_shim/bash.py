"""Claude's `Bash` tool -- run a shell command in the workspace.

Backed by pydantic-ai-harness's `ShellToolset`, which handles subprocess
execution, the per-command timeout, and tail-keeping output truncation. The
Claude `Bash` signature (`command`, optional `timeout` in seconds) and the
sandbox PATH augmentation are preserved by the adapter.
"""

from pydantic_ai.exceptions import ModelRetry

from ._backends import BASH_DEFAULT_TIMEOUT, BASH_MAX_TIMEOUT, shell


async def bash(command: str, timeout: int | None = None) -> str:
    """Run a shell command in the repository workspace.

    Returns the command's labeled stdout/stderr (truncated). `timeout` is in
    seconds (default 120, capped at 600).
    """
    secs = BASH_DEFAULT_TIMEOUT if not timeout or timeout <= 0 else min(int(timeout), BASH_MAX_TIMEOUT)
    try:
        out = await shell().run_command(command, timeout_seconds=float(secs))
    except (ModelRetry, OSError) as exc:
        # ModelRetry: the harness blocked the command; the shim's tools have always
        # surfaced such conditions as a returned error string rather than a
        # model-facing retry. OSError: subprocess startup failed (e.g. the
        # workspace cwd does not exist) -- the harness doesn't convert it, so catch
        # it here rather than let it abort the whole run.
        return f'error: {exc}'
    # On timeout the harness *returns* a `[Command timed out after Ns]` sentinel
    # rather than raising. The old tool surfaced timeouts as an `error:` string
    # (and `Grep` already wraps the same sentinel), so do the same here instead
    # of handing the model an unprefixed result it might read as success.
    if out.startswith('[Command timed out'):
        return f'error: {out}'
    return out
