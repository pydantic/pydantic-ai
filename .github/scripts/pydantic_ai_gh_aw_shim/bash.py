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
        return await shell().run_command(command, timeout_seconds=float(secs))
    except ModelRetry as exc:
        # The harness raises `ModelRetry` for a blocked command; the shim's tools
        # have always surfaced such conditions as a returned error string rather
        # than a model-facing retry, so preserve that.
        return f'error: {exc}'
