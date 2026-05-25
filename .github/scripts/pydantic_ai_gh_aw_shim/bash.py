"""Claude's `Bash` tool — run a shell command in the workspace."""

import subprocess

from .shared import clip, workspace


def bash(command: str, timeout: int | None = None) -> str:
    """Run a shell command in the repository workspace.

    Returns combined stdout+stderr (truncated). `timeout` is in seconds
    (default 120, capped at 600).
    """
    secs = 120 if not timeout or timeout <= 0 else min(int(timeout), 600)
    try:
        r = subprocess.run(command, shell=True, cwd=workspace(), capture_output=True, text=True, timeout=secs)
        return clip(f'exit={r.returncode}\n{r.stdout}{r.stderr}')
    except subprocess.TimeoutExpired:
        return f'error: command timed out after {secs}s'
