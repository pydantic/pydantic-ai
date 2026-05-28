"""Claude's `Grep` tool — recursively regex-search workspace files."""

import shutil
import subprocess

from .shared import attach_context, clip, resolve


def grep(pattern: str, path: str = '.') -> str:
    """Recursively regex-search workspace files via ripgrep.

    Returns `file:line:text` matches (capped). The AWF sandbox image ships
    ripgrep; if it's unexpectedly absent the agent can fall back to the Bash
    tool with `grep -rn` on its own.
    """
    rg = shutil.which('rg')
    if not rg:
        return 'error: ripgrep (rg) not found — use the Bash tool with `grep -rn` instead'
    base = resolve(path)
    try:
        r = subprocess.run(
            [rg, '-n', '--no-heading', '-e', pattern, str(base)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return 'error: grep timed out'
    return clip(attach_context(path) + (r.stdout or '(no matches)'))
