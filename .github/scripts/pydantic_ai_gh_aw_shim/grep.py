"""Claude's `Grep` tool — recursively regex-search workspace files."""

import shutil
import subprocess

from .shared import attach_context, clip, resolve


def grep(pattern: str, path: str = '.') -> str:
    """Recursively regex-search workspace files via ripgrep.

    Returns `file:line:text` matches (capped). gh-aw's runners and the AWF
    sandbox image both ship ripgrep; if it's ever missing in some deployment
    the agent can fall back to `Bash` (`grep -rn …`) on its own — a Python
    re-implementation is slow and not a real substitute.
    """
    rg = shutil.which('rg')
    if not rg:
        return 'error: ripgrep (rg) not installed — use the Bash tool with `grep -rn` instead'
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
