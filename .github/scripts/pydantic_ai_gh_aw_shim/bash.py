"""Claude's `Bash` tool — run a shell command in the workspace."""

import os
import subprocess

from .shared import clip, workspace

# Standard Unix binary locations prepended to whatever PATH the process
# inherits so tools like `rg`, `make`, `git`, and `uv` are reliably reachable
# even if the AWF sandbox launches with a minimal inherited PATH.
_STANDARD_PATHS = [
    '/opt/hostedtoolcache/gh-aw-tools/current/x64/bin',  # rg + uv installed by install-sandbox-tools.sh
    '/tmp/gh-aw/bin',  # fallback; launcher lives here too
    '/usr/local/bin',
    '/usr/bin',
    '/bin',
    '/usr/local/sbin',
    '/usr/sbin',
    '/sbin',
]


def _augmented_env() -> dict[str, str]:
    env = dict(os.environ)
    current = env.get('PATH', '')
    existing = set(current.split(':'))
    extra = ':'.join(p for p in _STANDARD_PATHS if p not in existing)
    env['PATH'] = f'{extra}:{current}' if extra else current
    return env


def bash(command: str, timeout: int | None = None) -> str:
    """Run a shell command in the repository workspace.

    Returns combined stdout+stderr (truncated). `timeout` is in seconds
    (default 120, capped at 600).
    """
    secs = 120 if not timeout or timeout <= 0 else min(int(timeout), 600)
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=workspace(),
            capture_output=True,
            text=True,
            timeout=secs,
            env=_augmented_env(),
        )
        return clip(f'exit={r.returncode}\n{r.stdout}{r.stderr}')
    except subprocess.TimeoutExpired:
        return f'error: command timed out after {secs}s'
