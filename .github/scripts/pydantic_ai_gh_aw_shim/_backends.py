"""Harness-backed toolsets that execute the Claude file/shell tools.

The Claude-named tool callables in this package (`Bash`, `Read`, `Write`,
`Edit`) keep their Claude Code signatures but delegate the actual work to
pydantic-ai-harness's `ShellToolset` and `FileSystemToolset`. The harness owns
the parts that were previously hand-rolled here: subprocess execution and
output truncation, path containment, symlink resolution before access, and
binary-file detection. The shim keeps only the thin signature adapters.

The toolsets are built per call so `workspace()` (and a test's
`GITHUB_WORKSPACE`) is read live; construction is cheap. They are used by
calling their methods directly, not by registering them on an agent, so the
agent still sees exactly the Claude tool surface gh-aw expects.
"""

import os
from pathlib import Path

from pydantic_ai_harness.filesystem import FileSystemToolset
from pydantic_ai_harness.shell import ShellToolset

from .shared import MAX_TOOL_OUTPUT, workspace

# Standard Unix binary locations prepended to PATH so `rg`, `make`, `git`, and
# `uv` are reachable even when the AWF sandbox starts with a minimal inherited
# PATH. Moved here from the old hand-rolled `bash` tool.
_STANDARD_PATHS = [
    '/opt/hostedtoolcache/gh-aw-tools/current/x64/bin',  # rg + uv (install-sandbox-tools.sh)
    '/tmp/gh-aw/bin',  # fallback; launcher lives here too
    '/usr/local/bin',
    '/usr/bin',
    '/bin',
    '/usr/local/sbin',
    '/usr/sbin',
    '/sbin',
]

# Claude's `Bash` timeout contract: default 120s, hard-capped at 600s.
BASH_DEFAULT_TIMEOUT = 120
BASH_MAX_TIMEOUT = 600


def augmented_env() -> dict[str, str]:
    """The process environment with the standard tool paths prepended to PATH."""
    env = dict(os.environ)
    current = env.get('PATH', '')
    existing = set(current.split(':'))
    extra = ':'.join(p for p in _STANDARD_PATHS if p not in existing)
    env['PATH'] = f'{extra}:{current}' if extra else current
    return env


def filesystem() -> FileSystemToolset[None]:
    """`FileSystemToolset` rooted at the live workspace.

    `protected_patterns=[]` keeps the prior behavior of allowing writes anywhere
    under the workspace. The harness still enforces containment (no path escapes
    the workspace root) and resolves symlinks before access -- a change from the
    old tools, which resolved any absolute path. For gh-aw the agent operates
    within `$GITHUB_WORKSPACE`, so containment to that root is the intended scope.
    """
    return FileSystemToolset[None](
        root_dir=Path(workspace()),
        allowed_patterns=[],
        denied_patterns=[],
        protected_patterns=[],
        max_read_lines=2000,
        max_search_results=1000,
        max_find_results=1000,
    )


def shell() -> ShellToolset[None]:
    """`ShellToolset` rooted at the live workspace, PATH augmented for the sandbox.

    Command/operator denylists are left empty to preserve the old `Bash` tool's
    "run anything" contract; the AWF sandbox is the security boundary. Output is
    capped at `MAX_TOOL_OUTPUT`, keeping the tail (where errors and exit info land).
    """
    return ShellToolset[None](
        cwd=Path(workspace()),
        allowed_commands=[],
        denied_commands=[],
        denied_operators=[],
        default_timeout=float(BASH_DEFAULT_TIMEOUT),
        max_output_chars=MAX_TOOL_OUTPUT,
        persist_cwd=False,
        allow_interactive=True,
        env=augmented_env(),
        denied_env_patterns=[],
    )
