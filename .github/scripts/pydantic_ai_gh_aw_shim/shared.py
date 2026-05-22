"""Shared helpers used by the native tools in `pydantic_ai_gh_aw_shim/`.

Stdlib-only (apart from the package's own modules). Anything
pydantic-ai-specific belongs in the CLI module, not here — each tool
module should import only `shared` and be otherwise independent so it can
be swapped out cleanly.
"""

import contextvars
import logging
import os
import pathlib

MAX_TOOL_OUTPUT = 16000
"""Cap on any native tool's stringified result. Larger outputs are clipped with
a `…[truncated N chars]` suffix so the model knows it didn't see everything."""

CONTEXT_FILE_NAMES = ('AGENTS.md', 'CLAUDE.md')
MAX_CONTEXT_FILE_CHARS = 8000

# Configured for output by `cli.configure_logging()` (called from `main()`);
# at import time the logger only has whatever handlers the embedding
# application has set up. Library code never calls `logging.basicConfig`.
logger = logging.getLogger('pydantic_ai_gh_aw_shim')


def workspace() -> str:
    """Resolve the workspace root live.

    Reading `GITHUB_WORKSPACE` lazily
    (rather than capturing at import time) means tests can set it via
    `monkeypatch.setenv` instead of patching a module-level constant.
    """
    return os.environ.get('GITHUB_WORKSPACE') or os.getcwd()


def resolve(path: str) -> pathlib.Path:
    """Resolve a relative path against the current workspace; absolute paths pass through."""
    p = pathlib.Path(path)
    return p if p.is_absolute() else pathlib.Path(workspace()) / p


def clip(text: str) -> str:
    """Truncate a tool result string to `MAX_TOOL_OUTPUT` chars."""
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    return text[:MAX_TOOL_OUTPUT] + f'\n…[truncated {len(text) - MAX_TOOL_OUTPUT} chars]'


# --------------------------------------------------------------------------- #
# Directory-scoped AGENTS.md / CLAUDE.md auto-loading
# When the agent touches a file, the shim transparently surfaces the
# directory's `AGENTS.md` / `CLAUDE.md` the first time it's relevant (walking
# up to the workspace root). Per-run state ensures each file is shown at most
# once. Held in a `ContextVar` so each `run()` (and each test) gets its own
# set without test-ordering leaks, while `Task` sub-agents naturally inherit
# the parent's set via the async context tree (mutations to the shared object
# propagate both ways).
# --------------------------------------------------------------------------- #
_seen_context_files_var: contextvars.ContextVar[set[pathlib.Path] | None] = contextvars.ContextVar(
    '_seen_context_files', default=None
)


def _seen_context_files() -> set[pathlib.Path]:
    """Return the current run's dedupe set, installing one lazily if a tool ran outside `run()`."""
    s = _seen_context_files_var.get()
    if s is None:
        s = set[pathlib.Path]()
        _seen_context_files_var.set(s)
    return s


def reset_context_state() -> None:
    """Install a fresh dedupe set in the current context.

    Called once at the
    start of each `run()` (and from tests that want a clean slate).
    """
    _seen_context_files_var.set(set())


def attach_context(path_arg: str | None) -> str:
    """Return any newly-discovered `AGENTS.md` / `CLAUDE.md` blocks to prepend to a path-taking tool result.

    Walks up from the target's directory to the
    workspace root, surfacing each file at most once per run.

    """
    if not path_arg:
        return ''
    try:
        ws = pathlib.Path(workspace()).resolve()
        target = resolve(path_arg).resolve()
    except OSError:
        return ''
    seen = _seen_context_files()
    cur = target if target.is_dir() else target.parent
    blocks: list[str] = []
    while True:
        for name in CONTEXT_FILE_NAMES:
            candidate = cur / name
            if not candidate.is_file() or candidate in seen:
                continue
            seen.add(candidate)
            try:
                content = candidate.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            rel = candidate.relative_to(ws) if candidate.is_relative_to(ws) else candidate
            blocks.append(
                f'--- context: {rel} (auto-loaded; shown once per run) ---\n{content[:MAX_CONTEXT_FILE_CHARS]}\n'
            )
        if cur == ws or cur.parent == cur or not cur.is_relative_to(ws):
            break
        cur = cur.parent
    if not blocks:
        return ''
    return '\n'.join(blocks) + '\n'
