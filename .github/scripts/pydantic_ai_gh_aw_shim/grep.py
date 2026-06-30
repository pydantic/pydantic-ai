r"""Claude's `Grep` tool -- recursively regex-search workspace files.

Runs ripgrep through pydantic-ai-harness's `ShellToolset` -- the same shell
capability that backs `Bash` -- instead of a hand-rolled subprocess: the harness
owns process execution, the sandbox PATH, output truncation, and the timeout,
while ripgrep keeps its speed and `.gitignore` filtering (a poor fit for the
harness's own `FileSystemToolset.search_files`, which walks every non-dotfile
including vendored/ignored trees and matches with Python `re`). The
directory-scoped AGENTS.md / CLAUDE.md context blocks are still prepended.

Two adapters bridge a shell command back into a grep tool:

- Containment. `ShellToolset` roots the cwd at the workspace but, unlike the
  harness `FileSystemToolset`, does not validate the `path` argument, so a bare
  `rg -- ../..` could escape `$GITHUB_WORKSPACE`. `path` is preflighted through
  the filesystem capability -- the same containment `Read`/`Glob`/`LS` enforce.

- Framing. `run_command` frames output as `[stdout]` / `[stderr]` blocks and
  appends `\n[exit code: N]` only on a non-zero exit. ripgrep exits 1 on "no
  matches" (not an error) and 2+ on a real error, so the exit code -- not the
  presence of `[stdout]` -- is what the adapter keys off, unwrapping the generic
  framing back into a grep-shaped result.
"""

import re
import shlex

from pydantic_ai.exceptions import ModelRetry

from ._backends import filesystem, shell
from .shared import attach_context, clip

GREP_TIMEOUT = 60.0

# `run_command` appends this only on a non-zero exit; ripgrep's exit code rides
# at the very end of the framed output, after any tail-truncation of the body.
_EXIT_CODE_RE = re.compile(r'\n\[exit code: (\d+)\]\Z')
# The harness tail-truncates an oversized body and prepends this marker, which
# elides the leading `[stdout]` header; tolerate it so a large match set is
# never mistaken for an error.
_TRUNCATION_PREFIX = '[... output truncated'
# `search_files` (the harness's own content search) returns this on no match;
# grep mirrors it even though it's ripgrep-backed, so the two search tools agree.
_NO_MATCHES = 'No matches found.'


def _split_exit_code(out: str) -> tuple[str, int]:
    """Split a `run_command` result into its body and the command's exit code (0 if absent)."""
    match = _EXIT_CODE_RE.search(out)
    if match is None:
        return out, 0
    return out[: match.start()], int(match.group(1))


async def grep(pattern: str, path: str = '.') -> str:
    """Recursively regex-search workspace files via ripgrep, returning `file:line:text` matches."""
    # `file_info` accepts '' as the workspace root, but `rg -- ''` errors on the
    # empty path argument, so normalize to the default search root up front.
    path = path or '.'
    command = f'rg --line-number --no-heading --color never -e {shlex.quote(pattern)} -- {shlex.quote(path)}'
    # Preflight `path` through the filesystem containment check (an escape or a
    # missing path comes back as `ModelRetry`) before handing it to the shell.
    try:
        await filesystem().file_info(path)
        out = await shell().run_command(command, timeout_seconds=GREP_TIMEOUT)
    except ModelRetry as exc:
        return f'error: {exc}'
    if out.startswith('[Command timed out'):
        return f'error: {out}'
    body, exit_code = _split_exit_code(out)
    if exit_code >= 2:  # bad pattern, unreadable path, ripgrep absent (127), ...
        return f'error: {out}'
    if exit_code == 1:  # ripgrep's "nothing matched"
        return clip(attach_context(path) + _NO_MATCHES)
    # exit 0: matches. Strip the truncation marker first (it precedes and elides
    # the `[stdout]` header), then the header itself.
    if body.startswith(_TRUNCATION_PREFIX):
        body = body.split('\n', 1)[1] if '\n' in body else ''
    body = body.removeprefix('[stdout]\n')
    return clip(attach_context(path) + (body or _NO_MATCHES))
