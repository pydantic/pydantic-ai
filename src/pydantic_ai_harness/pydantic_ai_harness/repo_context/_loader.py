"""Walk-up discovery, dedup, precedence, and rendering of instruction files."""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai._framing import neutralize_tags
from pydantic_ai.workspaces import Workspace

# TODO: switch to `wrap()` from the public `pydantic_ai.framing` module once it lands (#10957), registering
# `context-file` as a framing tag there.
# The `<` of an opening or closing `context-file` tag, in any case, with whitespace around the `/`, carrying
# attributes, or cut off at the end of the content, where the frame's own end tag would complete it; a longer tag
# name is left alone. Core's own framing tags are left to `neutralize_tags`.
_CONTEXT_FILE_TAG_OPENER = re.compile(r'<(?=\s*(?:/\s*)?context-file(?=[\s/>]|$))', re.IGNORECASE)


@dataclass(frozen=True)
class ContextFile:
    """A single instruction file discovered during walk-up or nested traversal."""

    directory: Path
    """The directory the file was found in."""

    path: Path
    """The file's path, named after its target when it is a symlink to a file in the same directory."""

    content: str
    """The file's text content."""


def _walk_dirs(workspace_dir: Path, home_dir: Path | None) -> list[Path]:
    """Directories to scan, ancestor-first (home first, workspace last).

    Walk up from `workspace_dir` to `home_dir` inclusive. When `home_dir` is
    `None`, or is not an ancestor of `workspace_dir`, only `workspace_dir` is
    scanned. The workspace backend handles filesystem path behavior.
    """
    if home_dir is None:
        return [workspace_dir]
    chain: list[Path] = [workspace_dir]
    if workspace_dir != home_dir:
        for parent in workspace_dir.parents:
            chain.append(parent)
            if parent == home_dir:
                break
        else:
            return [workspace_dir]
    return list(reversed(chain))


async def _real_file(workspace: Workspace, path: Path) -> Path | None:
    """The file `path` names, after symlinks, when it is a regular file in `path`'s own directory, else `None`.

    A link such as `CLAUDE.md -> AGENTS.md` returns `AGENTS.md` in the same directory, so the file is
    labeled and deduplicated by the name it is kept under.
    """
    text = str(path)
    try:
        entry = await workspace.stat(text)
    except (FileNotFoundError, NotADirectoryError):
        return None
    if entry.is_dir:
        return None
    real = Path(await workspace.realpath(text))
    # Autoloaded instructions belong to the directory being scanned, not a
    # symlink target in another directory (even another ancestor in the walk).
    if real.parent != Path(await workspace.realpath(str(path.parent))):
        return None
    return path.parent / real.name


async def discover_instruction_files(
    workspace: Workspace,
    workspace_dir: Path,
    home_dir: Path | None,
    filenames: Sequence[str],
) -> list[ContextFile]:
    """Collect instruction files from `home_dir` down to `workspace_dir`.

    Precedence is ancestor-first, workspace-last: the broadest context comes
    first and the most specific (closest to the model's recency window) comes
    last. Within a directory, `filenames` are tried in order.

    Files are deduped by the path a symlink resolves to and by content hash, so
    two names for the same content within one directory load once, under the
    name of the file a symlink points to.
    """
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    found: list[ContextFile] = []
    for directory in _walk_dirs(workspace_dir, home_dir):
        for filename in filenames:
            candidate = await _real_file(workspace, directory / filename)
            if candidate is None or candidate in seen_paths:
                continue
            content = (await workspace.read_bytes(str(candidate))).decode('utf-8', errors='replace')
            digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
            if digest in seen_hashes:
                continue
            seen_paths.add(candidate)
            seen_hashes.add(digest)
            found.append(ContextFile(directory=directory, path=candidate, content=content))
    return found


async def find_dir_context_file(workspace: Workspace, directory: Path, filenames: Sequence[str]) -> ContextFile | None:
    """Return the first existing instruction file in `directory`, or `None`."""
    for filename in filenames:
        candidate = await _real_file(workspace, directory / filename)
        if candidate is not None:
            return ContextFile(
                directory=directory,
                path=candidate,
                content=(await workspace.read_bytes(str(candidate))).decode('utf-8', errors='replace'),
            )
    return None


def render_context_file(file: ContextFile, *, label: str) -> str:
    """Render one file as a labeled block.

    The file and its path come from the repository, so neither can end the block: the `<` of every
    opening or closing `context-file` tag and of every framing tag core writes (`<system>`, `<context>`,
    `<tool_result>` and the rest) is escaped as `&lt;`, and the label is attribute-escaped. Everything
    else goes in as written.

    The closing tag sits on its own line, so the content's trailing line terminators go: keeping them
    would put a blank line before the tag for every file that ends the way a text file should. Only the
    terminators -- trailing spaces and tabs stay, being a hard line break in Markdown.
    """
    content = _CONTEXT_FILE_TAG_OPENER.sub('&lt;', neutralize_tags(file.content.rstrip('\r\n')))
    return f'<context-file path="{html.escape(label)}">\n{content}\n</context-file>'


def render_context_file_path(label: str) -> str:
    """Name a file without its contents, as an empty `<context-file>` tag with the label attribute-escaped."""
    return f'<context-file path="{html.escape(label)}" />'


def render_context_files(files: Sequence[ContextFile], *, relative_to: Path) -> str:
    """Render discovered files as labeled blocks in precedence order."""
    blocks = [render_context_file(file, label=_label(file.path, relative_to)) for file in files]
    return '\n\n'.join(blocks)


def _label(path: Path, relative_to: Path) -> str:
    """A stable display label: relative to `relative_to` when possible."""
    try:
        return path.relative_to(relative_to).as_posix()
    except ValueError:
        return path.as_posix()
