"""Walk-up discovery, dedup, precedence, and rendering of instruction files."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.workspaces import Workspace


@dataclass(frozen=True)
class ContextFile:
    """A single instruction file discovered during walk-up."""

    directory: Path
    """The directory the file was found in."""

    path: Path
    """The file's path."""

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


async def _is_file(workspace: Workspace, path: Path) -> bool:
    """True when `path` exists and is a regular file inside `workspace`."""
    text = str(path)
    try:
        entry = await workspace.stat(text)
    except (FileNotFoundError, NotADirectoryError):
        return False
    if entry.is_dir:
        return False
    # Autoloaded instructions belong to the directory being scanned, not a
    # symlink target in another directory (even another ancestor in the walk).
    return Path(await workspace.realpath(text)).parent == Path(await workspace.realpath(str(path.parent)))


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

    Files are deduped by the path the walker visits and by content hash, so
    two names for the same content within one directory load once.
    """
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    found: list[ContextFile] = []
    for directory in _walk_dirs(workspace_dir, home_dir):
        for filename in filenames:
            candidate = directory / filename
            if not await _is_file(workspace, candidate):
                continue
            if candidate in seen_paths:
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
        candidate = directory / filename
        if await _is_file(workspace, candidate):
            return ContextFile(
                directory=directory,
                path=candidate,
                content=(await workspace.read_bytes(str(candidate))).decode('utf-8', errors='replace'),
            )
    return None


def render_context_file(file: ContextFile, *, label: str) -> str:
    """Render one file as a labeled block.

    The closing tag sits on its own line, so the content's trailing line terminators go: keeping them
    would put a blank line before the tag for every file that ends the way a text file should. Only the
    terminators -- trailing spaces and tabs stay, being a hard line break in Markdown.
    """
    content = file.content.rstrip('\r\n')
    return f'<context-file path="{label}">\n{content}\n</context-file>'


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
