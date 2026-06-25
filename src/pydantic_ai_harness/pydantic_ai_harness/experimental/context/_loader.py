"""Walk-up discovery, dedup, precedence, and rendering of instruction files."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


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
    scanned.
    """
    workspace = workspace_dir.resolve()
    if home_dir is None:
        return [workspace]
    home = home_dir.resolve()
    chain: list[Path] = [workspace]
    if workspace != home:
        for parent in workspace.parents:
            chain.append(parent)
            if parent == home:
                break
        else:
            return [workspace]
    return list(reversed(chain))


def discover_instruction_files(
    workspace_dir: Path,
    home_dir: Path | None,
    filenames: Sequence[str],
) -> list[ContextFile]:
    """Collect instruction files from `home_dir` down to `workspace_dir`.

    Precedence is ancestor-first, workspace-last: the broadest context comes
    first and the most specific (closest to the model's recency window) comes
    last. Within a directory, `filenames` are tried in order.

    Files are deduped by resolved real path and by content hash, so a symlinked
    `AGENTS.md -> CLAUDE.md` or two ancestors sharing identical content load
    once. The first occurrence in precedence order wins.
    """
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    found: list[ContextFile] = []
    for directory in _walk_dirs(workspace_dir, home_dir):
        for filename in filenames:
            candidate = directory / filename
            if not candidate.is_file():
                continue
            real = candidate.resolve()
            if real in seen_paths:
                continue
            content = candidate.read_text(encoding='utf-8', errors='replace')
            digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
            if digest in seen_hashes:
                continue
            seen_paths.add(real)
            seen_hashes.add(digest)
            found.append(ContextFile(directory=directory, path=candidate, content=content))
    return found


def find_dir_context_file(directory: Path, filenames: Sequence[str]) -> ContextFile | None:
    """Return the first existing instruction file in `directory`, or `None`."""
    for filename in filenames:
        candidate = directory / filename
        if candidate.is_file():
            return ContextFile(
                directory=directory,
                path=candidate,
                content=candidate.read_text(encoding='utf-8', errors='replace'),
            )
    return None


def render_context_file(file: ContextFile, *, label: str) -> str:
    """Render one file as a labeled block."""
    return f'<context-file path="{label}">\n{file.content}\n</context-file>'


def render_context_files(files: Sequence[ContextFile], *, relative_to: Path) -> str:
    """Render discovered files as labeled blocks in precedence order."""
    blocks = [render_context_file(file, label=_label(file.path, relative_to)) for file in files]
    return '\n\n'.join(blocks)


def _label(path: Path, relative_to: Path) -> str:
    """A stable display label: relative to `relative_to` when possible."""
    try:
        return path.resolve().relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
