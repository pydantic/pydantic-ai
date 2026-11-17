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
from pydantic_ai_harness._workspace import stat_or_none

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


def directories_up_to(directory: Path, top: Path | None) -> list[Path]:
    """`directory` and each of its ancestors up to `top` inclusive, ancestor-first.

    When `top` is `None`, or is not an ancestor of `directory`, only `directory` is
    returned. The walk-up scans these with `top` as the home directory, and nested
    traversal with `top` as the working directory. The workspace backend handles
    filesystem path behavior.
    """
    if top is None:
        return [directory]
    chain: list[Path] = [directory]
    if directory != top:
        for parent in directory.parents:
            chain.append(parent)
            if parent == top:
                break
        else:
            return [directory]
    return list(reversed(chain))


async def find_root(workspace: Workspace, working_dir: Path, markers: Sequence[str]) -> Path | None:
    """The nearest of `working_dir` and its ancestors that contains an entry named in `markers`, or `None`.

    An entry of any kind counts, so a linked worktree's `.git` file marks its root like a `.git` directory.
    """
    for directory in (working_dir, *working_dir.parents):
        for marker in markers:
            if await workspace.exists(str(directory / marker)):
                return directory
    return None


_INSTRUCTION_NAMES = frozenset({'AGENTS.md', 'CLAUDE.md'})
"""Names a symlinked instruction file may point to whatever `filenames` holds, so `CLAUDE.md -> AGENTS.md` keeps
loading when only `CLAUDE.md` is configured."""


async def _real_file(
    workspace: Workspace, path: Path, filenames: Sequence[str], *, autoload_symlink_roots: Sequence[Path] = ()
) -> tuple[Path, Path] | None:
    """Return the resolved read/dedup target and display path of an authorized instruction file.

    A same-directory link such as `CLAUDE.md -> AGENTS.md` is labeled by its target's name. Its target must
    itself be an instruction file, named in `filenames` or one of `AGENTS.md` and `CLAUDE.md`: a repository's
    `AGENTS.md -> .env` would otherwise send a file the file tools may deny to the model.
    """
    text = str(path)
    entry = await stat_or_none(workspace, text)
    if entry is None or entry.is_dir:
        return None
    real = Path(await workspace.realpath(text))
    if real.parent == Path(await workspace.realpath(str(path.parent))):
        if real.name != path.name and real.name not in filenames and real.name not in _INSTRUCTION_NAMES:
            return None
        return real, path.parent / real.name
    # External targets require explicit authorization; their entry directory still defines their scope.
    if any(real.is_relative_to(root) for root in autoload_symlink_roots):
        return real, path
    return None


async def discover_instruction_files(
    workspace: Workspace,
    workspace_dir: Path,
    home_dir: Path | None,
    filenames: Sequence[str],
    *,
    autoload_symlink_roots: Sequence[Path] = (),
) -> list[ContextFile]:
    """Collect one instruction file per directory from `home_dir` down to `workspace_dir`.

    Precedence is ancestor-first, workspace-last: the broadest context comes first and the
    most specific (closest to the model's recency window) comes last. Each directory contributes the first of
    `filenames` that exists there and is an accepted file or link.

    Files are deduped by resolved target and content hash, so an ancestor file with the same content as an
    earlier one loads once. Same-directory links use the target's name. External targets must be beneath a
    resolved `autoload_symlink_roots` path and keep the scanned entry's name.
    """
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    found: list[ContextFile] = []
    for directory in directories_up_to(workspace_dir, home_dir):
        candidate = await _dir_context_file(workspace, directory, filenames, autoload_symlink_roots)
        if candidate is None:
            continue
        target, label = candidate
        if target in seen_paths:
            continue
        content = (await workspace.read_bytes(str(target))).decode('utf-8', errors='replace')
        digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
        if digest in seen_hashes:
            continue
        seen_paths.add(target)
        seen_hashes.add(digest)
        found.append(ContextFile(directory=directory, path=label, content=content))
    return found


async def _dir_context_file(
    workspace: Workspace, directory: Path, filenames: Sequence[str], autoload_symlink_roots: Sequence[Path] = ()
) -> tuple[Path, Path] | None:
    """The resolved target and label of the first of `filenames` in `directory` that `_real_file` accepts."""
    for filename in filenames:
        candidate = await _real_file(
            workspace, directory / filename, filenames, autoload_symlink_roots=autoload_symlink_roots
        )
        if candidate is not None:
            return candidate
    return None


async def dir_context_file_path(workspace: Workspace, directory: Path, filenames: Sequence[str]) -> Path | None:
    """Return the path of the first existing instruction file in `directory`, without reading it."""
    candidate = await _dir_context_file(workspace, directory, filenames)
    return None if candidate is None else candidate[1]


async def find_dir_context_file(workspace: Workspace, directory: Path, filenames: Sequence[str]) -> ContextFile | None:
    """Return the first existing instruction file in `directory`, or `None`."""
    path = await dir_context_file_path(workspace, directory, filenames)
    if path is None:
        return None
    content = (await workspace.read_bytes(str(path))).decode('utf-8', errors='replace')
    return ContextFile(directory=directory, path=path, content=content)


def render_context_file(file: ContextFile, *, label: str) -> str:
    """Render one file as a labeled block.

    The file and its path come from the repository, so neither can end the block: the `<` of every
    opening or closing `context-file` tag and of every framing tag core writes (`<system>`, `<pyai:context>`,
    `<pyai:tool_result>` and the rest) is escaped as `&lt;`, and the label is attribute-escaped. Everything
    else goes in as written.

    The closing tag sits on its own line, so the content's trailing line terminators go: keeping them
    would put a blank line before the tag for every file that ends the way a text file should. Only the
    terminators -- trailing spaces and tabs stay, being a hard line break in Markdown.
    """
    content = _CONTEXT_FILE_TAG_OPENER.sub('&lt;', neutralize_tags(file.content.rstrip('\r\n')))
    return f'<context-file path="{html.escape(label)}">\n{content}\n</context-file>'


def context_file_part_content(file: ContextFile) -> str:
    """A nested file's text as a `ContextFilePart` holds it: as written, without trailing line terminators.

    Pydantic AI sends a data part's content byte for byte: text in it that opens or closes the `<context_file>` frame, or
    looks like one of the application's frames, gets a note saying the file's author wrote it.
    """
    return file.content.rstrip('\r\n')


def render_context_files(files: Sequence[ContextFile], *, relative_to: Path) -> str:
    """Render discovered files as labeled blocks in precedence order."""
    blocks = [render_context_file(file, label=context_file_label(file.path, relative_to)) for file in files]
    return '\n\n'.join(blocks)


def context_file_label(path: Path, relative_to: Path) -> str:
    """A stable display label: relative to `relative_to`, with `..` steps for a file in one of its ancestors.

    An ancestor's file is labeled `../AGENTS.md` rather than by its absolute path, so the label doesn't
    carry the host's directory layout into the instructions or vary between machines.
    """
    if path.is_relative_to(relative_to):
        return path.relative_to(relative_to).as_posix()
    if relative_to.is_relative_to(path.parent):
        return '../' * len(relative_to.relative_to(path.parent).parts) + path.name
    return path.as_posix()
