"""What `Knowledge` indexes: workspace paths, inline documents, and custom sources."""

from __future__ import annotations

import posixpath
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness._workspace import workspace_path

DEFAULT_FILE_SUFFIXES: frozenset[str] = frozenset(
    {'.md', '.mdx', '.markdown', '.rst', '.txt', '.csv', '.tsv', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.html'}
)
"""File suffixes read from a directory source by default: text formats whose raw text is worth embedding."""


@dataclass(frozen=True)
class KnowledgeDocument:
    """One document to index: inline in `Knowledge.sources`, or returned by a `KnowledgeSource`."""

    id: str
    """Identifies the document in the store and in search results. Unique across all sources."""

    text: str
    """The document's text, which is chunked and embedded."""


class KnowledgeSource(Protocol):
    """Produces documents to index, for content that is not a text file in a workspace.

    `load` is called on every sync, so `Knowledge` sees documents added, changed, and removed
    since the last one. Convert formats such as PDF or HTML to text here.
    """

    async def load(self, workspace: Workspace) -> Sequence[KnowledgeDocument]:
        """Return the documents this source currently holds.

        Args:
            workspace: The workspace path sources are read from: `Knowledge(workspace=...)`, or else the
                run's workspace, which is unattached when the agent has none.
        """
        ...


KnowledgeSourceLike: TypeAlias = str | Path | KnowledgeDocument | KnowledgeSource
"""An entry in `Knowledge.sources`: a workspace file or directory path, an inline document, or a custom source."""


def is_path_source(source: KnowledgeSourceLike) -> bool:
    return isinstance(source, str | Path)


async def load_sources(
    sources: Sequence[KnowledgeSourceLike], workspace: Workspace, suffixes: Collection[str]
) -> list[KnowledgeDocument]:
    """Every document `sources` produce, in order, failing on a missing path or an id produced twice."""
    documents: list[KnowledgeDocument] = []
    for source in sources:
        if isinstance(source, str | Path):
            documents.extend(await _load_path(workspace, source, suffixes))
        elif isinstance(source, KnowledgeDocument):
            documents.append(source)
        else:
            documents.extend(await source.load(workspace))
    seen: set[str] = set()
    for document in documents:
        if document.id in seen:
            raise UserError(f'`Knowledge` sources produce the document id {document.id!r} more than once.')
        seen.add(document.id)
    return documents


async def _load_path(workspace: Workspace, source: str | Path, suffixes: Collection[str]) -> list[KnowledgeDocument]:
    spelled = posixpath.normpath(workspace_path(Path(source)))
    path = await workspace.resolve(spelled)
    try:
        entry = await workspace.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        raise UserError(f'`Knowledge` source {spelled!r} does not exist in the workspace.') from None
    if not entry.is_dir:
        return [await _read(workspace, spelled, path)]
    documents: list[KnowledgeDocument] = []
    await _walk(workspace, spelled, path, suffixes, documents, visited=set())
    return documents


async def _walk(
    workspace: Workspace,
    spelled: str,
    path: str,
    suffixes: Collection[str],
    documents: list[KnowledgeDocument],
    *,
    visited: set[str],
) -> None:
    """Read every file below `path` with one of `suffixes`, skipping hidden entries and symlinked directory loops."""
    real = await workspace.realpath(path)
    if real in visited:
        return
    visited.add(real)
    # Sorted, so documents are produced in the same order for the same files.
    for entry in sorted(await workspace.list_dir(path), key=lambda entry: entry.name):
        if entry.name.startswith('.'):
            continue
        child = posixpath.join(spelled, entry.name) if spelled != '.' else entry.name
        if entry.is_dir:
            await _walk(workspace, child, entry.path, suffixes, documents, visited=visited)
        elif posixpath.splitext(entry.name)[1].lower() in suffixes:
            documents.append(await _read(workspace, child, entry.path))


async def _read(workspace: Workspace, document_id: str, path: str) -> KnowledgeDocument:
    return KnowledgeDocument(document_id, (await workspace.read_bytes(path)).decode('utf-8', errors='replace'))
