"""The types a knowledge base backend works with, and the protocols it implements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from pydantic_ai.native_tools import FileSearchTool
    from pydantic_ai.tools import RunContext

FilterScalar: TypeAlias = str | int | float | bool
"""A metadata value a filter compares with."""

MetadataFilter: TypeAlias = Mapping[str, FilterScalar | Sequence[str | int | float]]
"""A backend-neutral metadata filter: every key must match.

A scalar value matches a passage whose metadata attribute equals it; a list matches one equal to any of
its values. `{'tenant_id': 'acme', 'source': ['manuals', 'faq']}` scopes a search to one tenant's
manuals and FAQ. Each backend translates it to its own filter syntax.
"""


def matches(metadata: Mapping[str, object], filter: MetadataFilter | None) -> bool:
    """Whether `metadata` satisfies every key of `filter`: equal to a scalar, or equal to one value of a list."""
    if not filter:
        return True
    for key, expected in filter.items():
        if key not in metadata:
            return False
        actual = metadata[key]
        if isinstance(expected, str | int | float | bool):
            if actual != expected:
                return False
        elif actual not in expected:
            return False
    return True


@dataclass(frozen=True)
class Passage:
    """One passage a search returned."""

    text: str
    """The passage text the model sees."""

    source: str | None = None
    """Where the passage came from, such as a URI, URL or file name; `None` when the backend reports none."""

    score: float | None = None
    """The relevance score the backend assigned, on the backend's own scale."""

    document_id: str | None = None
    """The id of the document the passage belongs to, which `read` takes; `None` when the backend can't read documents."""

    metadata: Mapping[str, object] = field(default_factory=dict[str, object])
    """The passage's metadata attributes."""


@dataclass(frozen=True)
class Document:
    """A whole document: what `InMemoryKnowledgeBase` stores, and what `read` returns."""

    document_id: str
    """The document's id, unique within the knowledge base."""

    text: str
    """The document text."""

    source: str | None = None
    """Where the document came from, shown with its passages."""

    metadata: Mapping[str, FilterScalar] = field(default_factory=dict[str, FilterScalar])
    """Metadata attributes a `MetadataFilter` matches against."""


class KnowledgeBaseResult(TypedDict):
    """One passage, carried in `ToolReturn.metadata['results']` for applications that render citations."""

    source: str | None
    score: float | None
    document_id: str | None
    metadata: dict[str, object]


@runtime_checkable
class KnowledgeBaseBackend(Protocol):
    """Searches a knowledge base. The one thing every backend does.

    A backend owns its transport and its failure policy: raise `ModelRetry` for a failure the model can
    recover from by searching again (throttling, a timeout), and let anything else propagate.
    """

    async def search(self, query: str, *, limit: int, filter: MetadataFilter | None) -> Sequence[Passage]:
        """Return up to `limit` passages relevant to `query`, best first, matching `filter` when one is given."""
        ...


@runtime_checkable
class ReadableKnowledgeBase(Protocol):
    """A backend that can also return a whole document by id, which adds the `read_document` tool."""

    async def read(self, document_id: str, *, max_chars: int, filter: MetadataFilter | None) -> Document | None:
        """Return the document, or `None` when there is none with that id or it doesn't match `filter`.

        `filter` is the run's search filter. Applying it here is what keeps a tenant-scoped agent from
        opening another tenant's document by id, so a backend must return `None` for a document whose
        metadata doesn't match. The text may be cut after `max_chars` characters (one more lets the
        caller see that it was cut).
        """
        ...


@runtime_checkable
class NativeFileSearchBackend(Protocol):
    """A backend whose index the model provider can also search natively, with `FileSearchTool`."""

    def file_search_tool(self, ctx: RunContext[object], *, limit: int) -> FileSearchTool | None:
        """The native tool that searches this index for the run's model, or `None` where the model can't."""
        ...
