"""The types a knowledge base backend works with, the filter language, and the protocols a backend implements."""

from __future__ import annotations

import operator as operator_module
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from pydantic_ai.native_tools import FileSearchTool
    from pydantic_ai.tools import RunContext

FilterScalar: TypeAlias = str | int | float | bool
"""A metadata value a filter compares with."""

FilterValue: TypeAlias = FilterScalar | Sequence[str | int | float]
"""What a comparison compares with: a scalar, or a list for `in` and `not in`."""

ComparisonOperator: TypeAlias = Literal['==', '!=', '>', '>=', '<', '<=', 'in', 'not in']
"""How a `FilterCondition` compares a metadata field with its value."""

LogicalOperator: TypeAlias = Literal['AND', 'OR', 'NOT']
"""How a `LogicalFilter` combines its conditions. `NOT` negates the AND of its conditions."""

SearchMode: TypeAlias = Literal['semantic', 'keyword', 'hybrid']
"""How a search matches a query: by meaning (embeddings), by words, or both. Backends support a subset."""


class FilterCondition(TypedDict):
    """One comparison: `{'field': 'year', 'operator': '>=', 'value': 2024}`."""

    field: str
    operator: ComparisonOperator
    value: FilterValue


class LogicalFilter(TypedDict):
    """Conditions combined: `{'operator': 'OR', 'conditions': [...]}`. Conditions may nest."""

    operator: LogicalOperator
    conditions: Sequence[FilterCondition | LogicalFilter]


FilterNode: TypeAlias = FilterCondition | LogicalFilter
"""A filter tree: one comparison, or comparisons combined with `AND`, `OR` and `NOT`."""

MetadataFilter: TypeAlias = FilterNode | Mapping[str, FilterValue]
"""A backend-neutral metadata filter, as a tree or as a shorthand mapping.

The tree is the full form, the one Haystack and LlamaIndex use: a `FilterCondition` compares a
field (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not in`), and a `LogicalFilter` combines conditions
with `AND`, `OR` or `NOT`. The shorthand `{'tenant_id': 'acme', 'source': ['manuals', 'faq']}` means
"every key matches": a scalar is `==`, a list is `in`. A document without the field matches the
negative comparisons (`!=`, `not in`) and no other, which is how Amazon Bedrock evaluates them too.
Each backend translates the filter to its own syntax and raises `UserError` for parts it can't apply.
"""

_COMPARISONS: frozenset[str] = frozenset({'==', '!=', '>', '>=', '<', '<=', 'in', 'not in'})
_LIST_COMPARISONS: frozenset[str] = frozenset({'in', 'not in'})
_LOGICAL: frozenset[str] = frozenset({'AND', 'OR', 'NOT'})
_ORDERINGS = {'>': operator_module.gt, '>=': operator_module.ge, '<': operator_module.lt, '<=': operator_module.le}


def _is_scalar(value: object) -> bool:
    return isinstance(value, str | int | float | bool)


def normalize_filter(filter: MetadataFilter | None) -> FilterNode | None:
    """Turn a filter into its tree form, checking its shape; `None` or an empty mapping means no filter.

    Raises:
        ValueError: The filter has an unknown operator, a list where a scalar belongs (or the other
            way round), or a logical node without conditions.
    """
    if not filter:
        return None
    if _is_node(filter):
        return _checked(filter)
    leaves: list[FilterNode] = [
        _checked({'field': key, 'operator': '==' if _is_scalar(value) else 'in', 'value': value})
        for key, value in filter.items()
    ]
    return leaves[0] if len(leaves) == 1 else {'operator': 'AND', 'conditions': leaves}


def _is_node(filter: Mapping[str, object]) -> bool:
    keys = set(filter)
    return keys == {'field', 'operator', 'value'} or keys == {'operator', 'conditions'}


def _checked(node: Mapping[str, object]) -> FilterNode:
    """Validate one node and its children, returning it typed as the tree it is."""
    operator = node.get('operator')
    if 'conditions' in node:
        conditions = node['conditions']
        if operator not in _LOGICAL:
            raise ValueError(f'Unknown logical operator {operator!r}; use one of AND, OR, NOT')
        if isinstance(conditions, str | bytes) or not isinstance(conditions, Sequence) or not conditions:
            raise ValueError(f'{operator} needs a non-empty list of conditions')
        children: list[FilterNode] = []
        for child in conditions:  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(child, Mapping) or not _is_node(child):  # pyright: ignore[reportUnknownArgumentType]
                raise ValueError(f'Not a filter condition: {child!r}')
            children.append(_checked(child))  # pyright: ignore[reportUnknownArgumentType]
        return {'operator': operator, 'conditions': children}  # pyright: ignore[reportReturnType]
    field_name, value = node.get('field'), node.get('value')
    if not isinstance(field_name, str):
        raise ValueError(f'A condition needs a field name, got {field_name!r}')
    if operator not in _COMPARISONS:
        raise ValueError(f'Unknown comparison operator {operator!r}; use one of {", ".join(sorted(_COMPARISONS))}')
    if operator in _LIST_COMPARISONS:
        if _is_scalar(value) or not isinstance(value, Sequence):
            raise ValueError(f'{operator!r} on {field_name!r} needs a list of values')
        value = list(value)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    elif not _is_scalar(value):
        raise ValueError(f'{operator!r} on {field_name!r} needs a single value, not {value!r}')
    return {'field': field_name, 'operator': operator, 'value': value}  # pyright: ignore[reportReturnType, reportUnknownVariableType]


def matches(metadata: Mapping[str, object], filter: MetadataFilter | None) -> bool:
    """Whether `metadata` satisfies `filter`. A missing field matches `!=` and `not in`, and no other comparison."""
    node = normalize_filter(filter)
    return node is None or _evaluate(metadata, node)


def allows_read(metadata: Mapping[str, object], filter: FilterNode | None) -> bool:
    """Whether a document with `metadata` may be read under `filter`, failing closed.

    Stricter than `matches`: every field the filter names must be present, so a negative condition
    (`!=`, `not in`) can't let a document through only because it lacks the field, or because its
    metadata couldn't be read. Backends use it in `read`, where the model names a document directly.
    """
    return filter is None or (_fields(filter) <= metadata.keys() and _evaluate(metadata, filter))


def _fields(node: FilterNode) -> set[str]:
    if 'conditions' in node:
        return set[str]().union(*(_fields(child) for child in node['conditions']))
    return {node['field']}


def _evaluate(metadata: Mapping[str, object], node: FilterNode) -> bool:
    if 'conditions' in node:
        results = [_evaluate(metadata, child) for child in node['conditions']]
        if node['operator'] == 'AND':
            return all(results)
        if node['operator'] == 'OR':
            return any(results)
        return not all(results)
    if node['field'] not in metadata:
        return node['operator'] in ('!=', 'not in')
    actual, operator, expected = metadata[node['field']], node['operator'], node['value']
    if operator == '==':
        return actual == expected
    if operator == '!=':
        return actual != expected
    if operator in _LIST_COMPARISONS:
        assert not isinstance(expected, str) and isinstance(expected, Sequence)
        return (actual in expected) == (operator == 'in')
    try:
        # Metadata values are untyped, so a comparison of, say, a string with a number fails as no match.
        return bool(_ORDERINGS[operator](actual, expected))  # pyright: ignore[reportArgumentType]
    except TypeError:
        return False


@dataclass(frozen=True)
class Passage:
    """One passage a search returned."""

    text: str
    """The passage text the model sees."""

    source: str | None = None
    """Where the passage came from, such as a URI, URL or file name; `None` when the backend reports none."""

    score: float | None = None
    """The relevance score the backend assigned, on the backend's own scale: not comparable across backends."""

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
    """Metadata attributes a filter matches against."""


class KnowledgeBaseResult(TypedDict):
    """One passage, carried in `ToolReturn.metadata['results']` for applications that render citations."""

    source: str | None
    score: float | None
    document_id: str | None
    metadata: dict[str, object]


@runtime_checkable
class KnowledgeBaseBackend(Protocol):
    """Searches a knowledge base. The one thing every backend does.

    A backend translates the filter tree and the search options to its service and raises `UserError`
    for what it can't apply, rather than searching without it. It owns its transport and failure
    policy: raise `ModelRetry` for a failure the model can recover from by searching again
    (throttling, a timeout), and let anything else propagate.
    """

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filter: FilterNode | None,
        mode: SearchMode | None = None,
        score_threshold: float | None = None,
    ) -> Sequence[Passage]:
        """Return up to `limit` passages relevant to `query`, best first.

        Args:
            query: What to look for.
            limit: The most passages to return.
            filter: Only passages whose metadata matches; `None` for all.
            mode: How to match the query; `None` for the backend's default.
            score_threshold: Drop passages scoring below this, on the backend's own scale; `None` for no cut.
        """
        ...


@runtime_checkable
class ReadableKnowledgeBase(Protocol):
    """A backend that can also return a whole document by id, which adds the `read_document` tool."""

    async def read(self, document_id: str, *, max_chars: int, filter: FilterNode | None) -> Document | None:
        """Return the document, or `None` when there is none with that id or it doesn't match `filter`.

        `filter` is the run's search filter. Applying it here is what keeps a tenant-scoped agent from
        opening another tenant's document by id, so a backend must return `None` for a document whose
        metadata doesn't match, and should fail closed with `allows_read`: a document missing a field the
        filter names, or whose metadata can't be read, is not returned. The text may stop after `max_chars` characters (one more lets the caller
        see that it continues).
        """
        ...


def is_readable(backend: object) -> bool:
    """Whether `backend` reads documents: it has `read`, and doesn't turn reading off with `can_read = False`.

    A backend whose reading depends on its configuration, like `BedrockKnowledgeBase` with
    `read_source_documents`, exposes `can_read`; one that always reads needs nothing beyond `read`.
    """
    return isinstance(backend, ReadableKnowledgeBase) and getattr(backend, 'can_read', True) is not False


@runtime_checkable
class NativeFileSearchBackend(Protocol):
    """A backend whose index the model provider can also search natively, with `FileSearchTool`."""

    def file_search_tool(self, ctx: RunContext[object], *, limit: int) -> FileSearchTool | None:
        """The native tool that searches this index for the run's model, or `None` where the model can't."""
        ...
