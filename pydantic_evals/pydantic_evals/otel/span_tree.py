from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import cache
from textwrap import indent
from typing import TYPE_CHECKING, Any, Callable, Union

from pydantic import TypeAdapter
from typing_extensions import TypedDict

if TYPE_CHECKING:  # pragma: no cover
    # Since opentelemetry isn't a required dependency, don't actually import these at runtime
    from opentelemetry.sdk.trace import ReadableSpan

# Should match opentelemetry.util.types.AttributeValue
AttributeValue = Union[
    str,
    bool,
    int,
    float,
    Sequence[str],
    Sequence[bool],
    Sequence[int],
    Sequence[float],
]


__all__ = 'SpanNode', 'SpanTree', 'SpanQuery'


class SpanQuery(TypedDict, total=False):
    """A serializable query for filtering SpanNodes based on various conditions.

    All fields are optional and combined with AND logic by default.
    """

    # These fields are ordered to match the implementation of SpanNode.matches_query for easy review.
    # * Individual span conditions come first because these are generally the cheapest to evaluate
    # * Logical combinations come next because they may just be combinations of individual span conditions
    # * Related-span conditions come last because they may require the most work to evaluate

    # Individual span conditions
    ## Name conditions
    name_equals: str
    name_contains: str
    name_matches_regex: str  # regex pattern

    ## Attribute conditions
    has_attributes: dict[str, Any]
    has_attribute_keys: list[str]

    ## Timing conditions
    min_duration: timedelta | float
    max_duration: timedelta | float

    # Logical combinations of conditions
    not_: SpanQuery
    and_: list[SpanQuery]
    or_: list[SpanQuery]

    # Child conditions
    min_child_count: int
    max_child_count: int
    some_child_has: SpanQuery
    all_children_have: SpanQuery
    no_child_has: SpanQuery

    # Recursive conditions
    stop_recursing_when: SpanQuery
    """If present, stop recursing through ancestors or descendants at nodes that match this condition."""

    ## Descendant conditions
    min_descendant_count: int
    max_descendant_count: int
    some_descendant_has: SpanQuery
    all_descendants_have: SpanQuery
    no_descendant_has: SpanQuery

    ## Ancestor conditions
    min_depth: int  # depth is equivalent to ancestor count; roots have depth 0
    max_depth: int
    some_ancestor_has: SpanQuery
    all_ancestors_have: SpanQuery
    no_ancestor_has: SpanQuery


@dataclass(repr=False)
class SpanNode:
    """A node in the span tree; provides references to parents/children for easy traversal and queries."""

    name: str
    trace_id: int
    span_id: int
    parent_span_id: int | None
    start_timestamp: datetime
    end_timestamp: datetime
    attributes: dict[str, AttributeValue]

    @property
    def duration(self) -> timedelta:
        """Return the span's duration as a timedelta, or None if start/end not set."""
        return self.end_timestamp - self.start_timestamp

    @property
    def children(self) -> list[SpanNode]:
        return list(self.children_by_id.values())

    @property
    def descendants(self) -> list[SpanNode]:
        """Return all descendants of this node in DFS order."""
        return self.find_descendants(lambda _: True)

    @property
    def ancestors(self) -> list[SpanNode]:
        """Return all ancestors of this node."""
        return self.find_ancestors(lambda _: True)

    @property
    def node_key(self) -> str:
        return f'{self.trace_id:032x}:{self.span_id:016x}'

    @property
    def parent_node_key(self) -> str | None:
        return None if self.parent_span_id is None else f'{self.trace_id:032x}:{self.parent_span_id:016x}'

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __post_init__(self):
        self.parent: SpanNode | None = None
        self.children_by_id: dict[str, SpanNode] = {}

    @staticmethod
    def from_readable_span(span: ReadableSpan) -> SpanNode:
        assert span.context is not None, 'Span has no context'
        assert span.start_time is not None, 'Span has no start time'
        assert span.end_time is not None, 'Span has no end time'
        return SpanNode(
            name=span.name,
            trace_id=span.context.trace_id,
            span_id=span.context.span_id,
            parent_span_id=span.parent.span_id if span.parent else None,
            start_timestamp=datetime.fromtimestamp(span.start_time / 1e9, tz=timezone.utc),
            end_timestamp=datetime.fromtimestamp(span.end_time / 1e9, tz=timezone.utc),
            attributes=dict(span.attributes or {}),
        )

    def add_child(self, child: SpanNode) -> None:
        """Attach a child node to this node's list of children."""
        assert child.trace_id == self.trace_id, f"traces don't match: {child.trace_id:032x} != {self.trace_id:032x}"
        assert child.parent_span_id == self.span_id, (
            f'parent span mismatch: {child.parent_span_id:016x} != {self.span_id:016x}'
        )
        self.children_by_id[child.node_key] = child
        child.parent = self

    # -------------------------------------------------------------------------
    # Child queries
    # -------------------------------------------------------------------------
    def find_children(self, predicate: SpanQuery | SpanPredicate) -> list[SpanNode]:
        """Return all immediate children that satisfy the given predicate."""
        return list(self._filter_children(predicate))

    def first_child(self, predicate: SpanQuery | SpanPredicate) -> SpanNode | None:
        """Return the first immediate child that satisfies the given predicate, or None if none match."""
        return next(self._filter_children(predicate), None)

    def any_child(self, predicate: SpanQuery | SpanPredicate) -> bool:
        """Returns True if there is at least one child that satisfies the predicate."""
        return self.first_child(predicate) is not None

    def _filter_children(self, predicate: SpanQuery | SpanPredicate) -> Iterator[SpanNode]:
        return (child for child in self.children if child.matches(predicate))

    # -------------------------------------------------------------------------
    # Descendant queries (DFS)
    # -------------------------------------------------------------------------
    def find_descendants(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
    ) -> list[SpanNode]:
        """Return all descendant nodes that satisfy the given predicate in DFS order."""
        return list(self._filter_descendants(predicate, stop_recursing_when))

    def first_descendant(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
    ) -> SpanNode | None:
        """DFS: Return the first descendant (in DFS order) that satisfies the given predicate, or `None` if none match."""
        return next(self._filter_descendants(predicate, stop_recursing_when), None)

    def any_descendant(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
    ) -> bool:
        """Returns `True` if there is at least one descendant that satisfies the predicate."""
        return self.first_descendant(predicate, stop_recursing_when) is not None

    def _filter_descendants(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None
    ) -> Iterator[SpanNode]:
        stack = list(self.children)
        while stack:
            node = stack.pop()
            if node.matches(predicate):
                yield node
            if stop_recursing_when is not None and node.matches(stop_recursing_when):
                continue
            stack.extend(node.children)

    # -------------------------------------------------------------------------
    # Ancestor queries (DFS "up" the chain)
    # -------------------------------------------------------------------------
    def find_ancestors(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
    ) -> list[SpanNode]:
        """Return all ancestors that satisfy the given predicate."""
        return list(self._filter_ancestors(predicate, stop_recursing_when))

    def first_ancestor(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
    ) -> SpanNode | None:
        """Return the closest ancestor that satisfies the given predicate, or `None` if none match."""
        return next(self._filter_ancestors(predicate, stop_recursing_when), None)

    def any_ancestor(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
    ) -> bool:
        """Returns True if any ancestor satisfies the predicate."""
        return self.first_ancestor(predicate, stop_recursing_when) is not None

    def _filter_ancestors(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None
    ) -> Iterator[SpanNode]:
        node = self.parent
        while node:
            if node.matches(predicate):
                yield node
            if stop_recursing_when is not None and node.matches(stop_recursing_when):
                break
            node = node.parent

    # -------------------------------------------------------------------------
    # Query matching
    # -------------------------------------------------------------------------
    def matches(self, query: SpanQuery | SpanPredicate) -> bool:
        """Check if the span node matches the query conditions or predicate."""
        if callable(query):
            return query(self)

        return self._matches_query(query)

    def _matches_query(self, query: SpanQuery) -> bool:  # noqa C901
        """Check if the span matches the query conditions."""
        # Logical combinations
        if or_ := query.get('or_'):
            if len(query) > 1:
                raise ValueError("Cannot combine 'or_' conditions with other conditions at the same level")
            return any(self._matches_query(q) for q in or_)
        if not_ := query.get('not_'):
            if self._matches_query(not_):
                return False
        if and_ := query.get('and_'):
            results = [self._matches_query(q) for q in and_]
            if not all(results):
                return False
        # At this point, all existing ANDs and no existing ORs have passed, so it comes down to this condition

        # Name conditions
        if (name_equals := query.get('name_equals')) and self.name != name_equals:
            return False
        if (name_contains := query.get('name_contains')) and name_contains not in self.name:
            return False
        if (name_matches_regex := query.get('name_matches_regex')) and not re.match(name_matches_regex, self.name):
            return False

        # Attribute conditions
        if (has_attributes := query.get('has_attributes')) and not all(
            self.attributes.get(key) == value for key, value in has_attributes.items()
        ):
            return False
        if (has_attributes_keys := query.get('has_attribute_keys')) and not all(
            key in self.attributes for key in has_attributes_keys
        ):
            return False

        # Timing conditions
        if (min_duration := query.get('min_duration')) is not None:
            if not isinstance(min_duration, timedelta):
                min_duration = timedelta(seconds=min_duration)
            if self.duration < min_duration:
                return False
        if (max_duration := query.get('max_duration')) is not None:
            if not isinstance(max_duration, timedelta):
                max_duration = timedelta(seconds=max_duration)
            if self.duration > max_duration:
                return False

        # Children conditions
        if (min_child_count := query.get('min_child_count')) and len(self.children) < min_child_count:
            return False
        if (max_child_count := query.get('max_child_count')) and len(self.children) > max_child_count:
            return False
        if (some_child_has := query.get('some_child_has')) and not any(
            child._matches_query(some_child_has) for child in self.children
        ):
            return False
        if (all_children_have := query.get('all_children_have')) and not all(
            child._matches_query(all_children_have) for child in self.children
        ):
            return False
        if (no_child_has := query.get('no_child_has')) and any(
            child._matches_query(no_child_has) for child in self.children
        ):
            return False

        # Descendant conditions
        # The following local functions with cache decorators are used to avoid repeatedly evaluating these properties
        @cache
        def descendants():
            return self.descendants

        @cache
        def pruned_descendants():
            stop_recursing_when = query.get('stop_recursing_when')
            return (
                self._filter_descendants(lambda _: True, stop_recursing_when) if stop_recursing_when else descendants()
            )

        if (min_descendant_count := query.get('min_descendant_count')) and len(descendants()) < min_descendant_count:
            return False
        if (max_descendant_count := query.get('max_descendant_count')) and len(descendants()) > max_descendant_count:
            return False
        if (some_descendant_has := query.get('some_descendant_has')) and not any(
            descendant._matches_query(some_descendant_has) for descendant in pruned_descendants()
        ):
            return False
        if (all_descendants_have := query.get('all_descendants_have')) and not all(
            descendant._matches_query(all_descendants_have) for descendant in pruned_descendants()
        ):
            return False
        if (no_descendant_has := query.get('no_descendant_has')) and any(
            descendant._matches_query(no_descendant_has) for descendant in pruned_descendants()
        ):
            return False

        # Ancestor conditions
        # The following local functions with cache decorators are used to avoid repeatedly evaluating these properties
        @cache
        def ancestors():
            return self.ancestors

        @cache
        def pruned_ancestors():
            stop_recursing_when = query.get('stop_recursing_when')
            return self._filter_ancestors(lambda _: True, stop_recursing_when) if stop_recursing_when else ancestors()

        if (min_depth := query.get('min_depth')) and len(ancestors()) < min_depth:
            return False
        if (max_depth := query.get('max_depth')) and len(ancestors()) > max_depth:
            return False
        if (some_ancestor_has := query.get('some_ancestor_has')) and not any(
            ancestor._matches_query(some_ancestor_has) for ancestor in pruned_ancestors()
        ):
            return False
        if (all_ancestors_have := query.get('all_ancestors_have')) and not all(
            ancestor._matches_query(all_ancestors_have) for ancestor in pruned_ancestors()
        ):
            return False
        if (no_ancestor_has := query.get('no_ancestor_has')) and any(
            ancestor._matches_query(no_ancestor_has) for ancestor in pruned_ancestors()
        ):
            return False

        return True

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------
    def repr_xml(
        self,
        include_children: bool = True,
        include_trace_id: bool = False,
        include_span_id: bool = False,
        include_start_timestamp: bool = False,
        include_duration: bool = False,
    ) -> str:
        """Return an XML-like string representation of the node.

        Optionally includes children, trace_id, span_id, start_timestamp, and duration.
        """
        first_line_parts = [f'<SpanNode name={self.name!r}']
        if include_trace_id:
            first_line_parts.append(f"trace_id='{self.trace_id:032x}'")
        if include_span_id:
            first_line_parts.append(f"span_id='{self.span_id:016x}'")
        if include_start_timestamp:
            first_line_parts.append(f'start_timestamp={self.start_timestamp.isoformat()!r}')
        if include_duration:
            first_line_parts.append(f"duration='{self.duration}'")

        extra_lines: list[str] = []
        if include_children and self.children:
            first_line_parts.append('>')
            for child in self.children:
                extra_lines.append(
                    indent(
                        child.repr_xml(
                            include_children=include_children,
                            include_trace_id=include_trace_id,
                            include_span_id=include_span_id,
                            include_start_timestamp=include_start_timestamp,
                            include_duration=include_duration,
                        ),
                        '  ',
                    )
                )
            extra_lines.append('</SpanNode>')
        else:
            if self.children:
                first_line_parts.append('children=...')
            first_line_parts.append('/>')
        return '\n'.join([' '.join(first_line_parts), *extra_lines])

    def __str__(self) -> str:
        if self.children:
            return f"<SpanNode name={self.name!r} span_id='{self.span_id:016x}'>...</SpanNode>"
        else:
            return f"<SpanNode name={self.name!r} span_id='{self.span_id:016x}' />"

    def __repr__(self) -> str:
        return self.repr_xml()


SpanPredicate = Callable[[SpanNode], bool]


@dataclass(repr=False)
class SpanTree:
    """A container that builds a hierarchy of SpanNode objects from a list of finished spans.

    You can then search or iterate the tree to make your assertions (using DFS for traversal).
    """

    roots: list[SpanNode] = field(default_factory=list)
    nodes_by_id: dict[str, SpanNode] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __post_init__(self):
        self._rebuild_tree()

    def add_spans(self, spans: list[SpanNode]) -> None:
        """Add a list of spans to the tree, rebuilding the tree structure."""
        for span in spans:
            self.nodes_by_id[span.node_key] = span
        self._rebuild_tree()

    def add_readable_spans(self, readable_spans: list[ReadableSpan]):
        self.add_spans([SpanNode.from_readable_span(span) for span in readable_spans])

    def _rebuild_tree(self):
        # Ensure spans are ordered by start_timestamp so that roots and children end up in the right order
        nodes = list(self.nodes_by_id.values())
        nodes.sort(key=lambda node: node.start_timestamp or datetime.min)
        self.nodes_by_id = {node.node_key: node for node in nodes}

        # Build the parent/child relationships
        for node in self.nodes_by_id.values():
            parent_node_key = node.parent_node_key
            if parent_node_key is not None:
                parent_node = self.nodes_by_id.get(parent_node_key)
                if parent_node is not None:
                    parent_node.add_child(node)

        # Determine the roots
        # A node is a "root" if its parent is None or if its parent's span_id is not in the current set of spans.
        self.roots = []
        for node in self.nodes_by_id.values():
            parent_node_key = node.parent_node_key
            if parent_node_key is None or parent_node_key not in self.nodes_by_id:
                self.roots.append(node)

    # -------------------------------------------------------------------------
    # Node filtering and iteration
    # -------------------------------------------------------------------------
    def find(self, predicate: SpanQuery | SpanPredicate) -> list[SpanNode]:
        """Find all nodes in the entire tree that match the predicate, scanning from each root in DFS order."""
        return list(self._filter(predicate))

    def first(self, predicate: SpanQuery | SpanPredicate) -> SpanNode | None:
        """Find the first node that matches a predicate, scanning from each root in DFS order. Returns `None` if not found."""
        return next(self._filter(predicate), None)

    def any(self, predicate: SpanQuery | SpanPredicate) -> bool:
        """Returns True if any node in the tree matches the predicate."""
        return self.first(predicate) is not None

    def _filter(self, predicate: SpanQuery | SpanPredicate) -> Iterator[SpanNode]:
        for node in self:
            if node.matches(predicate):
                yield node

    def __iter__(self) -> Iterator[SpanNode]:
        """Return an iterator over all nodes in the tree."""
        return iter(self.nodes_by_id.values())

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------
    def repr_xml(
        self,
        include_children: bool = True,
        include_trace_id: bool = False,
        include_span_id: bool = False,
        include_start_timestamp: bool = False,
        include_duration: bool = False,
    ) -> str:
        """Return an XML-like string representation of the tree, optionally including children, trace_id, span_id, duration, and timestamps."""
        if not self.roots:
            return '<SpanTree />'
        repr_parts = [
            '<SpanTree>',
            *[
                indent(
                    root.repr_xml(
                        include_children=include_children,
                        include_trace_id=include_trace_id,
                        include_span_id=include_span_id,
                        include_start_timestamp=include_start_timestamp,
                        include_duration=include_duration,
                    ),
                    '  ',
                )
                for root in self.roots
            ],
            '</SpanTree>',
        ]
        return '\n'.join(repr_parts)

    def __str__(self):
        return f'<SpanTree num_roots={len(self.roots)} total_spans={len(self.nodes_by_id)} />'

    def __repr__(self):
        return self.repr_xml()


SPAN_TREE_ADAPTER = TypeAdapter(SpanTree)
"""This adapter can be used to serialize and deserialize `SpanTree` objects to and from JSON."""
