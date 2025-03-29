from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from textwrap import indent
from typing import TYPE_CHECKING, Any, Callable

from typing_extensions import TypedDict

if TYPE_CHECKING:  # pragma: no cover
    # Since opentelemetry isn't a required dependency, don't actually import these at runtime
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.trace import SpanContext
    from opentelemetry.util.types import AttributeValue

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

    # Related-span conditions
    ## Ancestor conditions
    min_depth: int  # depth is equivalent to ancestor count; roots have depth 0
    max_depth: int
    some_ancestor_has: SpanQuery
    all_ancestors_have: SpanQuery
    no_ancestor_has: SpanQuery

    ## Child conditions
    min_child_count: int
    max_child_count: int
    some_child_has: SpanQuery
    all_children_have: SpanQuery
    no_child_has: SpanQuery

    ## Descendant conditions
    min_descendant_count: int
    max_descendant_count: int
    some_descendant_has: SpanQuery
    all_descendants_have: SpanQuery
    no_descendant_has: SpanQuery


class SpanNode:
    """A node in the span tree; provides references to parents/children for easy traversal and queries."""

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __init__(self, span: ReadableSpan):
        self._span = span
        # If a span has no context, it's going to cause problems. We may need to add improved handling of this scenario.
        assert self._span.context is not None, f'{span=} has no context'

        self.parent: SpanNode | None = None
        self.children_by_id: dict[int, SpanNode] = {}  # note: we rely on insertion order to determine child order

    def add_child(self, child: SpanNode) -> None:
        """Attach a child node to this node's list of children."""
        self.children_by_id[child.span_id] = child
        child.parent = self

    # -------------------------------------------------------------------------
    # Utility properties
    # -------------------------------------------------------------------------
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
    def context(self) -> SpanContext:
        """Return the SpanContext of the wrapped span."""
        assert self._span.context is not None
        return self._span.context

    @property
    def parent_context(self) -> SpanContext | None:
        """Return the SpanContext of the parent of the wrapped span."""
        return self._span.parent

    @property
    def span_id(self) -> int:
        """Return the integer span_id from the SpanContext."""
        return self.context.span_id

    @property
    def trace_id(self) -> int:
        """Return the integer trace_id from the SpanContext."""
        return self.context.trace_id

    @property
    def name(self) -> str:
        """Convenience for the span's name."""
        return self._span.name

    @property
    def start_timestamp(self) -> datetime:
        """Return the span's start time as a UTC datetime, or None if not set."""
        assert self._span.start_time is not None
        return datetime.fromtimestamp(self._span.start_time / 1e9, tz=timezone.utc)

    @property
    def end_timestamp(self) -> datetime:
        """Return the span's end time as a UTC datetime, or None if not set."""
        assert self._span.end_time is not None
        return datetime.fromtimestamp(self._span.end_time / 1e9, tz=timezone.utc)

    @property
    def duration(self) -> timedelta:
        """Return the span's duration as a timedelta, or None if start/end not set."""
        return self.end_timestamp - self.start_timestamp

    @property
    def attributes(self) -> Mapping[str, AttributeValue]:
        # Note: It would be nice to expose the non-JSON-serialized versions of (logfire-recorded) attributes with
        # nesting etc. This just exposes the JSON-serialized version, but doing more would be difficult.
        return self._span.attributes or {}

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
    def find_descendants(self, predicate: SpanQuery | SpanPredicate) -> list[SpanNode]:
        """Return all descendant nodes that satisfy the given predicate in DFS order."""
        return list(self._filter_descendants(predicate))

    def first_descendant(self, predicate: SpanQuery | SpanPredicate) -> SpanNode | None:
        """DFS: Return the first descendant (in DFS order) that satisfies the given predicate, or `None` if none match."""
        return next(self._filter_descendants(predicate), None)

    def any_descendant(self, predicate: SpanQuery | SpanPredicate) -> bool:
        """Returns `True` if there is at least one descendant that satisfies the predicate."""
        return self.first_descendant(predicate) is not None

    def _filter_descendants(self, predicate: SpanQuery | SpanPredicate) -> Iterator[SpanNode]:
        stack = list(self.children)
        while stack:
            node = stack.pop()
            if node.matches(predicate):
                yield node
            stack.extend(node.children)

    # -------------------------------------------------------------------------
    # Ancestor queries (DFS "up" the chain)
    # -------------------------------------------------------------------------
    def find_ancestors(self, predicate: SpanQuery | SpanPredicate) -> list[SpanNode]:
        """Return all ancestors that satisfy the given predicate."""
        return list(self._filter_ancestors(predicate))

    def first_ancestor(self, predicate: SpanQuery | SpanPredicate) -> SpanNode | None:
        """Return the closest ancestor that satisfies the given predicate, or `None` if none match."""
        return next(self._filter_ancestors(predicate), None)

    def any_ancestor(self, predicate: SpanQuery | SpanPredicate) -> bool:
        """Returns True if any ancestor satisfies the predicate."""
        return self.first_ancestor(predicate) is not None

    def _filter_ancestors(self, predicate: SpanQuery | SpanPredicate) -> Iterator[SpanNode]:
        node = self.parent
        while node:
            if node.matches(predicate):
                yield node
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

        # Ancestor conditions
        if (min_depth := query.get('min_depth')) and len(self.ancestors) < min_depth:
            return False
        if (max_depth := query.get('max_depth')) and len(self.ancestors) > max_depth:
            return False
        if (some_ancestor_has := query.get('some_ancestor_has')) and not any(
            ancestor._matches_query(some_ancestor_has) for ancestor in self.ancestors
        ):
            return False
        if (all_ancestors_have := query.get('all_ancestors_have')) and not all(
            ancestor._matches_query(all_ancestors_have) for ancestor in self.ancestors
        ):
            return False
        if (no_ancestor_has := query.get('no_ancestor_has')) and any(
            ancestor._matches_query(no_ancestor_has) for ancestor in self.ancestors
        ):
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
        if (min_descendant_count := query.get('min_descendant_count')) and len(self.descendants) < min_descendant_count:
            return False
        if (max_descendant_count := query.get('max_descendant_count')) and len(self.descendants) > max_descendant_count:
            return False
        if (some_descendant_has := query.get('some_descendant_has')) and not any(
            descendant._matches_query(some_descendant_has) for descendant in self.descendants
        ):
            return False
        if (all_descendants_have := query.get('all_descendants_have')) and not all(
            descendant._matches_query(all_descendants_have) for descendant in self.descendants
        ):
            return False
        if (no_descendant_has := query.get('no_descendant_has')) and any(
            descendant._matches_query(no_descendant_has) for descendant in self.descendants
        ):
            return False

        return True

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------
    def repr_xml(
        self,
        include_children: bool = True,
        include_span_id: bool = False,
        include_trace_id: bool = False,
        include_start_timestamp: bool = False,
        include_duration: bool = False,
    ) -> str:
        """Return an XML-like string representation of the node.

        Optionally includes children, span_id, trace_id, start_timestamp, and duration.
        """
        first_line_parts = [f'<SpanNode name={self.name!r}']
        if include_span_id:
            first_line_parts.append(f'span_id={self.span_id:016x}')
        if include_trace_id:
            first_line_parts.append(f'trace_id={self.trace_id:032x}')
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
                            include_span_id=include_span_id,
                            include_trace_id=include_trace_id,
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
            return f'<SpanNode name={self.name!r} span_id={self.span_id:016x}>...</SpanNode>'
        else:
            return f'<SpanNode name={self.name!r} span_id={self.span_id:016x} />'

    def __repr__(self) -> str:
        return self.repr_xml()


SpanPredicate = Callable[[SpanNode], bool]


class SpanTree:
    """A container that builds a hierarchy of SpanNode objects from a list of finished spans.

    You can then search or iterate the tree to make your assertions (using DFS for traversal).
    """

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __init__(self, spans: list[ReadableSpan] | None = None):
        self.nodes_by_id: dict[int, SpanNode] = {}
        self.roots: list[SpanNode] = []
        if spans:  # pragma: no cover
            self.add_spans(spans)

    def add_spans(self, spans: list[ReadableSpan]) -> None:
        """Add a list of spans to the tree, rebuilding the tree structure."""
        for span in spans:
            node = SpanNode(span)
            self.nodes_by_id[node.span_id] = node
        self._rebuild_tree()

    def _rebuild_tree(self):
        # Ensure spans are ordered by start_timestamp so that roots and children end up in the right order
        nodes = list(self.nodes_by_id.values())
        nodes.sort(key=lambda node: node.start_timestamp or datetime.min)
        self.nodes_by_id = {node.span_id: node for node in nodes}

        # Build the parent/child relationships
        for node in self.nodes_by_id.values():
            parent_ctx = node.parent_context
            if parent_ctx is not None:
                parent_node = self.nodes_by_id.get(parent_ctx.span_id)
                if parent_node is not None:
                    parent_node.add_child(node)

        # Determine the roots
        # A node is a "root" if its parent is None or if its parent's span_id is not in the current set of spans.
        self.roots = []
        for node in self.nodes_by_id.values():
            parent_ctx = node.parent_context
            if parent_ctx is None or parent_ctx.span_id not in self.nodes_by_id:
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
        include_span_id: bool = False,
        include_trace_id: bool = False,
        include_start_timestamp: bool = False,
        include_duration: bool = False,
    ) -> str:
        """Return an XML-like string representation of the tree, optionally including children, span_id, trace_id, duration, and timestamps."""
        if not self.roots:
            return '<SpanTree />'
        repr_parts = [
            '<SpanTree>',
            *[
                indent(
                    root.repr_xml(
                        include_children=include_children,
                        include_span_id=include_span_id,
                        include_trace_id=include_trace_id,
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
