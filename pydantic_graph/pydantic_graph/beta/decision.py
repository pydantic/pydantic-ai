"""Decision node implementation for conditional branching in graph execution.

This module provides the Decision node type and related classes for implementing
conditional branching logic in parallel control flow graphs. Decision nodes allow the graph
to choose different execution paths based on runtime conditions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import Never, Self, TypeVar

from pydantic_graph.beta.id_types import ForkID, JoinID, NodeID
from pydantic_graph.beta.paths import Path, PathBuilder, TransformFunction
from pydantic_graph.beta.util import TypeOrTypeExpression

if TYPE_CHECKING:
    from pydantic_graph.beta.node_types import DestinationNode

StateT = TypeVar('StateT', infer_variance=True)
"""Type variable for graph state."""

DepsT = TypeVar('DepsT', infer_variance=True)
"""Type variable for graph dependencies."""

HandledT = TypeVar('HandledT', infer_variance=True)
"""Type variable used to track types handled by the branches of a Decision."""

T = TypeVar('T', infer_variance=True)
"""Generic type variable."""


@dataclass(kw_only=True)
class Decision(Generic[StateT, DepsT, HandledT]):
    """Decision node for conditional branching in graph execution.

    A Decision node evaluates conditions and routes execution to different
    branches based on the input data type or custom matching logic.
    """

    id: NodeID
    """Unique identifier for this decision node."""

    branches: list[DecisionBranch[Any]]
    """List of branches that can be taken from this decision."""

    note: str | None
    """Optional documentation note for this decision."""

    def branch(self, branch: DecisionBranch[T]) -> Decision[StateT, DepsT, HandledT | T]:
        """Add a new branch to this decision.

        Args:
            branch: The branch to add to this decision.

        Returns:
            A new Decision with the additional branch.

        Note:
            TODO(P3): Add an overload that skips the need for `match`, and is just less flexible about the building.
        """
        return Decision(id=self.id, branches=self.branches + [branch], note=self.note)

    def _force_handled_contravariant(self, inputs: HandledT) -> Never:  # pragma: no cover
        """Forces this type to be contravariant in the HandledT type variable.

        This is an implementation detail of how we can type-check that all possible input types have
        been exhaustively covered.

        Args:
            inputs: Input data of handled types.

        Raises:
            RuntimeError: Always, as this method should never be executed.
        """
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


SourceT = TypeVar('SourceT', infer_variance=True)
"""Type variable for source data for a DecisionBranch."""


@dataclass
class DecisionBranch(Generic[SourceT]):
    """Represents a single branch within a decision node.

    Each branch defines the conditions under which it should be taken
    and the path to follow when those conditions are met.

    Note: with the current design, it is actually _critical_ that this class is invariant in SourceT for the sake
    of type-checking that inputs to a Decision are actually handled. See the `# type: ignore` comment in
    `tests.graph.beta.test_graph_edge_cases.test_decision_no_matching_branch` for an example of how this works.
    """

    source: TypeOrTypeExpression[SourceT]
    """The expected type of data for this branch.

    This is necessary for exhaustiveness-checking when handling the inputs to a decision node."""

    matches: Callable[[Any], bool] | None
    """An optional predicate function used to determine whether input data matches this branch.

    If `None`, default logic is used which attempts to check the value for type-compatibility with the `source` type:
    * If `source` is `Any` or `object`, the branch will always match
    * If `source` is a `Literal` type, this branch will match if the value is one of the parametrizing literal values
    * If `source` is any other type, the value will be checked for matching using `isinstance`

    Inputs are tested against each branch of a decision node in order, and the path of the first matching branch is
    used to handle the input value.
    """

    path: Path
    """The execution path to follow when an input value matches this branch of a decision node.

    This can include transforming, mapping, and broadcasting the output before sending to the next node or nodes.

    The path can also include position-aware labels which are used when generating mermaid diagrams."""


OutputT = TypeVar('OutputT', infer_variance=True)
"""Type variable for the output data of a node."""

NewOutputT = TypeVar('NewOutputT', infer_variance=True)
"""Type variable for transformed output."""


@dataclass(init=False)
class DecisionBranchBuilder(Generic[StateT, DepsT, OutputT, SourceT, HandledT]):
    """Builder for constructing decision branches with fluent API.

    This builder provides methods to configure branches with destinations,
    forks, and transformations in a type-safe manner.

    Instances of this class should be created using [`GraphBuilder.match`][pydantic_graph.beta.graph_builder.GraphBuilder],
    not created directly.
    """

    _decision: Decision[StateT, DepsT, HandledT]
    """The parent decision node."""
    _source: TypeOrTypeExpression[SourceT]
    """The expected source type for this branch."""
    _matches: Callable[[Any], bool] | None
    """Optional matching predicate."""
    _path_builder: PathBuilder[StateT, DepsT, OutputT]
    """Builder for the execution path."""

    def __init__(
        self,
        *,
        decision: Decision[StateT, DepsT, HandledT],
        source: TypeOrTypeExpression[SourceT],
        matches: Callable[[Any], bool] | None,
        path_builder: PathBuilder[StateT, DepsT, OutputT],
    ):
        # This manually-defined initializer is necessary due to https://github.com/python/mypy/issues/17623.
        self._decision = decision
        self._source = source
        self._matches = matches
        self._path_builder = path_builder

    def to(
        self,
        destination: DestinationNode[StateT, DepsT, OutputT],
        /,
        *extra_destinations: DestinationNode[StateT, DepsT, OutputT],
    ) -> DecisionBranch[SourceT]:
        """Set the destination(s) for this branch.

        Args:
            destination: The primary destination node.
            *extra_destinations: Additional destination nodes.

        Returns:
            A completed DecisionBranch with the specified destinations.
        """
        return DecisionBranch(
            source=self._source, matches=self._matches, path=self._path_builder.to(destination, *extra_destinations)
        )

    def broadcast(
        self,
        get_forks: Callable[[Self], Sequence[DecisionBranch[SourceT]]],
        /,
    ) -> DecisionBranch[SourceT]:
        """Create a broadcast fork in the execution path.

        Args:
            get_forks: Function (typically a lambda) that returns the broadcast forks downstream of this decision branch.

        Returns:
            A completed DecisionBranch with broadcast-forked execution paths.
        """
        fork_decision_branches = get_forks(self)
        new_paths = [b.path for b in fork_decision_branches]
        return DecisionBranch(source=self._source, matches=self._matches, path=self._path_builder.broadcast(new_paths))

    def transform(
        self, func: TransformFunction[StateT, DepsT, OutputT, NewOutputT], /
    ) -> DecisionBranchBuilder[StateT, DepsT, NewOutputT, SourceT, HandledT]:
        """Apply a transformation to the branch's output.

        Args:
            func: Transformation function to apply.

        Returns:
            A new DecisionBranchBuilder where the provided transform is applied prior to generating the final output.
        """
        return DecisionBranchBuilder(
            decision=self._decision,
            source=self._source,
            matches=self._matches,
            path_builder=self._path_builder.transform(func),
        )

    def map(
        self: DecisionBranchBuilder[StateT, DepsT, Iterable[T], SourceT, HandledT],
        *,
        fork_id: ForkID | None = None,
        downstream_join_id: JoinID | None = None,
    ) -> DecisionBranchBuilder[StateT, DepsT, T, SourceT, HandledT]:
        """Spread the branch's output.

        To do this, the current output must be iterable, and any subsequent steps in the path being built for this
        branch will be applied to each item of the current output in parallel.

        Args:
            fork_id: Optional ID for the fork, defaults to a generated value
            downstream_join_id: Optional ID of a downstream join node which is involved when mapping empty iterables

        Returns:
            A new DecisionBranchBuilder where mapping is performed prior to generating the final output.
        """
        # TODO: Need to cover this in a test
        return DecisionBranchBuilder(
            decision=self._decision,
            source=self._source,
            matches=self._matches,
            path_builder=self._path_builder.map(fork_id=fork_id, downstream_join_id=downstream_join_id),
        )

    def label(self, label: str) -> DecisionBranchBuilder[StateT, DepsT, OutputT, SourceT, HandledT]:
        """Apply a label to the branch at the current point in the path being built.

        These labels are only used in generated mermaid diagrams.

        Args:
            label: The label to apply.

        Returns:
            A new DecisionBranchBuilder where the label has been applied at the end of the current path being built.
        """
        return DecisionBranchBuilder(
            decision=self._decision,
            source=self._source,
            matches=self._matches,
            path_builder=self._path_builder.label(label),
        )
