"""Decision node implementation for conditional branching in graph execution.

This module provides the Decision node type and related classes for implementing
conditional branching logic in execution graphs. Decision nodes allow the graph
to choose different execution paths based on runtime conditions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import Self, TypeVar

from pydantic_graph.v2.id_types import ForkId, NodeId
from pydantic_graph.v2.paths import Path, PathBuilder
from pydantic_graph.v2.step import StepFunction
from pydantic_graph.v2.util import TypeOrTypeExpression

if TYPE_CHECKING:
    from pydantic_graph.v2.node_types import DestinationNode

StateT = TypeVar('StateT', infer_variance=True)
"""Type variable for graph state."""

DepsT = TypeVar('DepsT', infer_variance=True)
"""Type variable for dependencies."""

OutputT = TypeVar('OutputT', infer_variance=True)
"""Type variable for output data."""

BranchSourceT = TypeVar('BranchSourceT', infer_variance=True)
"""Type variable for branch source data."""

DecisionHandledT = TypeVar('DecisionHandledT', infer_variance=True)
"""Type variable for types handled by the decision."""

HandledT = TypeVar('HandledT', infer_variance=True)
"""Type variable for handled types."""

S = TypeVar('S', infer_variance=True)
"""Generic type variable."""

T = TypeVar('T', infer_variance=True)
"""Generic type variable."""

NewOutputT = TypeVar('NewOutputT', infer_variance=True)
"""Type variable for transformed output."""

SourceT = TypeVar('SourceT', infer_variance=True)
"""Type variable for source data."""


@dataclass
class Decision(Generic[StateT, DepsT, HandledT]):
    """Decision node for conditional branching in graph execution.

    A Decision node evaluates conditions and routes execution to different
    branches based on the input data type or custom matching logic.
    """

    id: NodeId
    """Unique identifier for this decision node."""

    branches: list[DecisionBranch[Any]]
    """List of branches that can be taken from this decision."""

    note: str | None
    """Optional documentation note for this decision."""

    def branch(self, branch: DecisionBranch[S]) -> Decision[StateT, DepsT, HandledT | S]:
        """Add a new branch to this decision.

        Args:
            branch: The branch to add to this decision.

        Returns:
            A new Decision with the additional branch.

        Note:
            TODO(P3): Add an overload that skips the need for `match`, and is just less flexible about the building.
        """
        return Decision(id=self.id, branches=self.branches + [branch], note=self.note)

    def _force_handled_contravariant(self, inputs: HandledT) -> None:
        """Force type variance for proper generic typing.

        Args:
            inputs: Input data of handled types.

        Raises:
            RuntimeError: Always, as this method should never be executed.
        """
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


@dataclass
class DecisionBranch(Generic[SourceT]):
    """Represents a single branch within a decision node.

    Each branch defines the conditions under which it should be taken
    and the path to follow when those conditions are met.
    """

    source: TypeOrTypeExpression[SourceT]
    """The expected type of data for this branch."""

    matches: Callable[[Any], bool] | None
    """Optional predicate function to match against input data."""

    path: Path
    """The execution path to follow when this branch is taken."""


@dataclass
class DecisionBranchBuilder(Generic[StateT, DepsT, OutputT, BranchSourceT, DecisionHandledT]):
    """Builder for constructing decision branches with fluent API.

    This builder provides methods to configure branches with destinations,
    forks, and transformations in a type-safe manner.
    """

    decision: Decision[StateT, DepsT, DecisionHandledT]
    """The parent decision node."""

    source: TypeOrTypeExpression[BranchSourceT]
    """The expected source type for this branch."""

    matches: Callable[[Any], bool] | None
    """Optional matching predicate."""

    path_builder: PathBuilder[StateT, DepsT, OutputT]
    """Builder for the execution path."""

    @property
    def last_fork_id(self) -> ForkId | None:
        """Get the ID of the last fork in the path.

        Returns:
            The fork ID if a fork exists, None otherwise.
        """
        last_fork = self.path_builder.last_fork
        if last_fork is None:
            return None
        return last_fork.fork_id

    def to(
        self,
        destination: DestinationNode[StateT, DepsT, OutputT],
        /,
        *extra_destinations: DestinationNode[StateT, DepsT, OutputT],
    ) -> DecisionBranch[BranchSourceT]:
        """Set the destination(s) for this branch.

        Args:
            destination: The primary destination node.
            *extra_destinations: Additional destination nodes.

        Returns:
            A completed DecisionBranch with the specified destinations.
        """
        return DecisionBranch(
            source=self.source, matches=self.matches, path=self.path_builder.to(destination, *extra_destinations)
        )

    def fork(
        self,
        get_forks: Callable[[Self], Sequence[Decision[StateT, DepsT, DecisionHandledT | BranchSourceT]]],
        /,
    ) -> DecisionBranch[BranchSourceT]:
        """Create a fork in the execution path.

        Args:
            get_forks: Function that generates fork decisions.

        Returns:
            A DecisionBranch with forked execution paths.
        """
        n_initial_branches = len(self.decision.branches)
        fork_decisions = get_forks(self)
        new_paths = [b.path for fd in fork_decisions for b in fd.branches[n_initial_branches:]]
        return DecisionBranch(source=self.source, matches=self.matches, path=self.path_builder.fork(new_paths))

    def transform(
        self, func: StepFunction[StateT, DepsT, OutputT, NewOutputT], /
    ) -> DecisionBranchBuilder[StateT, DepsT, NewOutputT, BranchSourceT, DecisionHandledT]:
        """Apply a transformation to the branch's output.

        Args:
            func: Transformation function to apply.

        Returns:
            A new builder with the transformed output type.
        """
        return DecisionBranchBuilder(
            decision=self.decision,
            source=self.source,
            matches=self.matches,
            path_builder=self.path_builder.transform(func),
        )

    def spread(
        self: DecisionBranchBuilder[StateT, DepsT, Iterable[T], BranchSourceT, DecisionHandledT],
    ) -> DecisionBranchBuilder[StateT, DepsT, T, BranchSourceT, DecisionHandledT]:
        return DecisionBranchBuilder(
            decision=self.decision, source=self.source, matches=self.matches, path_builder=self.path_builder.spread()
        )

    def label(self, label: str) -> DecisionBranchBuilder[StateT, DepsT, OutputT, BranchSourceT, DecisionHandledT]:
        return DecisionBranchBuilder(
            decision=self.decision,
            source=self.source,
            matches=self.matches,
            path_builder=self.path_builder.label(label),
        )
