from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Self

from pydantic_graph.v2.id_types import ForkId, NodeId
from pydantic_graph.v2.paths import Path, PathBuilder
from pydantic_graph.v2.step import StepFunction
from pydantic_graph.v2.util import TypeOrTypeExpression

if TYPE_CHECKING:
    from pydantic_graph.v2.node_types import DestinationNode


@dataclass
class Decision[StateT, HandledT]:
    """A decision."""

    id: NodeId
    branches: list[DecisionBranch[Any]]
    note: str | None

    def branch[S](self, branch: DecisionBranch[S]) -> Decision[StateT, HandledT | S]:
        # TODO(P3): Add an overload that skips the need for `match`, and is just less flexible about the building.
        #   I discussed this with Douwe but don't fully remember the details...
        return Decision(id=self.id, branches=self.branches + [branch], note=self.note)

    def _force_handled_contravariant(self, inputs: HandledT) -> None:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


@dataclass
class DecisionBranch[SourceT]:
    """A decision branch."""

    source: TypeOrTypeExpression[SourceT]
    matches: Callable[[Any], bool] | None
    path: Path


@dataclass
class DecisionBranchBuilder[StateT, OutputT, BranchSourceT, DecisionHandledT]:
    """A builder for a decision branch."""

    decision: Decision[StateT, DecisionHandledT]
    source: TypeOrTypeExpression[BranchSourceT]
    matches: Callable[[Any], bool] | None
    path_builder: PathBuilder[StateT, OutputT]

    @property
    def last_fork_id(self) -> ForkId | None:
        last_fork = self.path_builder.last_fork
        if last_fork is None:
            return None
        return last_fork.fork_id

    def to(
        self,
        destination: DestinationNode[StateT, OutputT],
        /,
        *extra_destinations: DestinationNode[StateT, OutputT],
    ) -> DecisionBranch[BranchSourceT]:
        return DecisionBranch(
            source=self.source, matches=self.matches, path=self.path_builder.to(destination, *extra_destinations)
        )

    def fork(
        self,
        get_forks: Callable[[Self], Sequence[Decision[StateT, DecisionHandledT | BranchSourceT]]],
        /,
    ) -> DecisionBranch[BranchSourceT]:
        n_initial_branches = len(self.decision.branches)
        fork_decisions = get_forks(self)
        new_paths = [b.path for fd in fork_decisions for b in fd.branches[n_initial_branches:]]
        return DecisionBranch(source=self.source, matches=self.matches, path=self.path_builder.fork(new_paths))

    def transform[NewOutputT](
        self, func: StepFunction[StateT, OutputT, NewOutputT], /
    ) -> DecisionBranchBuilder[StateT, NewOutputT, BranchSourceT, DecisionHandledT]:
        return DecisionBranchBuilder(
            decision=self.decision,
            source=self.source,
            matches=self.matches,
            path_builder=self.path_builder.transform(func),
        )

    def spread[T](
        self: DecisionBranchBuilder[StateT, Iterable[T], BranchSourceT, DecisionHandledT],
    ) -> DecisionBranchBuilder[StateT, T, BranchSourceT, DecisionHandledT]:
        return DecisionBranchBuilder(
            decision=self.decision, source=self.source, matches=self.matches, path_builder=self.path_builder.spread()
        )

    def label(self, label: str) -> DecisionBranchBuilder[StateT, OutputT, BranchSourceT, DecisionHandledT]:
        return DecisionBranchBuilder(
            decision=self.decision,
            source=self.source,
            matches=self.matches,
            path_builder=self.path_builder.label(label),
        )
