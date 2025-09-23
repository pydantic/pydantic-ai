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
DepsT = TypeVar('DepsT', infer_variance=True)
OutputT = TypeVar('OutputT', infer_variance=True)
BranchSourceT = TypeVar('BranchSourceT', infer_variance=True)
DecisionHandledT = TypeVar('DecisionHandledT', infer_variance=True)

HandledT = TypeVar('HandledT', infer_variance=True)
S = TypeVar('S', infer_variance=True)
T = TypeVar('T', infer_variance=True)
NewOutputT = TypeVar('NewOutputT', infer_variance=True)
SourceT = TypeVar('SourceT', infer_variance=True)


@dataclass
class Decision(Generic[StateT, DepsT, HandledT]):
    """A decision."""

    id: NodeId
    branches: list[DecisionBranch[Any]]
    note: str | None

    def branch(self, branch: DecisionBranch[S]) -> Decision[StateT, DepsT, HandledT | S]:
        # TODO(P3): Add an overload that skips the need for `match`, and is just less flexible about the building.
        #   I discussed this with Douwe but don't fully remember the details...
        return Decision(id=self.id, branches=self.branches + [branch], note=self.note)

    def _force_handled_contravariant(self, inputs: HandledT) -> None:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


@dataclass
class DecisionBranch(Generic[SourceT]):
    """A decision branch."""

    source: TypeOrTypeExpression[SourceT]
    matches: Callable[[Any], bool] | None
    path: Path


@dataclass
class DecisionBranchBuilder(Generic[StateT, DepsT, OutputT, BranchSourceT, DecisionHandledT]):
    """A builder for a decision branch."""

    decision: Decision[StateT, DepsT, DecisionHandledT]
    source: TypeOrTypeExpression[BranchSourceT]
    matches: Callable[[Any], bool] | None
    path_builder: PathBuilder[StateT, DepsT, OutputT]

    @property
    def last_fork_id(self) -> ForkId | None:
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
        return DecisionBranch(
            source=self.source, matches=self.matches, path=self.path_builder.to(destination, *extra_destinations)
        )

    def fork(
        self,
        get_forks: Callable[[Self], Sequence[Decision[StateT, DepsT, DecisionHandledT | BranchSourceT]]],
        /,
    ) -> DecisionBranch[BranchSourceT]:
        n_initial_branches = len(self.decision.branches)
        fork_decisions = get_forks(self)
        new_paths = [b.path for fd in fork_decisions for b in fd.branches[n_initial_branches:]]
        return DecisionBranch(source=self.source, matches=self.matches, path=self.path_builder.fork(new_paths))

    def transform(
        self, func: StepFunction[StateT, DepsT, OutputT, NewOutputT], /
    ) -> DecisionBranchBuilder[StateT, DepsT, NewOutputT, BranchSourceT, DecisionHandledT]:
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
