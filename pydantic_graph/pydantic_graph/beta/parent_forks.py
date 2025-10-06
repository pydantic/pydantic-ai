"""Parent fork identification and deadlock avoidance in parallel graph execution.

This module provides functionality to identify "parent forks" in a graph, which are dominating
fork nodes that control access to join nodes. A parent fork is a fork node that:

1. Dominates a join node (all paths to the join must pass through the fork)
2. Does not participate in cycles that bypass it to reach the join

Identifying parent forks is crucial for deadlock avoidance in parallel execution. When a join
node waits for all its incoming branches, knowing the parent fork helps determine when it's
safe to proceed without risking deadlock.

In most typical graphs, such dominating forks exist naturally. However, when there are multiple
subsequent forks, the choice of parent fork can be ambiguous and may need to be specified by
the graph designer.

TODO(P3): Expand this documentation with more detailed examples and edge cases.
"""

from __future__ import annotations

from collections.abc import Hashable
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Generic

from typing_extensions import TypeVar

T = TypeVar('T', bound=Hashable, infer_variance=True, default=str)


@dataclass
class ParentFork(Generic[T]):
    """Represents a parent fork node and its relationship to a join node.

    A parent fork is a dominating fork that controls the execution flow to a join node.
    It tracks which nodes lie between the fork and the join, which is essential for
    determining when it's safe to proceed past the join point.
    """

    fork_id: T
    """The identifier of the fork node that serves as the parent."""

    intermediate_nodes: set[T]
    """The set of node IDs of nodes upstream of the join and downstream of the parent fork.

    If there are no graph walkers in these nodes that were a part of a previous fork, it is safe to proceed downstream
    of the join.
    """


@dataclass
class ParentForkFinder(Generic[T]):
    """Analyzes graph structure to identify parent forks for join nodes.

    This class implements algorithms to find dominating forks in a directed graph,
    which is essential for coordinating parallel execution and avoiding deadlocks.
    """

    nodes: set[T]
    """All node identifiers in the graph."""

    start_ids: set[T]
    """Node identifiers that serve as entry points to the graph."""

    fork_ids: set[T]
    """Node identifiers that represent fork nodes (nodes that create parallel branches)."""

    edges: dict[T, list[T]]  # source_id to list of destination_ids
    """Graph edges represented as adjacency list mapping source nodes to destinations."""

    def find_parent_fork(self, join_id: T) -> ParentFork[T] | None:
        """Find the parent fork for a given join node.

        Searches for the _most_ ancestral dominating fork that can serve as a parent fork
        for the specified join node. A valid parent fork must dominate the join without
        allowing cycles that bypass it.

        Args:
            join_id: The identifier of the join node to analyze.

        Returns:
            A ParentFork object containing the fork ID and intermediate nodes if a valid
            parent fork exists, or None if no valid parent fork can be found (which would
            indicate potential deadlock risk).

        Note:
            If every dominating fork of the join lets it participate in a cycle that avoids
            the fork, None is returned since no valid "parent fork" exists.
        """
        visited: set[str] = set()
        cur = join_id  # start at J and walk up the immediate dominator chain

        # TODO(P2): Make it a node-configuration option to choose the most _or_ the least ancestral node as parent fork? Or manually specified(?)
        parent_fork: ParentFork[T] | None = None
        while True:
            cur = self._immediate_dominator(cur)
            if cur is None:  # reached the root
                break

            # The visited-tracking shouldn't be necessary, but I included it to prevent infinite loops if there are bugs
            assert cur not in visited, f'Cycle detected in dominator tree: {join_id} → {cur} → {visited}'
            visited.add(cur)

            if cur not in self.fork_ids:
                continue  # not a fork, so keep climbing

            upstream_nodes = self._get_upstream_nodes_if_parent(join_id, cur)
            if upstream_nodes is not None:  # found upstream nodes without a cycle
                parent_fork = ParentFork[T](cur, upstream_nodes)
            elif parent_fork is not None:
                # We reached a fork that is an ancestor of a parent fork but is not itself a parent fork.
                # This means there is a cycle to J that is downstream of `cur`, and so any node further upstream
                # will fail to be a parent fork for the same reason. So we can stop here and just return `parent_fork`.
                return parent_fork

        # No dominating fork passed the cycle test to be a "parent" fork
        return parent_fork

    @cached_property
    def _predecessors(self) -> dict[T, list[T]]:
        """Compute and cache the predecessor mapping for all nodes.

        Returns:
            A dictionary mapping each node to a list of its immediate predecessors.
        """
        predecessors: dict[T, list[T]] = {n: [] for n in self.nodes}
        for source_id in self.nodes:
            for destination_id in self.edges.get(source_id, []):
                predecessors[destination_id].append(source_id)
        return predecessors

    @cached_property
    def _dominators(self) -> dict[T, set[T]]:
        """Compute the dominator sets for all nodes using iterative dataflow analysis.

        A node D dominates node N if every path from a start node to N must pass through D.
        This is computed using a fixed-point iteration algorithm.

        Returns:
            A dictionary mapping each node to its set of dominators.
        """
        node_ids = set(self.nodes)
        start_ids = self.start_ids

        dom: dict[T, set[T]] = {n: set(node_ids) for n in node_ids}
        for s in start_ids:
            dom[s] = {s}

        changed = True
        while changed:
            changed = False
            for n in node_ids - start_ids:
                preds = self._predecessors[n]
                if not preds:  # unreachable from any start
                    continue
                intersection = set[T].intersection(*(dom[p] for p in preds)) if preds else set[T]()
                new_dom = {n} | intersection
                if new_dom != dom[n]:
                    dom[n] = new_dom
                    changed = True
        return dom

    def _immediate_dominator(self, node_id: T) -> T | None:
        """Find the immediate dominator of a node.

        The immediate dominator is the closest dominator to a node (other than itself)
        in the dominator tree.

        Args:
            node_id: The node to find the immediate dominator for.

        Returns:
            The immediate dominator's ID if one exists, None otherwise.
        """
        dom = self._dominators
        candidates = dom[node_id] - {node_id}
        for c in candidates:
            if all((c == d) or (c not in dom[d]) for d in candidates):
                return c
        return None

    def _get_upstream_nodes_if_parent(self, join_id: T, fork_id: T) -> set[T] | None:
        """Check if a fork is a valid parent and return upstream nodes.

        Tests whether the given fork can serve as a parent fork for the join by checking
        for cycles that bypass the fork. If valid, returns all nodes that can reach the
        join without going through the fork.

        Args:
            join_id: The join node being analyzed.
            fork_id: The potential parent fork to test.

        Returns:
            The set of node IDs upstream of the join (excluding the fork) if the fork is
            a valid parent, or None if a cycle exists that bypasses the fork (making it
            invalid as a parent fork).

        Note:
            If, in the graph with fork_id removed, a path exists that starts and ends at
            the join (i.e., join is on a cycle avoiding the fork), we return None because
            the fork would not be a valid "parent fork".
        """
        upstream: set[T] = set()
        stack = [join_id]
        while stack:
            v = stack.pop()
            for p in self._predecessors[v]:
                if p == fork_id:
                    continue
                if p == join_id:
                    return None  # J sits on a cycle w/out the specified node
                if p not in upstream:
                    upstream.add(p)
                    stack.append(p)
        return upstream


def main_test():
    """Run basic smoke tests to verify parent fork finding functionality.

    Tests both valid cases (where a parent fork exists) and invalid cases
    (where cycles bypass potential parent forks).
    """
    join_id = 'J'
    nodes = {'start', 'A', 'B', 'C', 'F', 'F2', 'I', 'J', 'end'}
    start_ids = {'start'}
    fork_ids = {'F', 'F2'}
    valid_edges = {
        'start': ['F2'],
        'F2': ['I'],
        'I': ['F'],
        'F': ['A', 'B'],
        'A': ['J'],
        'B': ['J'],
        'J': ['C'],
        'C': ['end', 'I'],
    }
    invalid_edges = deepcopy(valid_edges)
    invalid_edges['C'].append('A')

    print(ParentForkFinder(nodes, start_ids, fork_ids, valid_edges).find_parent_fork(join_id))
    # > DominatingFork(fork_id='F', intermediate_nodes={'A', 'B'})
    print(ParentForkFinder(nodes, start_ids, fork_ids, invalid_edges).find_parent_fork(join_id))
    # > None


if __name__ == '__main__':
    main_test()
