"""Tests for parent fork identification and dominator analysis."""

import pytest
from inline_snapshot import snapshot

from pydantic_graph.beta.parent_forks import ParentForkFinder
from pydantic_graph.exceptions import GraphBuildingError


def test_parent_fork_basic():
    """Test basic parent fork identification."""
    join_id = 'J'
    nodes = {'start', 'F', 'A', 'B', 'J', 'end'}
    start_ids = {'start'}
    fork_ids = {'F'}
    edges = {
        'start': ['F'],
        'F': ['A', 'B'],
        'A': ['J'],
        'B': ['J'],
        'J': ['end'],
    }

    finder = ParentForkFinder(nodes, start_ids, fork_ids, edges)
    parent_fork = finder.find_parent_fork(join_id)

    assert parent_fork is not None
    assert parent_fork.fork_id == 'F'
    assert 'A' in parent_fork.intermediate_nodes
    assert 'B' in parent_fork.intermediate_nodes


def test_parent_fork_with_cycle():
    """Test parent fork identification when there's a cycle bypassing the fork."""
    join_id = 'J'
    nodes = {'start', 'F', 'A', 'B', 'C', 'J', 'end'}
    start_ids = {'start'}
    fork_ids = {'F'}
    # C creates a cycle back to A, bypassing F
    edges = {
        'start': ['F'],
        'F': ['A', 'B'],
        'A': ['J'],
        'B': ['J'],
        'J': ['C'],
        'C': ['A'],  # Cycle that bypasses F
    }

    finder = ParentForkFinder(nodes, start_ids, fork_ids, edges)
    parent_fork = finder.find_parent_fork(join_id)

    # Should return None because J sits on a cycle avoiding F
    assert parent_fork is None


def test_parent_fork_nested_forks():
    """Test parent fork identification with nested forks.

    In this case, it should return the most ancestral valid parent fork.
    """
    join_id = 'J'
    nodes = {'start', 'F1', 'F2', 'A', 'B', 'C', 'J', 'end'}
    start_ids = {'start'}
    fork_ids = {'F1', 'F2'}
    edges = {
        'start': ['F1'],
        'F1': ['F2'],
        'F2': ['A', 'B'],
        'A': ['J'],
        'B': ['J'],
        'J': ['end'],
    }

    finder = ParentForkFinder(nodes, start_ids, fork_ids, edges)
    parent_fork = finder.find_parent_fork(join_id)

    assert parent_fork is not None
    # Should find F1 as the most ancestral parent fork
    assert parent_fork.fork_id == 'F1'


def test_parent_fork_parallel_nested_forks():
    """Test parent fork identification with nested forks.

    This test is mostly included to document the current behavior, which is always to use the most ancestral
    valid fork, even if the most ancestral fork isn't guaranteed to pass through the specified join, and another
    fork is.

    We might want to change this behavior at some point, but if we do, we'll probably want to do so in some sort
    of user-specified way to ensure we don't break user code.
    """
    nodes = {'start', 'F1', 'F2-A', 'F2-B', 'A1', 'A2', 'B1', 'B2', 'C', 'J-A', 'J-B', 'J', 'end'}
    start_ids = {'start'}
    fork_ids = {'F1', 'F2A', 'F2B'}
    edges = {
        'start': ['F1'],
        'F1': ['F2-A', 'F2-B'],
        'F2-A': ['A1', 'A2'],
        'F2-B': ['B1', 'B2'],
        'A1': ['J-A'],
        'A2': ['J-A'],
        'B1': ['J-B'],
        'B2': ['J-B'],
        'J-A': ['J'],
        'J-B': ['J'],
        'J': ['end'],
    }

    finder = ParentForkFinder(nodes, start_ids, fork_ids, edges)
    parent_fork_ids = [
        finder.find_parent_fork(join_id).fork_id  # pyright: ignore[reportOptionalMemberAccess]
        for join_id in ['J-A', 'J-B', 'J']
    ]
    assert parent_fork_ids == snapshot(['F1', 'F1', 'F1'])  # NOT: ['F2-A', 'F2-B', 'F1'] as one might suspect


def test_parent_fork_no_forks():
    """Test parent fork identification when there are no forks."""
    join_id = 'J'
    nodes = {'start', 'A', 'B', 'J', 'end'}
    start_ids = {'start'}
    fork_ids = set[str]()
    edges = {
        'start': ['A'],
        'A': ['B'],
        'B': ['J'],
        'J': ['end'],
    }

    finder = ParentForkFinder(nodes, start_ids, fork_ids, edges)
    parent_fork = finder.find_parent_fork(join_id)

    assert parent_fork is None


def test_parent_fork_unreachable_join():
    """Test parent fork identification when join is unreachable from start."""
    join_id = 'J'
    nodes = {'start', 'F', 'A', 'B', 'J', 'end'}
    start_ids = {'start'}
    fork_ids = {'F'}
    # J is not reachable from start
    edges = {
        'start': ['end'],
        'F': ['A', 'B'],
        'A': ['J'],
        'B': ['J'],
    }

    finder = ParentForkFinder(nodes, start_ids, fork_ids, edges)
    parent_fork = finder.find_parent_fork(join_id)

    # Should return None or a parent fork with no intermediate nodes
    assert parent_fork is None or len(parent_fork.intermediate_nodes) == 0


def test_parent_fork_self_loop():
    """Test parent fork identification with a self-loop at the join."""
    join_id = 'J'
    nodes = {'start', 'F', 'A', 'B', 'J', 'end'}
    start_ids = {'start'}
    fork_ids = {'F'}
    edges = {
        'start': ['F'],
        'F': ['A', 'B'],
        'A': ['J'],
        'B': ['J'],
        'J': ['J', 'end'],  # Self-loop
    }

    finder = ParentForkFinder(nodes, start_ids, fork_ids, edges)
    parent_fork = finder.find_parent_fork(join_id)

    # Self-loop means J is on a cycle avoiding F
    assert parent_fork is None


def test_parent_fork_multiple_paths_to_fork():
    """Test parent fork with multiple paths from start to the fork."""
    join_id = 'J'
    nodes = {'start1', 'start2', 'F', 'A', 'B', 'J', 'end'}
    start_ids = {'start1', 'start2'}
    fork_ids = {'F'}
    edges = {
        'start1': ['F'],
        'start2': ['F'],
        'F': ['A', 'B'],
        'A': ['J'],
        'B': ['J'],
        'J': ['end'],
    }

    finder = ParentForkFinder(nodes, start_ids, fork_ids, edges)
    parent_fork = finder.find_parent_fork(join_id)

    assert parent_fork is not None
    assert parent_fork.fork_id == 'F'


def test_parent_fork_complex_intermediate_nodes():
    """Test parent fork with complex intermediate node structure."""
    join_id = 'J'
    nodes = {'start', 'F', 'A1', 'A2', 'B1', 'B2', 'J', 'end'}
    start_ids = {'start'}
    fork_ids = {'F'}
    edges = {
        'start': ['F'],
        'F': ['A1', 'B1'],
        'A1': ['A2'],
        'A2': ['J'],
        'B1': ['B2'],
        'B2': ['J'],
        'J': ['end'],
    }

    finder = ParentForkFinder(nodes, start_ids, fork_ids, edges)
    parent_fork = finder.find_parent_fork(join_id)

    assert parent_fork is not None
    assert parent_fork.fork_id == 'F'
    # All intermediate nodes between F and J
    assert 'A1' in parent_fork.intermediate_nodes
    assert 'A2' in parent_fork.intermediate_nodes
    assert 'B1' in parent_fork.intermediate_nodes
    assert 'B2' in parent_fork.intermediate_nodes


def test_parent_fork_early_return_on_ancestor_with_cycle():
    """Test early return when encountering ancestor fork with cycle."""
    join_id = 'J'
    nodes = {'start', 'F1', 'F2', 'A', 'B', 'C', 'J', 'end'}
    start_ids = {'start'}
    fork_ids = {'F1', 'F2'}
    edges = {
        'start': ['F1'],
        'F1': ['F2', 'C'],  # F1 has two paths
        'F2': ['A', 'B'],  # F2 is the inner fork
        'A': ['J'],
        'B': ['J'],
        'J': ['end'],
        'C': ['J'],  # C creates a path from F1 to J but doesn't bypass it
    }

    finder = ParentForkFinder(nodes, start_ids, fork_ids, edges)
    parent_fork = finder.find_parent_fork(join_id)

    assert parent_fork is not None
    # Returns F1 as the most ancestral valid fork
    assert parent_fork.fork_id == 'F1'


def test_parent_fork_explicit_fail_with_cycle():
    join_id = 'J'
    nodes = {'start', 'F', 'A', 'B', 'J', 'end'}
    start_ids = {'start'}
    fork_ids = {'F'}
    edges = {
        'start': ['F'],
        'F': ['J'],  # F1 has two paths
        'J': ['A', 'B'],  # F2 is the inner fork
        'A': ['J'],
        'B': ['end'],
    }

    finder = ParentForkFinder(nodes, start_ids, fork_ids, edges)
    parent_fork = finder.find_parent_fork(join_id)
    assert parent_fork is None

    with pytest.raises(
        GraphBuildingError,
        match="There is a cycle in the graph passing through 'J' that does not include 'F'. Parent forks of a join must be a part of any cycles involving that join.",
    ):
        finder.find_parent_fork(join_id, parent_fork_id='F')


def test_parent_fork_ancestor_fork_with_cycle():
    """Test early return when ancestor fork has cycle but descendant fork is valid.

    This test covers the case where:
    - F2 is a valid parent fork (part of the cycle, so skipped during backwards walk)
    - F1 is an ancestor of F2 but invalid (cycle to J bypasses F1)
    - Should return F2 as the parent fork when walking up the dominator chain
    """
    join_id = 'J'
    nodes = {'start', 'F1', 'F2', 'A', 'J', 'end'}
    start_ids = {'start'}
    fork_ids = {'F1', 'F2'}
    # J -> F2 creates a cycle, but F2 is part of it so it's valid.
    # F1 is an ancestor but the cycle bypasses it.
    edges = {
        'start': ['F1'],
        'F1': ['F2'],
        'F2': ['A'],
        'A': ['J'],
        'J': ['F2', 'end'],  # Cycle back to F2
    }

    finder = ParentForkFinder(nodes, start_ids, fork_ids, edges)
    parent_fork = finder.find_parent_fork(join_id)

    # Should find F2 as valid parent, then hit F1 which has a cycle,
    # and return F2 (hitting the early return path with assert False)
    assert parent_fork is not None
    assert parent_fork.fork_id == 'F2'
