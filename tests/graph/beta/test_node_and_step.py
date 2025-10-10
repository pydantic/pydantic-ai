"""Tests for node and step primitives."""

from typing import Any

from pydantic_graph.beta.decision import Decision
from pydantic_graph.beta.id_types import NodeID
from pydantic_graph.beta.node import EndNode, StartNode
from pydantic_graph.beta.node_types import is_destination, is_source
from pydantic_graph.beta.step import Step, StepContext


def test_step_context_repr():
    """Test StepContext.__repr__ method."""
    ctx = StepContext(state=None, deps=None, inputs=42)
    repr_str = repr(ctx)
    assert 'StepContext' in repr_str
    assert 'inputs=42' in repr_str


def test_start_node_id():
    """Test that StartNode has the correct ID."""
    start = StartNode[int]()
    assert start.id == '__start__'


def test_end_node_id():
    """Test that EndNode has the correct ID."""
    end = EndNode[int]()
    assert end.id == '__end__'


def test_is_source_type_guard():
    """Test is_source type guard function."""

    # Test with StartNode
    start = StartNode[int]()
    assert is_source(start)

    # Test with Step
    async def my_step(ctx: StepContext[Any, Any, Any]):
        return 42  # pragma: no cover

    step = Step[None, None, None, int](id=NodeID('test'), call=my_step)
    assert is_source(step)

    # Test with EndNode (should be False)
    end = EndNode[int]()
    assert not is_source(end)


def test_is_destination_type_guard():
    """Test is_destination type guard function."""
    # Test with EndNode
    end = EndNode[int]()
    assert is_destination(end)

    # Test with Step
    async def my_step(ctx: StepContext[Any, Any, Any]):
        return 42  # pragma: no cover

    step = Step[None, None, None, int](id=NodeID('test'), call=my_step)
    assert is_destination(step)

    # Test with Decision
    decision = Decision[None, None, int](id=NodeID('test_decision'), branches=[], note=None)
    assert is_destination(decision)

    # Test with StartNode (should be False)
    start = StartNode[int]()
    assert not is_destination(start)
