from __future__ import annotations as _annotations

import asyncio

import pytest
from inline_snapshot import snapshot
from pytest_mock import MockerFixture

from ..conftest import try_import

with try_import() as imports_successful:
    import logfire
    from logfire.testing import CaptureLogfire

    from pydantic_evals.otel._context_subtree import (
        context_subtree,
    )
    from pydantic_evals.otel.span_tree import SpanQuery, SpanTree

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


@pytest.fixture(autouse=True)
def use_logfire(capfire: CaptureLogfire):
    assert capfire


async def test_context_subtree_concurrent():
    """Test that context_subtree correctly records spans in independent async contexts."""

    # Create independent async tasks
    async def task1():
        with context_subtree() as tree:
            with logfire.span('task1'):
                with logfire.span('task1_child1'):
                    await asyncio.sleep(0.01)
                with logfire.span('task1_child2'):
                    await asyncio.sleep(0.01)
        return tree

    async def task2():
        with context_subtree() as tree:
            with logfire.span('task2'):
                with logfire.span('task2_child1'):
                    await asyncio.sleep(0.01)
                    with logfire.span('task2_grandchild'):
                        await asyncio.sleep(0.01)
        return tree

    # Execute tasks concurrently
    tree1, tree2 = await asyncio.gather(task1(), task2())
    assert isinstance(tree1, SpanTree)
    assert isinstance(tree2, SpanTree)

    # Verify that tree1 only contains spans from task1
    assert len(tree1.roots) == 1, 'tree1 should have exactly one root span'
    assert tree1.roots[0].name == 'task1', 'tree1 root should be task1'
    assert not tree1.any(lambda node: node.name == 'task2'), 'tree1 should not contain task2 spans'
    assert not tree1.any(lambda node: node.name == 'task2_child1'), 'tree1 should not contain task2_child1 spans'
    assert not tree1.any(lambda node: node.name == 'task2_grandchild'), (
        'tree1 should not contain task2_grandchild spans'
    )

    # Verify task1 children
    task1_root = tree1.roots[0]
    assert len(task1_root.children) == 2, 'task1 should have exactly two children'
    task1_child_names = {child.name for child in task1_root.children}
    assert task1_child_names == {
        'task1_child1',
        'task1_child2',
    }, "task1's children should be task1_child1 and task1_child2"

    # Verify that tree2 only contains spans from task2
    assert len(tree2.roots) == 1, 'tree2 should have exactly one root span'
    assert tree2.roots[0].name == 'task2', 'tree2 root should be task2'
    assert not tree2.any(lambda node: node.name == 'task1'), 'tree2 should not contain task1 spans'
    assert not tree2.any(lambda node: node.name == 'task1_child1'), 'tree2 should not contain task1_child1 spans'
    assert not tree2.any(lambda node: node.name == 'task1_child2'), 'tree2 should not contain task1_child2 spans'

    # Verify task2 structure
    task2_root = tree2.roots[0]
    assert len(task2_root.children) == 1, 'task2 should have exactly one child'
    assert task2_root.children[0].name == 'task2_child1', "task2's child should be task2_child1"

    # Verify grandchild
    task2_child = task2_root.children[0]
    assert len(task2_child.children) == 1, 'task2_child1 should have exactly one child'
    assert task2_child.children[0].name == 'task2_grandchild', "task2_child1's child should be task2_grandchild"


@pytest.fixture
async def span_tree() -> SpanTree:
    """Fixture that creates a span tree with a predefined structure and attributes."""
    # Create spans with a tree structure and attributes
    with context_subtree() as tree:
        with logfire.span('root', level='0'):
            with logfire.span('child1', level='1', type='important'):
                with logfire.span('grandchild1', level='2', type='important'):
                    pass
                with logfire.span('grandchild2', level='2', type='normal'):
                    pass
            with logfire.span('child2', level='1', type='normal'):
                with logfire.span('grandchild3', level='2', type='normal'):
                    pass
    assert isinstance(tree, SpanTree)
    return tree


async def test_span_tree_flattened(span_tree: SpanTree):
    """Test the __iter__ method of SpanTree."""
    assert len(list(span_tree)) == 6, 'Should have 6 spans in total'

    # Check that all expected nodes are in the flattened list, ordered by start_timestamp
    node_names = [node.name for node in span_tree]
    expected_names = ['root', 'child1', 'grandchild1', 'grandchild2', 'child2', 'grandchild3']
    assert node_names == expected_names


async def test_span_tree_find_all(span_tree: SpanTree):
    """Test the find_all method of SpanTree."""
    # Find nodes with important type
    important_nodes = list(span_tree.find(lambda node: node.attributes.get('type') == 'important'))
    assert len(important_nodes) == 2
    important_names = {node.name for node in important_nodes}
    assert important_names == {'child1', 'grandchild1'}

    # Find nodes with level 2
    level2_nodes = list(span_tree.find(lambda node: node.attributes.get('level') == '2'))
    assert len(level2_nodes) == 3
    level2_names = {node.name for node in level2_nodes}
    assert level2_names == {'grandchild1', 'grandchild2', 'grandchild3'}


async def test_span_tree_any(span_tree: SpanTree):
    """Test the any() method of SpanTree."""
    # Test existence of a node by name
    assert span_tree.any(lambda node: node.name == 'grandchild2')

    # Test non-existence
    assert not span_tree.any(lambda node: node.name == 'non_existent')

    # Test existence by attribute
    assert span_tree.any(lambda node: node.attributes.get('type') == 'important')


async def test_span_node_find_children(span_tree: SpanTree):
    """Test the find_children method of SpanNode."""
    root_node = span_tree.roots[0]
    assert root_node.name == 'root'

    # Find all children with a level attribute
    child_nodes = list(root_node.find_children(lambda node: 'level' in node.attributes))
    assert len(child_nodes) == 2

    # Check that the children have the expected names
    child_names = {node.name for node in child_nodes}
    assert child_names == {'child1', 'child2'}


async def test_span_node_first_child(span_tree: SpanTree):
    """Test the first_child method of SpanNode."""
    root_node = span_tree.roots[0]

    # Find first child with important type
    first_important_child = root_node.first_child(lambda node: node.attributes.get('type') == 'important')
    assert first_important_child is not None
    assert first_important_child.name == 'child1'

    # Test for non-existent attribute
    non_existent = root_node.first_child(lambda node: node.attributes.get('non_existent') == 'value')
    assert non_existent is None


async def test_span_node_any_child(span_tree: SpanTree):
    """Test the any_child method of SpanNode."""
    root_node = span_tree.roots[0]

    # Test existence of child with normal type
    assert root_node.any_child(lambda node: node.attributes.get('type') == 'normal')

    # Test non-existence
    assert not root_node.any_child(lambda node: node.name == 'non_existent')


async def test_span_node_find_descendants(span_tree: SpanTree):
    """Test the find_descendants method of SpanNode."""
    root_node = span_tree.roots[0]

    # Find all descendants with level 2
    level2_nodes = list(root_node.find_descendants(lambda node: node.attributes.get('level') == '2'))
    assert len(level2_nodes) == 3

    # Check that they have the expected names
    level2_names = {node.name for node in level2_nodes}
    assert level2_names == {'grandchild1', 'grandchild2', 'grandchild3'}

    # Test descendant counts
    assert root_node.matches({'min_descendant_count': 5, 'max_descendant_count': 5})
    assert not root_node.matches({'min_descendant_count': 4, 'max_descendant_count': 4})
    assert not root_node.matches({'min_descendant_count': 6, 'max_descendant_count': 6})

    child1_node = root_node.first_child(lambda node: node.name == 'child1')
    assert child1_node is not None
    assert child1_node.matches({'min_descendant_count': 2, 'max_descendant_count': 2})


async def test_span_node_matches(span_tree: SpanTree):
    """Test the matches method of SpanNode."""
    root_node = span_tree.roots[0]
    child1_node = root_node.first_child(lambda node: node.name == 'child1')
    assert child1_node is not None

    # Test matches by name
    assert child1_node.matches(SpanQuery(name_equals='child1'))
    assert not child1_node.matches(SpanQuery(name_equals='child2'))

    # Test matches by attributes
    assert child1_node.matches(SpanQuery(has_attributes={'level': '1', 'type': 'important'}))
    assert not child1_node.matches(SpanQuery(has_attributes={'level': '2', 'type': 'important'}))

    # Test matches by both name and attributes
    assert child1_node.matches(SpanQuery(name_equals='child1', has_attributes={'type': 'important'}))
    assert not child1_node.matches(SpanQuery(name_equals='child1', has_attributes={'type': 'normal'}))


async def test_span_tree_repr(span_tree: SpanTree):
    assert repr(SpanTree()) == snapshot('<SpanTree />')
    assert str(span_tree) == snapshot('<SpanTree num_roots=1 total_spans=6 />')
    assert repr(span_tree) == snapshot("""\
<SpanTree>
  <SpanNode name='root' >
    <SpanNode name='child1' >
      <SpanNode name='grandchild1' />
      <SpanNode name='grandchild2' />
    </SpanNode>
    <SpanNode name='child2' >
      <SpanNode name='grandchild3' />
    </SpanNode>
  </SpanNode>
</SpanTree>\
""")
    assert span_tree.repr_xml(include_children=False) == snapshot("""\
<SpanTree>
  <SpanNode name='root' children=... />
</SpanTree>\
""")
    assert span_tree.repr_xml(include_span_id=True) == snapshot("""\
<SpanTree>
  <SpanNode name='root' span_id='0000000000000001' >
    <SpanNode name='child1' span_id='0000000000000003' >
      <SpanNode name='grandchild1' span_id='0000000000000005' />
      <SpanNode name='grandchild2' span_id='0000000000000007' />
    </SpanNode>
    <SpanNode name='child2' span_id='0000000000000009' >
      <SpanNode name='grandchild3' span_id='000000000000000b' />
    </SpanNode>
  </SpanNode>
</SpanTree>\
""")
    assert span_tree.repr_xml(include_trace_id=True) == snapshot("""\
<SpanTree>
  <SpanNode name='root' trace_id='00000000000000000000000000000001' >
    <SpanNode name='child1' trace_id='00000000000000000000000000000001' >
      <SpanNode name='grandchild1' trace_id='00000000000000000000000000000001' />
      <SpanNode name='grandchild2' trace_id='00000000000000000000000000000001' />
    </SpanNode>
    <SpanNode name='child2' trace_id='00000000000000000000000000000001' >
      <SpanNode name='grandchild3' trace_id='00000000000000000000000000000001' />
    </SpanNode>
  </SpanNode>
</SpanTree>\
""")
    assert span_tree.repr_xml(include_start_timestamp=True) == snapshot("""\
<SpanTree>
  <SpanNode name='root' start_timestamp='1970-01-01T00:00:01+00:00' >
    <SpanNode name='child1' start_timestamp='1970-01-01T00:00:02+00:00' >
      <SpanNode name='grandchild1' start_timestamp='1970-01-01T00:00:03+00:00' />
      <SpanNode name='grandchild2' start_timestamp='1970-01-01T00:00:05+00:00' />
    </SpanNode>
    <SpanNode name='child2' start_timestamp='1970-01-01T00:00:08+00:00' >
      <SpanNode name='grandchild3' start_timestamp='1970-01-01T00:00:09+00:00' />
    </SpanNode>
  </SpanNode>
</SpanTree>\
""")
    assert span_tree.repr_xml(include_duration=True) == snapshot("""\
<SpanTree>
  <SpanNode name='root' duration='0:00:11' >
    <SpanNode name='child1' duration='0:00:05' >
      <SpanNode name='grandchild1' duration='0:00:01' />
      <SpanNode name='grandchild2' duration='0:00:01' />
    </SpanNode>
    <SpanNode name='child2' duration='0:00:03' >
      <SpanNode name='grandchild3' duration='0:00:01' />
    </SpanNode>
  </SpanNode>
</SpanTree>\
""")


async def test_span_node_repr(span_tree: SpanTree):
    node = span_tree.first({'name_equals': 'child2'})
    assert node is not None

    leaf_node = span_tree.first({'name_equals': 'grandchild1'})
    assert str(leaf_node) == snapshot("<SpanNode name='grandchild1' span_id='0000000000000005' />")

    assert str(node) == snapshot("<SpanNode name='child2' span_id='0000000000000009'>...</SpanNode>")
    assert repr(node) == snapshot("""\
<SpanNode name='child2' >
  <SpanNode name='grandchild3' />
</SpanNode>\
""")
    assert node.repr_xml(include_children=False) == snapshot("<SpanNode name='child2' children=... />")
    assert node.repr_xml(include_span_id=True) == snapshot("""\
<SpanNode name='child2' span_id='0000000000000009' >
  <SpanNode name='grandchild3' span_id='000000000000000b' />
</SpanNode>\
""")
    assert node.repr_xml(include_trace_id=True) == snapshot("""\
<SpanNode name='child2' trace_id='00000000000000000000000000000001' >
  <SpanNode name='grandchild3' trace_id='00000000000000000000000000000001' />
</SpanNode>\
""")
    assert node.repr_xml(include_start_timestamp=True) == snapshot("""\
<SpanNode name='child2' start_timestamp='1970-01-01T00:00:08+00:00' >
  <SpanNode name='grandchild3' start_timestamp='1970-01-01T00:00:09+00:00' />
</SpanNode>\
""")
    assert node.repr_xml(include_duration=True) == snapshot("""\
<SpanNode name='child2' duration='0:00:03' >
  <SpanNode name='grandchild3' duration='0:00:01' />
</SpanNode>\
""")


async def test_span_tree_ancestors_methods():
    """Test the ancestor traversal methods in SpanNode."""
    # Create spans with a deep structure for testing ancestor methods
    with context_subtree() as tree:
        with logfire.span('root', depth=0):
            with logfire.span('level1', depth=1):
                with logfire.span('level2', depth=2):
                    with logfire.span('level3', depth=3):
                        with logfire.span('leaf', depth=4):
                            # Add a log message to test nested logs
                            logfire.info('This is a leaf node log message')
    assert isinstance(tree, SpanTree)

    # Get the leaf node
    leaf_node = tree.first(lambda node: node.name == 'leaf')
    assert leaf_node is not None

    # Test find_ancestors
    ancestors = list(leaf_node.find_ancestors(lambda node: True))
    assert len(ancestors) == 4
    ancestor_names = [node.name for node in ancestors]
    assert ancestor_names == ['level3', 'level2', 'level1', 'root']

    # Test first_ancestor by name instead of depth comparison to avoid type issues
    level2_ancestor = leaf_node.first_ancestor(lambda node: node.name == 'level2')
    assert level2_ancestor is not None
    assert level2_ancestor.name == 'level2'

    # Test any_ancestor
    assert leaf_node.any_ancestor(lambda node: node.name == 'root')
    assert not leaf_node.any_ancestor(lambda node: node.name == 'non_existent')

    # Test ancestor query matches
    assert leaf_node.matches({'min_depth': 4, 'max_depth': 4})
    assert not leaf_node.matches({'min_depth': 3, 'max_depth': 3})
    assert not leaf_node.matches({'min_depth': 5, 'max_depth': 5})

    assert [node.name for node in leaf_node.ancestors] == ['level3', 'level2', 'level1', 'root']
    assert leaf_node.matches({'some_ancestor_has': {'name_equals': 'level1'}})
    assert not leaf_node.matches({'some_ancestor_has': {'name_equals': 'level4'}})

    assert not leaf_node.matches({'all_ancestors_have': {'name_matches_regex': 'level'}})
    assert leaf_node.matches({'all_ancestors_have': {'name_matches_regex': 'level|root'}})

    assert not leaf_node.matches({'no_ancestor_has': {'name_matches_regex': 'root'}})
    assert leaf_node.matches({'no_ancestor_has': {'name_matches_regex': 'abc'}})

    # Test stop_recursing_when:
    assert not leaf_node.matches(
        {'some_ancestor_has': {'name_equals': 'level1'}, 'stop_recursing_when': {'name_equals': 'level2'}}
    )
    assert leaf_node.matches(
        {'all_ancestors_have': {'name_matches_regex': 'level'}, 'stop_recursing_when': {'name_equals': 'level1'}}
    )
    assert leaf_node.matches(
        {'no_ancestor_has': {'name_matches_regex': 'root'}, 'stop_recursing_when': {'name_equals': 'level1'}}
    )


async def test_span_tree_descendants_methods():
    """Test the descendant traversal methods in SpanNode."""
    # Create spans with a deep structure for testing descendant methods
    with context_subtree() as tree:
        with logfire.span('root', depth=0):
            with logfire.span('level1', depth=1):
                with logfire.span('level2', depth=2):
                    with logfire.span('level3', depth=3):
                        logfire.info('leaf', depth=4)
    assert isinstance(tree, SpanTree)

    # Get the root node
    root_node = tree.roots[0]
    assert root_node.name == 'root'

    # Test find_descendants
    descendants = list(root_node.find_descendants(lambda node: True))
    assert len(descendants) == 4
    descendant_names = [node.name for node in descendants]
    assert descendant_names == ['level1', 'level2', 'level3', 'leaf']

    # Test first_descendant
    level2_descendant = root_node.first_descendant(lambda node: node.name == 'level2')
    assert level2_descendant is not None
    assert level2_descendant.name == 'level2'

    # Test any_descendant
    assert root_node.any_descendant(lambda node: node.name == 'leaf')
    assert not root_node.any_descendant(lambda node: node.name == 'non_existent')

    # Test descendant-related conditions in matches function
    # Test some_descendant_has
    assert root_node.matches({'some_descendant_has': {'name_equals': 'leaf'}})

    level2_node = root_node.first_descendant(lambda node: node.name == 'level2')
    assert level2_node is not None
    assert level2_node.matches({'some_descendant_has': {'name_equals': 'leaf'}})
    assert not level2_node.matches({'some_descendant_has': {'name_equals': 'level1'}})

    # Test all_descendants_have
    assert root_node.matches({'all_descendants_have': {'has_attribute_keys': ['depth']}})
    assert root_node.matches({'some_descendant_has': {'has_attributes': {'depth': 3}}})
    assert not root_node.matches({'all_descendants_have': {'has_attributes': {'depth': 3}}})

    # Test no_descendant_has
    no_descendant_query: SpanQuery = {'no_descendant_has': {'name_equals': 'non_existent'}}
    assert root_node.matches(no_descendant_query)

    level1_node = root_node.first_descendant(lambda node: node.name == 'level1')
    assert level1_node is not None
    assert level1_node.matches({'no_descendant_has': {'name_equals': 'level1'}})
    assert not level1_node.matches({'no_descendant_has': {'name_equals': 'level2'}})

    # Test complex descendant queries
    assert root_node.matches({'some_descendant_has': {'name_equals': 'leaf', 'has_attributes': {'depth': 4}}})

    # Test descendant queries with logical combinations
    logical_descendant_query: SpanQuery = {
        'some_descendant_has': {'and_': [{'name_contains': 'level'}, {'has_attributes': {'depth': 2}}]}
    }
    assert root_node.matches(logical_descendant_query)

    level3_node = root_node.first_descendant(lambda node: node.name == 'level3')
    assert level3_node is not None
    assert not level3_node.matches(logical_descendant_query)

    # Test descendant queries with negation
    negated_descendant_query: SpanQuery = {'no_descendant_has': {'not_': {'has_attributes': {'depth': 4}}}}
    assert not root_node.matches(negated_descendant_query)  # Should fail because level3 has depth=3

    leaf_node = root_node.first_descendant(lambda node: node.name == 'leaf')
    assert leaf_node is not None
    assert leaf_node.matches(negated_descendant_query)
    assert leaf_node.matches({'no_descendant_has': {'has_attributes': {'depth': 4}}})

    # Test stop_recursing_when:
    assert not root_node.matches(
        {'some_descendant_has': {'name_equals': 'leaf'}, 'stop_recursing_when': {'name_equals': 'level2'}}
    )
    assert root_node.matches(
        {'all_descendants_have': {'has_attribute_keys': ['depth']}, 'stop_recursing_when': {'name_equals': 'level2'}}
    )
    assert root_node.matches(
        {'no_descendant_has': {'name_equals': 'leaf'}, 'stop_recursing_when': {'name_equals': 'level3'}}
    )


async def test_log_levels_and_exceptions():
    """Test recording different log levels and exceptions in spans."""
    with context_subtree() as tree:
        # Test different log levels
        with logfire.span('parent_span'):
            logfire.debug('Debug message')
            logfire.info('Info message')
            logfire.warn('Warning message')

            # Create child span with error
            with logfire.span('error_child') as error_span:
                logfire.error('Error occurred')
                # Record exception
                try:
                    raise ValueError('Test exception')
                except ValueError as e:
                    error_span.record_exception(e)
    assert isinstance(tree, SpanTree)

    # Verify log levels are preserved
    parent_span = tree.first(lambda node: node.name == 'parent_span')
    assert parent_span is not None

    # Find the error child span
    error_child = parent_span.first_child(lambda node: node.name == 'error_child')
    assert error_child is not None

    # Verify attributes reflect log levels and exceptions
    log_nodes = list(
        parent_span.find_descendants(
            lambda node: 'Debug message' in str(node.attributes)
            or 'Info message' in str(node.attributes)
            or 'Warning message' in str(node.attributes)
            or 'Error occurred' in str(node.attributes)
        )
    )
    assert len(log_nodes) > 0, 'Should have log messages as spans'


async def test_span_query_basics(span_tree: SpanTree):
    """Test basic SpanQuery conditions on a span tree."""
    # Test name equality condition
    name_equals_query: SpanQuery = {'name_equals': 'child1'}
    matched_node = span_tree.first(name_equals_query)
    assert matched_node is not None
    assert matched_node.name == 'child1'

    # Test name contains condition
    name_contains_query: SpanQuery = {'name_contains': 'child'}
    matched_nodes = list(span_tree.find(name_contains_query))
    assert len(matched_nodes) == 5  # All nodes with "child" in name
    assert all('child' in node.name for node in matched_nodes)

    # Test name regex match condition
    name_regex_query: SpanQuery = {'name_matches_regex': r'^grand.*\d$'}
    matched_nodes = list(span_tree.find(name_regex_query))
    assert len(matched_nodes) == 3  # All grandchild nodes
    assert all(node.name.startswith('grand') and node.name[-1].isdigit() for node in matched_nodes)

    # Test has_attributes condition
    attr_query: SpanQuery = {'has_attributes': {'level': '1', 'type': 'important'}}
    matched_node = span_tree.first(attr_query)
    assert matched_node is not None
    assert matched_node.name == 'child1'
    assert matched_node.attributes.get('level') == '1'
    assert matched_node.attributes.get('type') == 'important'

    # Test has_attribute_keys condition
    attr_keys_query: SpanQuery = {'has_attribute_keys': ['level', 'type']}
    matched_nodes = list(span_tree.find(attr_keys_query))
    assert len(matched_nodes) == 5  # All nodes except root have both keys
    assert all('level' in node.attributes and 'type' in node.attributes for node in matched_nodes)


async def test_span_query_negation():
    """Test negation in SpanQuery."""

    # Create a simple tree for testing negation
    with context_subtree() as tree:
        with logfire.span('parent', category='main'):
            with logfire.span('child1', category='important'):
                pass
            with logfire.span('child2', category='normal'):
                pass
    assert isinstance(tree, SpanTree)

    # Test negation of name attribute
    not_query: SpanQuery = {'not_': {'name_equals': 'child1'}}
    matched_nodes = list(tree.find(not_query))
    assert len(matched_nodes) == 2
    assert all(node.name != 'child1' for node in matched_nodes)

    # Test negation of attribute condition
    not_attr_query: SpanQuery = {'not_': {'has_attributes': {'category': 'important'}}}
    matched_nodes = list(tree.find(not_attr_query))
    assert len(matched_nodes) == 2
    assert all(node.attributes.get('category') != 'important' for node in matched_nodes)

    # Test direct negation using the matches function
    parent_node = tree.first(lambda node: node.name == 'parent')
    assert parent_node is not None

    assert parent_node.matches({'name_equals': 'parent'})
    assert not parent_node.matches({'not_': {'name_equals': 'parent'}})


async def test_span_query_logical_combinations():
    """Test logical combinations (AND/OR) in SpanQuery."""

    with context_subtree() as tree:
        with logfire.span('root1', level='0'):
            with logfire.span('child1', level='1', category='important'):
                pass
            with logfire.span('child2', level='1', category='normal'):
                pass
            with logfire.span('special', level='1', category='important', priority='high'):
                pass
    assert isinstance(tree, SpanTree)

    # Test AND logic
    and_query: SpanQuery = {'and_': [{'name_contains': '1'}, {'has_attributes': {'level': '1'}}]}
    matched_nodes = list(tree.find(and_query))
    assert len(matched_nodes) == 1, matched_nodes
    assert all(node.name in ['child1'] for node in matched_nodes)

    # Test OR logic
    or_query: SpanQuery = {'or_': [{'name_contains': '2'}, {'has_attributes': {'level': '0'}}]}
    matched_nodes = list(tree.find(or_query))
    assert len(matched_nodes) == 2
    assert any(node.name == 'child2' for node in matched_nodes)
    assert any(node.attributes.get('level') == '0' for node in matched_nodes)

    # Test complex combination (AND + OR)
    complex_query: SpanQuery = {
        'and_': [
            {'has_attributes': {'level': '1'}},
            {'or_': [{'has_attributes': {'category': 'important'}}, {'name_equals': 'child2'}]},
        ]
    }
    matched_nodes = list(tree.find(complex_query))
    assert len(matched_nodes) == 3  # child1, child2, special
    matched_names = [node.name for node in matched_nodes]
    assert set(matched_names) == {'child1', 'child2', 'special'}


async def test_span_query_timing_conditions():
    """Test timing-related conditions in SpanQuery."""
    from datetime import timedelta

    with context_subtree() as tree:
        with logfire.span('fast_operation'):
            pass

        with logfire.span('medium_operation'):
            logfire.info('add a wait')

        with logfire.span('slow_operation'):
            logfire.info('add a wait')
            logfire.info('add a wait')
    assert isinstance(tree, SpanTree)

    durations = sorted([node.duration for node in tree if node.duration > timedelta(seconds=0)])
    fast_threshold = (durations[0] + durations[1]) / 2
    medium_threshold = (durations[1] + durations[2]) / 2

    # Test min_duration
    min_duration_query: SpanQuery = {'min_duration': fast_threshold}
    matched_nodes = list(tree.find(min_duration_query))
    assert len(matched_nodes) == 2
    assert 'fast_operation' not in [node.name for node in matched_nodes]

    # Test max_duration
    max_duration_queries: list[SpanQuery] = [
        {'min_duration': 0.001, 'max_duration': medium_threshold},
        {'min_duration': 0.001, 'max_duration': medium_threshold.seconds},
    ]
    for max_duration_query in max_duration_queries:
        matched_nodes = list(tree.find(max_duration_query))
        assert len(matched_nodes) == 2
        assert 'slow_operation' not in [node.name for node in matched_nodes]

    # Test min and max duration together using timedelta
    duration_range_query: SpanQuery = {
        'min_duration': fast_threshold,
        'max_duration': medium_threshold,
    }
    matched_node = tree.first(duration_range_query)
    assert matched_node is not None
    assert matched_node.name == 'medium_operation'


async def test_span_query_descendant_conditions():
    """Test descendant-related conditions in SpanQuery."""

    with context_subtree() as tree:
        with logfire.span('parent1'):
            with logfire.span('child1', type='important'):
                pass
            with logfire.span('child2', type='normal'):
                pass

        with logfire.span('parent2'):
            with logfire.span('child3', type='normal'):
                pass
            with logfire.span('child4', type='normal'):
                pass
    assert isinstance(tree, SpanTree)

    # Test some_child_has condition
    some_child_query: SpanQuery = {'some_child_has': {'has_attributes': {'type': 'important'}}}
    matched_node = tree.first(some_child_query)
    assert matched_node is not None
    assert matched_node.name == 'parent1'

    # Test all_children_have condition
    all_children_query: SpanQuery = {'all_children_have': {'has_attributes': {'type': 'normal'}}, 'min_child_count': 1}
    matched_node = tree.first(all_children_query)
    assert matched_node is not None
    assert matched_node.name == 'parent2'
    # A couple more tests for coverage reasons:
    assert tree.first({'all_children_have': {'has_attributes': {'type': 'unusual'}}, 'min_child_count': 1}) is None
    assert not matched_node.matches({'no_child_has': {'has_attributes': {'type': 'normal'}}})

    # Test no_child_has condition
    no_child_query: SpanQuery = {'no_child_has': {'has_attributes': {'type': 'important'}}, 'min_child_count': 1}
    matched_node = tree.first(no_child_query)
    assert matched_node is not None
    assert matched_node.name == 'parent2'


async def test_span_query_complex_hierarchical_conditions():
    """Test complex hierarchical queries with nested structures."""

    with context_subtree() as tree:
        with logfire.span('app', service='web'):
            with logfire.span('request', method='GET', path='/api/v1/users'):
                with logfire.span('db_query', table='users'):
                    pass
                with logfire.span('cache_lookup', cache='redis'):
                    pass
            with logfire.span('request', method='POST', path='/api/v1/users'):
                with logfire.span('db_query', table='users'):
                    pass
                with logfire.span('notification', channel='email'):
                    pass
    assert isinstance(tree, SpanTree)

    # Find the app span that has a POST request with a notification child
    complex_query: SpanQuery = {
        'name_equals': 'app',
        'some_child_has': {
            'name_equals': 'request',
            'has_attributes': {'method': 'POST'},
            'some_child_has': {'name_equals': 'notification'},
        },
    }

    matched_node = tree.first(complex_query)
    assert matched_node is not None
    assert matched_node.name == 'app'

    # Find request spans with both db_query and another operation
    request_with_db_and_other: SpanQuery = {
        'name_equals': 'request',
        'some_child_has': {'not_': {'name_equals': 'db_query'}},
    }

    matched_nodes = list(tree.find(request_with_db_and_other))
    assert len(matched_nodes) == 2  # Both requests have db_query and another operation


async def test_matches_function_directly():
    """Test the matches function directly with various SpanQuery combinations."""

    # Create a test span tree
    with context_subtree() as tree:
        with logfire.span('parent', level='1', category='main'):
            with logfire.span('child1', level='2', category='important'):
                pass
            with logfire.span('child2', level='2', category='normal'):
                pass
    assert isinstance(tree, SpanTree)

    parent_node = tree.roots[0]
    child1_node = parent_node.children[0]
    child2_node = parent_node.children[1]

    # Basic matches tests
    assert parent_node.matches({'name_equals': 'parent'})
    assert not child1_node.matches({'name_equals': 'parent'})

    # Test attribute matching
    assert parent_node.matches({'has_attributes': {'level': '1'}})
    assert not child1_node.matches({'has_attributes': {'level': '1'}})

    # Test logical combinations
    complex_query: SpanQuery = {'and_': [{'name_equals': 'child1'}, {'has_attributes': {'category': 'important'}}]}
    assert child1_node.matches(complex_query)
    assert not child2_node.matches(complex_query)

    # Test with descendants
    descendant_query: SpanQuery = {'some_child_has': {'name_equals': 'child1'}}
    assert parent_node.matches(descendant_query)
    assert not child1_node.matches(descendant_query)


async def test_span_query_child_count():
    """Test min_child_count and max_child_count conditions in SpanQuery."""

    # Create a tree with varying numbers of children
    with context_subtree() as tree:
        with logfire.span('parent_no_children'):
            pass

        with logfire.span('parent_one_child'):
            with logfire.span('child1'):
                pass

        with logfire.span('parent_two_children'):
            with logfire.span('child2'):
                pass
            with logfire.span('child3'):
                pass

        with logfire.span('parent_three_children'):
            with logfire.span('child4'):
                pass
            with logfire.span('child5'):
                pass
            with logfire.span('child6'):
                pass
    assert isinstance(tree, SpanTree)

    # Test min_child_count
    min_2_query: SpanQuery = {'min_child_count': 2}
    matched_nodes = list(tree.find(min_2_query))
    assert len(matched_nodes) == 2
    matched_names = {node.name for node in matched_nodes}
    assert matched_names == {'parent_two_children', 'parent_three_children'}

    # Test max_child_count
    max_1_query: SpanQuery = {'max_child_count': 1}
    matched_nodes = list(tree.find(max_1_query))
    assert len(matched_nodes) == 8  # parent_no_children, parent_one_child, and all the leaf nodes
    assert 'parent_two_children' not in {node.name for node in matched_nodes}
    assert 'parent_three_children' not in {node.name for node in matched_nodes}

    # Test both min and max together (range)
    child_range_query: SpanQuery = {'min_child_count': 1, 'max_child_count': 2}
    matched_nodes = list(tree.find(child_range_query))
    assert len(matched_nodes) == 2
    matched_names = {node.name for node in matched_nodes}
    assert matched_names == {'parent_one_child', 'parent_two_children'}

    # Test with other conditions
    complex_query: SpanQuery = {'name_contains': 'parent', 'min_child_count': 2}
    matched_nodes = list(tree.find(complex_query))
    assert len(matched_nodes) == 2
    assert all('parent' in node.name and len(node.children) >= 2 for node in matched_nodes)

    # Test direct usage of matches function
    parent_three = tree.first(lambda node: node.name == 'parent_three_children')
    assert parent_three is not None

    assert parent_three.matches({'min_child_count': 3})
    assert parent_three.matches({'min_child_count': 2, 'max_child_count': 3})
    assert not parent_three.matches({'max_child_count': 2})

    # Test with logical operators
    logical_query: SpanQuery = {
        'and_': [{'name_contains': 'parent'}, {'min_child_count': 1}],
        'not_': {'max_child_count': 1},
    }
    matched_nodes = list(tree.find(logical_query))
    assert len(matched_nodes) == 2
    matched_names = {node.name for node in matched_nodes}
    assert matched_names == {'parent_two_children', 'parent_three_children'}


async def test_or_cannot_be_mixed(span_tree: SpanTree):
    with pytest.raises(ValueError) as exc_info:
        span_tree.first({'name_equals': 'child1', 'or_': [SpanQuery(name_equals='child2')]})
    assert str(exc_info.value) == snapshot("Cannot combine 'or_' conditions with other conditions at the same level")


async def test_context_subtree_invalid_tracer_provider(mocker: MockerFixture):
    """Test that context_subtree correctly records spans in independent async contexts."""
    # from opentelemetry import trace

    mocker.patch('pydantic_evals.otel._context_in_memory_span_exporter.get_tracer_provider', return_value=None)
    with pytest.raises(TypeError) as exc_info:
        with context_subtree():
            pass
    assert str(exc_info.value) == snapshot(
        "Expected `tracer_provider` to have an `add_span_processor` method; got an instance of <class 'NoneType'>. For help resolving this, please create an issue at https://github.com/pydantic/pydantic-ai/issues."
    )


async def test_context_subtree_not_configured(mocker: MockerFixture):
    """Test that context_subtree correctly records spans in independent async contexts."""
    from opentelemetry.trace import ProxyTracerProvider

    mocker.patch(
        'pydantic_evals.otel._context_in_memory_span_exporter.get_tracer_provider', return_value=ProxyTracerProvider()
    )
    with context_subtree() as span_tree:
        pass
    assert str(span_tree) == snapshot(
        'To make use of the `span_tree` in an evaluator, you need to call '
        '`logfire.configure(...)` before running an evaluation. For more information, '
        'refer to the documentation at '
        'https://ai.pydantic.dev/evals/#opentelemetry-integration.'
    )
