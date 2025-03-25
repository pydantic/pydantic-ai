from __future__ import annotations as _annotations

import asyncio

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    import logfire
    from logfire.testing import CaptureLogfire

    from pydantic_evals.otel._context_in_memory_span_exporter import (
        context_subtree,
    )
    from pydantic_evals.otel.span_tree import SpanTree

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
    assert task1_child_names == {'task1_child1', 'task1_child2'}, (
        "task1's children should be task1_child1 and task1_child2"
    )

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
    """Test the flattened() method of SpanTree."""
    flattened_nodes = span_tree.flattened()
    assert len(flattened_nodes) == 6, 'Should have 6 spans in total'

    # Check that all expected nodes are in the flattened list
    node_names = {node.name for node in flattened_nodes}
    expected_names = {'root', 'child1', 'child2', 'grandchild1', 'grandchild2', 'grandchild3'}
    assert node_names == expected_names


async def test_span_tree_find_all(span_tree: SpanTree):
    """Test the find_all method of SpanTree."""
    # Find nodes with important type
    important_nodes = span_tree.find_all(lambda node: node.attributes.get('type') == 'important')
    assert len(important_nodes) == 2
    important_names = {node.name for node in important_nodes}
    assert important_names == {'child1', 'grandchild1'}

    # Find nodes with level 2
    level2_nodes = span_tree.find_all(lambda node: node.attributes.get('level') == '2')
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
    child_nodes = root_node.find_children(lambda node: 'level' in node.attributes)
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
    level2_nodes = root_node.find_descendants(lambda node: node.attributes.get('level') == '2')
    assert len(level2_nodes) == 3

    # Check that they have the expected names
    level2_names = {node.name for node in level2_nodes}
    assert level2_names == {'grandchild1', 'grandchild2', 'grandchild3'}


async def test_span_node_matches(span_tree: SpanTree):
    """Test the matches method of SpanNode."""
    root_node = span_tree.roots[0]
    child1_node = root_node.first_child(lambda node: node.name == 'child1')
    assert child1_node is not None

    # Test matches by name
    assert child1_node.matches(name='child1')
    assert not child1_node.matches(name='child2')

    # Test matches by attributes
    assert child1_node.matches(attributes={'level': '1', 'type': 'important'})
    assert not child1_node.matches(attributes={'level': '2', 'type': 'important'})

    # Test matches by both name and attributes
    assert child1_node.matches(name='child1', attributes={'type': 'important'})
    assert not child1_node.matches(name='child1', attributes={'type': 'normal'})


async def test_span_tree_ancestors_methods():
    """Test the ancestor traversal methods in SpanNode."""
    # Configure logfire
    logfire.configure()

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
    leaf_node = tree.find_first(lambda node: node.name == 'leaf')
    assert leaf_node is not None

    # Test find_ancestors
    ancestors = leaf_node.find_ancestors(lambda node: True)
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


async def test_log_levels_and_exceptions():
    """Test recording different log levels and exceptions in spans."""
    # Configure logfire
    logfire.configure()

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
    parent_span = tree.find_first(lambda node: node.name == 'parent_span')
    assert parent_span is not None

    # Find the error child span
    error_child = parent_span.first_child(lambda node: node.name == 'error_child')
    assert error_child is not None

    # Verify attributes reflect log levels and exceptions
    log_nodes = parent_span.find_descendants(
        lambda node: 'Debug message' in str(node.attributes)
        or 'Info message' in str(node.attributes)
        or 'Warning message' in str(node.attributes)
        or 'Error occurred' in str(node.attributes)
    )
    assert len(log_nodes) > 0, 'Should have log messages as spans'


async def test_span_query_basics(span_tree: SpanTree):
    """Test basic SpanQuery conditions on a span tree."""
    from pydantic_evals.otel.span_tree import SpanQuery, as_predicate

    # Test name equality condition
    name_equals_query: SpanQuery = {'name_equals': 'child1'}
    matched_node = span_tree.find_first(as_predicate(name_equals_query))
    assert matched_node is not None
    assert matched_node.name == 'child1'

    # Test name contains condition
    name_contains_query: SpanQuery = {'name_contains': 'child'}
    matched_nodes = span_tree.find_all(as_predicate(name_contains_query))
    assert len(matched_nodes) == 5  # All nodes with "child" in name
    assert all('child' in node.name for node in matched_nodes)

    # Test name regex match condition
    name_regex_query: SpanQuery = {'name_matches_regex': r'^grand.*\d$'}
    matched_nodes = span_tree.find_all(as_predicate(name_regex_query))
    assert len(matched_nodes) == 3  # All grandchild nodes
    assert all(node.name.startswith('grand') and node.name[-1].isdigit() for node in matched_nodes)

    # Test has_attributes condition
    attr_query: SpanQuery = {'has_attributes': {'level': '1', 'type': 'important'}}
    matched_node = span_tree.find_first(as_predicate(attr_query))
    assert matched_node is not None
    assert matched_node.name == 'child1'
    assert matched_node.attributes.get('level') == '1'
    assert matched_node.attributes.get('type') == 'important'

    # Test has_attribute_keys condition
    attr_keys_query: SpanQuery = {'has_attribute_keys': ['level', 'type']}
    matched_nodes = span_tree.find_all(as_predicate(attr_keys_query))
    assert len(matched_nodes) == 5  # All nodes except root have both keys
    assert all('level' in node.attributes and 'type' in node.attributes for node in matched_nodes)


async def test_span_query_negation():
    """Test negation in SpanQuery."""
    from pydantic_evals.otel.span_tree import SpanQuery, as_predicate, matches

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
    matched_nodes = tree.find_all(as_predicate(not_query))
    assert len(matched_nodes) == 2
    assert all(node.name != 'child1' for node in matched_nodes)

    # Test negation of attribute condition
    not_attr_query: SpanQuery = {'not_': {'has_attributes': {'category': 'important'}}}
    matched_nodes = tree.find_all(as_predicate(not_attr_query))
    assert len(matched_nodes) == 2
    assert all(node.attributes.get('category') != 'important' for node in matched_nodes)

    # Test direct negation using the matches function
    parent_node = tree.find_first(lambda node: node.name == 'parent')
    assert parent_node is not None

    assert matches({'name_equals': 'parent'}, parent_node)
    assert not matches({'not_': {'name_equals': 'parent'}}, parent_node)


async def test_span_query_logical_combinations():
    """Test logical combinations (AND/OR) in SpanQuery."""
    from pydantic_evals.otel.span_tree import SpanQuery, as_predicate

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
    matched_nodes = tree.find_all(as_predicate(and_query))
    assert len(matched_nodes) == 1, matched_nodes
    assert all(node.name in ['child1'] for node in matched_nodes)

    # Test OR logic
    or_query: SpanQuery = {'or_': [{'name_contains': '2'}, {'has_attributes': {'level': '0'}}]}
    matched_nodes = tree.find_all(as_predicate(or_query))
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
    matched_nodes = tree.find_all(as_predicate(complex_query))
    assert len(matched_nodes) == 3  # child1, child2, special
    matched_names = [node.name for node in matched_nodes]
    assert set(matched_names) == {'child1', 'child2', 'special'}


async def test_span_query_timing_conditions():
    """Test timing-related conditions in SpanQuery."""
    from datetime import timedelta

    from pydantic_evals.otel.span_tree import SpanQuery, as_predicate

    with context_subtree() as tree:
        with logfire.span('fast_operation'):
            pass

        with logfire.span('medium_operation'):
            logfire.info('add a wait')

        with logfire.span('slow_operation'):
            logfire.info('add a wait')
            logfire.info('add a wait')
    assert isinstance(tree, SpanTree)

    durations = sorted([node.duration for node in tree.flattened() if node.duration > timedelta(seconds=0)])
    fast_threshold = (durations[0] + durations[1]) / 2
    medium_threshold = (durations[1] + durations[2]) / 2

    # Test min_duration
    min_duration_query: SpanQuery = {'min_duration': fast_threshold}
    matched_nodes = tree.find_all(as_predicate(min_duration_query))
    assert len(matched_nodes) == 2
    assert 'fast_operation' not in [node.name for node in matched_nodes]

    # Test max_duration
    max_duration_query: SpanQuery = {'min_duration': 0.001, 'max_duration': medium_threshold}
    matched_nodes = tree.find_all(as_predicate(max_duration_query))
    assert len(matched_nodes) == 2
    assert 'slow_operation' not in [node.name for node in matched_nodes]

    # Test min and max duration together using timedelta
    duration_range_query: SpanQuery = {
        'min_duration': fast_threshold,
        'max_duration': medium_threshold,
    }
    matched_node = tree.find_first(as_predicate(duration_range_query))
    assert matched_node is not None
    assert matched_node.name == 'medium_operation'


async def test_span_query_descendant_conditions():
    """Test descendant-related conditions in SpanQuery."""
    from pydantic_evals.otel.span_tree import SpanQuery, as_predicate

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
    matched_node = tree.find_first(as_predicate(some_child_query))
    assert matched_node is not None
    assert matched_node.name == 'parent1'

    # Test all_children_have condition
    all_children_query: SpanQuery = {'all_children_have': {'has_attributes': {'type': 'normal'}}}
    matched_node = tree.find_first(as_predicate(all_children_query))
    assert matched_node is not None
    assert matched_node.name == 'parent2'

    # Test no_child_has condition
    no_child_query: SpanQuery = {'no_child_has': {'has_attributes': {'type': 'important'}}}
    matched_node = tree.find_first(as_predicate(no_child_query))
    assert matched_node is not None
    assert matched_node.name == 'parent2'


async def test_span_query_complex_hierarchical_conditions():
    """Test complex hierarchical queries with nested structures."""
    from pydantic_evals.otel.span_tree import SpanQuery, as_predicate

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

    matched_node = tree.find_first(as_predicate(complex_query))
    assert matched_node is not None
    assert matched_node.name == 'app'

    # Find request spans with both db_query and another operation
    request_with_db_and_other: SpanQuery = {
        'name_equals': 'request',
        'some_child_has': {'not_': {'name_equals': 'db_query'}},
    }

    matched_nodes = tree.find_all(as_predicate(request_with_db_and_other))
    assert len(matched_nodes) == 2  # Both requests have db_query and another operation


async def test_span_query_as_predicate_conversion():
    """Test that as_predicate correctly converts SpanQuery to a callable predicate."""
    from pydantic_evals.otel.span_tree import SpanQuery, as_predicate, matches

    # Create a simple test span
    with context_subtree() as tree:
        with logfire.span('test_span', category='test'):
            pass
    assert isinstance(tree, SpanTree)

    test_node = tree.roots[0]
    assert test_node.name == 'test_span'

    # Create a query and convert it to a predicate
    query: SpanQuery = {'name_equals': 'test_span', 'has_attributes': {'category': 'test'}}
    predicate = as_predicate(query)

    # Test the predicate directly
    assert predicate(test_node)

    # Verify it's equivalent to calling matches directly
    assert matches(query, test_node)

    # Test with a non-matching query
    non_matching_query: SpanQuery = {'name_equals': 'different_span'}
    non_matching_predicate = as_predicate(non_matching_query)

    assert not non_matching_predicate(test_node)
    assert not matches(non_matching_query, test_node)


async def test_matches_function_directly():
    """Test the matches function directly with various SpanQuery combinations."""
    from pydantic_evals.otel.span_tree import SpanQuery, matches

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
    assert matches({'name_equals': 'parent'}, parent_node)
    assert not matches({'name_equals': 'parent'}, child1_node)

    # Test attribute matching
    assert matches({'has_attributes': {'level': '1'}}, parent_node)
    assert not matches({'has_attributes': {'level': '1'}}, child1_node)

    # Test logical combinations
    complex_query: SpanQuery = {'and_': [{'name_equals': 'child1'}, {'has_attributes': {'category': 'important'}}]}
    assert matches(complex_query, child1_node)
    assert not matches(complex_query, child2_node)

    # Test with descendants
    descendant_query: SpanQuery = {'some_child_has': {'name_equals': 'child1'}}
    assert matches(descendant_query, parent_node)
    assert not matches(descendant_query, child1_node)


async def test_span_query_child_count():
    """Test min_child_count and max_child_count conditions in SpanQuery."""
    from pydantic_evals.otel.span_tree import SpanQuery, as_predicate, matches

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
    matched_nodes = tree.find_all(as_predicate(min_2_query))
    assert len(matched_nodes) == 2
    matched_names = {node.name for node in matched_nodes}
    assert matched_names == {'parent_two_children', 'parent_three_children'}

    # Test max_child_count
    max_1_query: SpanQuery = {'max_child_count': 1}
    matched_nodes = tree.find_all(as_predicate(max_1_query))
    assert len(matched_nodes) == 8  # parent_no_children, parent_one_child, and all the leaf nodes
    assert 'parent_two_children' not in {node.name for node in matched_nodes}
    assert 'parent_three_children' not in {node.name for node in matched_nodes}

    # Test both min and max together (range)
    child_range_query: SpanQuery = {'min_child_count': 1, 'max_child_count': 2}
    matched_nodes = tree.find_all(as_predicate(child_range_query))
    assert len(matched_nodes) == 2
    matched_names = {node.name for node in matched_nodes}
    assert matched_names == {'parent_one_child', 'parent_two_children'}

    # Test with other conditions
    complex_query: SpanQuery = {'name_contains': 'parent', 'min_child_count': 2}
    matched_nodes = tree.find_all(as_predicate(complex_query))
    assert len(matched_nodes) == 2
    assert all('parent' in node.name and len(node.children) >= 2 for node in matched_nodes)

    # Test direct usage of matches function
    parent_three = tree.find_first(lambda node: node.name == 'parent_three_children')
    assert parent_three is not None

    assert matches({'min_child_count': 3}, parent_three)
    assert matches({'min_child_count': 2, 'max_child_count': 3}, parent_three)
    assert not matches({'max_child_count': 2}, parent_three)

    # Test with logical operators
    logical_query: SpanQuery = {
        'and_': [{'name_contains': 'parent'}, {'min_child_count': 1}],
        'not_': {'max_child_count': 1},
    }
    matched_nodes = tree.find_all(as_predicate(logical_query))
    assert len(matched_nodes) == 2
    matched_names = {node.name for node in matched_nodes}
    assert matched_names == {'parent_two_children', 'parent_three_children'}
