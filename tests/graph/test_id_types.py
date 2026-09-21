from pydantic_graph.id_types import NodeID, generate_placeholder_node_id, replace_placeholder_id


def test_replace_placeholder_id_returns_label_for_placeholder():
    """A placeholder id is reduced to the simplified label it was generated with, not a boolean.

    Unit test rather than a graph-building test: the function's contract is that it returns a
    `str` (the extracted label), and that string return is what `_build_placeholder_node_id_remapping`
    relies on to detect and deterministically rename placeholder ids.
    """
    placeholder = generate_placeholder_node_id('MyNode')
    assert replace_placeholder_id(NodeID(placeholder)) == 'MyNode'


def test_replace_placeholder_id_passes_through_non_placeholder():
    """A non-placeholder id is returned unchanged, which is how the remapping tells the two apart."""
    assert replace_placeholder_id(NodeID('already_simple')) == 'already_simple'
