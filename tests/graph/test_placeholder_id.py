from pydantic_graph.id_types import generate_placeholder_node_id, replace_placeholder_id


def test_roundtrip_label_with_colon():
    label = 'a:b'
    node_id = generate_placeholder_node_id(label)
    assert replace_placeholder_id(node_id) == label


def test_roundtrip_label_with_multiple_colons():
    label = 'a:b:c'
    node_id = generate_placeholder_node_id(label)
    assert replace_placeholder_id(node_id) == label


def test_roundtrip_plain_label():
    label = 'plain'
    node_id = generate_placeholder_node_id(label)
    assert replace_placeholder_id(node_id) == label


def test_non_placeholder_id_unchanged():
    assert replace_placeholder_id('regular_node') == 'regular_node'
