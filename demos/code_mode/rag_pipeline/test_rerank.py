"""Test PineconeGuardToolset truncation logic."""

from __future__ import annotations

from .guard import PineconeGuardToolset


def test_truncation() -> None:
    """Test that PineconeGuardToolset truncates documents correctly."""
    guard: PineconeGuardToolset[None] = PineconeGuardToolset(wrapped=None, max_chars_per_doc=100)  # type: ignore

    # Test string documents
    long_doc = 'A' * 200
    args = {'documents': [long_doc, 'short'], 'query': 'test'}
    result = guard._truncate_documents(args)
    assert len(result['documents'][0]) == 100, f'Expected 100, got {len(result["documents"][0])}'
    assert result['documents'][1] == 'short'
    print('✓ String documents truncated correctly')

    # Test dict documents with 'text' key
    args = {'documents': [{'text': long_doc}, {'text': 'short'}], 'query': 'test'}
    result = guard._truncate_documents(args)
    assert len(result['documents'][0]['text']) == 100
    assert result['documents'][1]['text'] == 'short'
    print('✓ Dict documents with "text" key truncated correctly')

    # Test dict documents with 'content' key
    args = {'documents': [{'content': long_doc}], 'query': 'test'}
    result = guard._truncate_documents(args)
    assert len(result['documents'][0]['content']) == 100
    print('✓ Dict documents with "content" key truncated correctly')

    # Test empty documents
    args = {'documents': [], 'query': 'test'}
    result = guard._truncate_documents(args)
    assert result['documents'] == []
    print('✓ Empty documents handled correctly')

    # Test no documents key
    args = {'query': 'test'}
    result = guard._truncate_documents(args)
    assert 'documents' not in result or result.get('documents') == []
    print('✓ Missing documents key handled correctly')

    print('\nAll truncation tests passed!')


if __name__ == '__main__':
    test_truncation()
