"""Quick verification script for Anthropic multimodal changes."""
import asyncio

from pydantic_ai_slim.pydantic_ai.models.anthropic import AnthropicModel


async def test_map_binary_to_block():
    """Test the new _map_binary_to_block helper method."""

    # Test with image data
    image_data = b'\x89PNG\r\n\x1a\n fake image data'
    image_block = AnthropicModel._map_binary_to_block(image_data, 'image/png')
    assert image_block['type'] == 'image'
    assert image_block['source']['type'] == 'base64'
    assert image_block['source']['media_type'] == 'image/png'
    print('✓ Image block creation works')

    # Test with PDF data
    pdf_data = b'%PDF-1.4 fake pdf data'
    pdf_block = AnthropicModel._map_binary_to_block(pdf_data, 'application/pdf')
    assert pdf_block['type'] == 'document'
    assert pdf_block['source']['type'] == 'base64'
    assert pdf_block['source']['media_type'] == 'application/pdf'
    print('✓ PDF block creation works')

    # Test with text data
    text_data = b'Sample text content'
    text_block = AnthropicModel._map_binary_to_block(text_data, 'text/plain')
    assert text_block['type'] == 'document'
    assert text_block['source']['type'] == 'text'
    assert text_block['source']['media_type'] == 'text/plain'
    assert text_block['source']['data'] == 'Sample text content'
    print('✓ Text block creation works')

    # Test with unsupported media type
    try:
        AnthropicModel._map_binary_to_block(b'data', 'application/unknown')
        assert False, 'Should have raised RuntimeError'
    except RuntimeError as e:
        assert 'Unsupported binary content media type' in str(e)
        print('✓ Unsupported media type raises error')

    print('\n✅ All _map_binary_to_block tests passed!')


if __name__ == '__main__':
    asyncio.run(test_map_binary_to_block())
