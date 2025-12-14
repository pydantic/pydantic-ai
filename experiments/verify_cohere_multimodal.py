"""Verification script: Cohere Command A Vision supports multimodal inputs.

This script demonstrates that Cohere's Command A Vision model DOES support
multimodal inputs (images via URL and base64), contradicting our current
error message that says "Cohere does not yet support multi-modal inputs".

Expected: API successfully processes both URL and base64 images.
"""

import asyncio
import base64
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import pydantic_ai
sys.path.insert(0, str(Path(__file__).parent.parent))

import cohere
from cohere import ImageUrl, ImageUrlContent, UserChatMessageV2


async def test_cohere_image_url():
    """Test that Cohere Command A Vision accepts image URLs."""
    api_key = os.getenv('CO_API_KEY')
    if not api_key:
        print('ERROR: CO_API_KEY environment variable not set')
        print('Please run: source .env && python experiments/verify_cohere_multimodal.py')
        sys.exit(1)

    client = cohere.AsyncClientV2(api_key=api_key)

    print('Test 1: Sending image via URL...')
    try:
        message = UserChatMessageV2(
            role='user',
            content=[
                {'type': 'text', 'text': 'What is in this image? Be brief.'},
                ImageUrlContent(
                    type='image_url',
                    image_url=ImageUrl(
                        url='https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/200px-Cat03.jpg',
                        detail='auto',
                    ),
                ),
            ],
        )

        response = await client.chat(model='command-a-vision-07-2025', messages=[message], max_tokens=100)

        result_text = response.message.content[0].text
        print(f'✓ SUCCESS: Image URL processed correctly')
        print(f'Response: {result_text}')
        return True, result_text

    except Exception as e:
        print(f'✗ FAILED: {type(e).__name__}: {e}')
        print(f'\nFull error: {e}')
        return False, str(e)


async def test_cohere_image_base64():
    """Test that Cohere Command A Vision accepts base64-encoded images."""
    api_key = os.getenv('CO_API_KEY')
    client = cohere.AsyncClientV2(api_key=api_key)

    print('\n\nTest 2: Sending image via base64 data URI...')

    # Simple 1x1 red PNG image (base64 encoded)
    red_pixel_png = (
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='
    )
    data_uri = f'data:image/png;base64,{red_pixel_png}'

    try:
        message = UserChatMessageV2(
            role='user',
            content=[
                {'type': 'text', 'text': 'What color is this pixel?'},
                ImageUrlContent(type='image_url', image_url=ImageUrl(url=data_uri, detail='auto')),
            ],
        )

        response = await client.chat(model='command-a-vision-07-2025', messages=[message], max_tokens=100)

        result_text = response.message.content[0].text
        print(f'✓ SUCCESS: Base64 image processed correctly')
        print(f'Response: {result_text}')
        return True, result_text

    except Exception as e:
        print(f'✗ FAILED: {type(e).__name__}: {e}')
        print(f'\nFull error: {e}')
        return False, str(e)


async def main():
    print('=' * 80)
    print('Cohere Multimodal Support Verification')
    print('=' * 80)
    print()
    print('Purpose: Verify that Cohere Command A Vision DOES support multimodal')
    print('Model: command-a-vision-07-2025')
    print('Expected: Both tests succeed')
    print()

    url_success, url_response = await test_cohere_image_url()
    base64_success, base64_response = await test_cohere_image_base64()

    print('\n' + '=' * 80)
    print('RESULTS:')
    print('=' * 80)
    print(f'Image URL test: {"✓ PASS" if url_success else "✗ FAIL"}')
    print(f'Base64 image test: {"✓ PASS" if base64_success else "✗ FAIL"}')
    print()

    if url_success and base64_success:
        print('✓ VERIFICATION COMPLETE: Cohere Command A Vision DOES support multimodal')
        print('  → Our current code incorrectly blocks multimodal inputs')
        print('  → We should implement ImageUrl and BinaryContent support')
    else:
        print('✗ SOME TESTS FAILED - See errors above')

    print('\n' + '=' * 80)
    print('API Responses:')
    print('=' * 80)
    if url_success:
        print(f'\nURL test response:\n{url_response}')
    if base64_success:
        print(f'\nBase64 test response:\n{base64_response}')

    return url_success and base64_success


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
