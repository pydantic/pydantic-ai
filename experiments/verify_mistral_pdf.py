"""Verification script: Mistral vision API does NOT support PDFs.

This script demonstrates that Mistral's vision/chat API rejects PDF files.
The Mistral SDK has `DocumentURLChunk` but it's for the separate Document AI API,
not for the vision models.

Expected: API will reject the PDF with an error.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import pydantic_ai
sys.path.insert(0, str(Path(__file__).parent.parent))

from mistralai import Mistral


async def test_mistral_pdf_rejection():
    """Test that Mistral vision API rejects PDFs."""
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        print('ERROR: MISTRAL_API_KEY environment variable not set')
        print('Please run: source .env && python experiments/verify_mistral_pdf.py')
        sys.exit(1)

    client = Mistral(api_key=api_key)

    # Test 1: Try to send a PDF URL (will fail)
    print('Test 1: Attempting to send PDF via DocumentURLChunk...')
    try:
        from mistralai.models import DocumentURLChunk, UserMessage

        messages = [
            UserMessage(
                role='user',
                content=[
                    {'type': 'text', 'text': 'What is in this document?'},
                    DocumentURLChunk(
                        document_url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf',
                        type='document_url',
                    ),
                ],
            )
        ]

        response = await client.chat.stream_async(
            model='mistral-large-2512', messages=messages, max_tokens=100
        )

        result_text = ''
        async for chunk in response:
            if chunk.data.choices[0].delta.content:
                result_text += chunk.data.choices[0].delta.content

        print(f'ERROR: API accepted PDF! Response: {result_text}')
        print('This should not happen - Mistral vision API should reject PDFs')
        return False

    except Exception as e:
        print(f'✓ EXPECTED: API rejected PDF with error: {type(e).__name__}: {e}')
        print(f'\nFull error: {e}')
        return True


async def test_mistral_image_works():
    """Verify that images DO work (for comparison)."""
    api_key = os.getenv('MISTRAL_API_KEY')
    client = Mistral(api_key=api_key)

    print('\n\nTest 2: Verifying images work correctly...')
    try:
        from mistralai.models import ImageURLChunk, UserMessage

        messages = [
            UserMessage(
                role='user',
                content=[
                    {'type': 'text', 'text': 'What is in this image?'},
                    ImageURLChunk(
                        image_url='https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/200px-Cat03.jpg'
                    ),
                ],
            )
        ]

        response = await client.chat.stream_async(
            model='mistral-large-2512', messages=messages, max_tokens=50
        )

        result_text = ''
        async for chunk in response:
            if chunk.data.choices[0].delta.content:
                result_text += chunk.data.choices[0].delta.content

        print(f'✓ SUCCESS: Image processed correctly')
        print(f'Response: {result_text[:100]}...')
        return True

    except Exception as e:
        print(f'ERROR: Image request failed: {e}')
        return False


async def main():
    print('=' * 80)
    print('Mistral PDF Support Verification')
    print('=' * 80)
    print()
    print('Purpose: Verify that Mistral vision API does NOT support PDFs')
    print('Expected: Test 1 fails (PDF rejected), Test 2 succeeds (image works)')
    print()

    pdf_rejected = await test_mistral_pdf_rejection()
    image_works = await test_mistral_image_works()

    print('\n' + '=' * 80)
    print('RESULTS:')
    print('=' * 80)
    print(f'PDF rejected (expected): {"✓ PASS" if pdf_rejected else "✗ FAIL"}')
    print(f'Image works (expected): {"✓ PASS" if image_works else "✗ FAIL"}')
    print()

    if pdf_rejected and image_works:
        print('✓ VERIFICATION COMPLETE: Mistral vision API does NOT support PDFs')
        print('  → Our code should reject PDFs for Mistral models')
    else:
        print('✗ UNEXPECTED RESULTS')

    return pdf_rejected and image_works


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
