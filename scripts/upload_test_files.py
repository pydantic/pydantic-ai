"""Upload test files to providers and print file IDs for use in tests.

Usage: source .env && uv run python scripts/upload_test_files.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ASSETS = Path(__file__).parent.parent / 'tests' / 'assets'


async def upload_openai() -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], timeout=60.0, max_retries=3)

    image_file = await client.files.create(
        file=ASSETS / 'kiwi.jpg',
        purpose='assistants',
    )
    print(f'  image: {image_file.id}')

    doc_file = await client.files.create(
        file=ASSETS / 'dummy.pdf',
        purpose='assistants',
    )
    print(f'  document: {doc_file.id}')


async def upload_anthropic() -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'], timeout=60.0, max_retries=3)

    with open(ASSETS / 'kiwi.jpg', 'rb') as f:
        image_file = client.beta.files.upload(file=f)  # type: ignore[reportUnknownMemberType]
    print(f'  image: {image_file.id}')  # type: ignore[reportUnknownMemberType]

    with open(ASSETS / 'dummy.pdf', 'rb') as f:
        doc_file = client.beta.files.upload(file=f)  # type: ignore[reportUnknownMemberType]
    print(f'  document: {doc_file.id}')  # type: ignore[reportUnknownMemberType]


async def upload_xai() -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ['XAI_API_KEY'],
        base_url='https://api.x.ai/v1',
        timeout=60.0,
        max_retries=3,
    )

    image_file = await client.files.create(
        file=ASSETS / 'kiwi.jpg',
        purpose='assistants',
    )
    print(f'  image: {image_file.id}')

    doc_file = await client.files.create(
        file=ASSETS / 'dummy.pdf',
        purpose='assistants',
    )
    print(f'  document: {doc_file.id}')


async def upload_google() -> None:
    from google.genai import Client

    client = Client(api_key=os.environ.get('GEMINI_API_KEY', os.environ.get('GOOGLE_API_KEY', '')))

    files_to_upload = [
        ('kiwi.jpg', 'image/jpeg', 'image'),
        ('dummy.pdf', 'application/pdf', 'document'),
        ('marcelo.mp3', 'audio/mpeg', 'audio'),
        ('small_video.mp4', 'video/mp4', 'video'),
    ]

    for filename, mime_type, label in files_to_upload:
        result = client.files.upload(
            file=ASSETS / filename,
            config={'mime_type': mime_type},
        )
        print(f'  {label}: {result.uri}')


async def upload_google_vertex() -> None:
    from google.cloud import storage

    bucket_name = 'pydantic-ai-test-files-vertex'
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    files_to_upload = [
        ('kiwi.jpg', 'image/jpeg', 'image'),
        ('dummy.pdf', 'application/pdf', 'document'),
        ('marcelo.mp3', 'audio/mpeg', 'audio'),
        ('small_video.mp4', 'video/mp4', 'video'),
    ]

    for filename, content_type, label in files_to_upload:
        blob = bucket.blob(f'test-files/{filename}')
        blob.upload_from_filename(str(ASSETS / filename), content_type=content_type)
        print(f'  {label}: gs://{bucket_name}/test-files/{filename}')


async def upload_bedrock_s3() -> None:
    import boto3

    bucket = os.environ.get('BEDROCK_S3_BUCKET', 'pydantic-ai-test-files')
    s3 = boto3.client('s3')

    files_to_upload = [
        ('kiwi.jpg', 'image/jpeg', 'image'),
        ('dummy.pdf', 'application/pdf', 'document'),
        ('small_video.mp4', 'video/mp4', 'video'),
    ]

    for filename, content_type, label in files_to_upload:
        key = f'test-files/{filename}'
        s3.upload_file(
            str(ASSETS / filename),
            bucket,
            key,
            ExtraArgs={'ContentType': content_type},
        )
        print(f'  {label}: s3://{bucket}/{key}')


async def main() -> None:
    providers = sys.argv[1:] if len(sys.argv) > 1 else ['openai', 'anthropic', 'xai', 'google', 'google-vertex', 'bedrock']

    for provider in providers:
        print(f'\n--- {provider} ---')
        if provider == 'openai':
            await upload_openai()
        elif provider == 'anthropic':
            await upload_anthropic()
        elif provider == 'xai':
            await upload_xai()
        elif provider == 'google':
            await upload_google()
        elif provider == 'google-vertex':
            await upload_google_vertex()
        elif provider == 'bedrock':
            await upload_bedrock_s3()
        else:
            print(f'  Unknown provider: {provider}')


if __name__ == '__main__':
    asyncio.run(main())
