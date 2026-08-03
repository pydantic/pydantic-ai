"""Verify Vertex AI can use gs:// URIs in generateContent requests.

Usage:
    unset GOOGLE_API_KEY GEMINI_API_KEY
    GOOGLE_PROJECT=gen-lang-client-0498264908 GOOGLE_LOCATION=global uv run python scripts/verify_vertex_gcs.py
"""

from __future__ import annotations

import asyncio
import os

GCS_BUCKET = 'pydantic-ai-test-files-vertex'
GCS_FILES = {
    'image': (f'gs://{GCS_BUCKET}/test-files/kiwi.jpg', 'image/jpeg'),
    'document': (f'gs://{GCS_BUCKET}/test-files/dummy.pdf', 'application/pdf'),
    'audio': (f'gs://{GCS_BUCKET}/test-files/marcelo.mp3', 'audio/mpeg'),
    'video': (f'gs://{GCS_BUCKET}/test-files/small_video.mp4', 'video/mp4'),
}


async def test_vertex_with_gcs_uri(file_type: str, gcs_uri: str, mime_type: str) -> None:
    from google.genai import Client
    from google.genai.types import Content, Part

    project = os.environ['GOOGLE_PROJECT']
    location = os.environ.get('GOOGLE_LOCATION', 'global')

    client = Client(vertexai=True, project=project, location=location)

    prompt = {
        'image': 'Describe this image in one sentence.',
        'document': 'What is this document about? One sentence.',
        'audio': 'Describe this audio in one sentence.',
        'video': 'Describe this video in one sentence.',
    }[file_type]

    try:
        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                Content(
                    role='user',
                    parts=[
                        Part(text=prompt),
                        Part(file_data={'file_uri': gcs_uri, 'mime_type': mime_type}),
                    ],
                )
            ],
        )
        text = response.text[:100] if response.text else '(no text)'
        print(f'  {file_type}: OK — {text}')
    except Exception as e:
        print(f'  {file_type}: FAILED — {type(e).__name__}: {e}')


async def main() -> None:
    print(f'Project: {os.environ.get("GOOGLE_PROJECT")}')
    print(f'Location: {os.environ.get("GOOGLE_LOCATION", "global")}')
    print()

    for file_type, (gcs_uri, mime_type) in GCS_FILES.items():
        await test_vertex_with_gcs_uri(file_type, gcs_uri, mime_type)


if __name__ == '__main__':
    asyncio.run(main())
