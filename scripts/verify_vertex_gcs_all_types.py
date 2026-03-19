"""Verify all file types work with gs:// URIs in Vertex tool results.

Usage:
    unset GOOGLE_API_KEY GEMINI_API_KEY
    GOOGLE_PROJECT=gen-lang-client-0498264908 GOOGLE_LOCATION=global uv run python scripts/verify_vertex_gcs_all_types.py
"""

from __future__ import annotations

import asyncio
import os

GCS_BUCKET = 'pydantic-ai-test-files-vertex'
FILES = {
    'image': (f'gs://{GCS_BUCKET}/test-files/kiwi.jpg', 'image/jpeg'),
    'document': (f'gs://{GCS_BUCKET}/test-files/dummy.pdf', 'application/pdf'),
    'audio': (f'gs://{GCS_BUCKET}/test-files/marcelo.mp3', 'audio/mpeg'),
    'video': (f'gs://{GCS_BUCKET}/test-files/small_video.mp4', 'video/mp4'),
}


async def test_file_type(file_type: str, gcs_uri: str, mime_type: str) -> None:
    from google.genai import Client
    from google.genai.types import Content, FunctionCall, FunctionResponse, Part, Tool, FunctionDeclaration, Schema

    project = os.environ['GOOGLE_PROJECT']
    location = os.environ.get('GOOGLE_LOCATION', 'global')
    client = Client(vertexai=True, project=project, location=location)

    contents = [
        Content(role='user', parts=[Part(text=f'Use the get_{file_type} tool and describe what you get.')]),
        Content(role='model', parts=[Part(function_call=FunctionCall(name=f'get_{file_type}', args={}))]),
        Content(
            role='user',
            parts=[
                Part(
                    function_response=FunctionResponse(
                        name=f'get_{file_type}',
                        response={'result': f'Here is the {file_type}'},
                    )
                ),
                Part(file_data={'file_uri': gcs_uri, 'mime_type': mime_type}),
            ],
        ),
    ]

    tools = [
        Tool(function_declarations=[
            FunctionDeclaration(
                name=f'get_{file_type}',
                description=f'Gets a {file_type}',
                parameters=Schema(type='OBJECT', properties={}),
            )
        ])
    ]

    try:
        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config={'tools': tools},
        )
        text = response.text[:150] if response.text else '(no text)'
        print(f'  {file_type}: OK — {text}')
    except Exception as e:
        print(f'  {file_type}: FAILED — {type(e).__name__}: {e}')


async def main() -> None:
    print(f'Vertex AI tool results with gs:// URIs')
    print(f'Project: {os.environ.get("GOOGLE_PROJECT")}')
    print()

    for file_type, (gcs_uri, mime_type) in FILES.items():
        await test_file_type(file_type, gcs_uri, mime_type)


if __name__ == '__main__':
    asyncio.run(main())
