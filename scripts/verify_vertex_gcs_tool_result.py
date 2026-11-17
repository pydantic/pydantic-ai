"""Verify Vertex AI can use gs:// URIs in tool result (function response) parts.

Usage:
    unset GOOGLE_API_KEY GEMINI_API_KEY
    GOOGLE_PROJECT=gen-lang-client-0498264908 GOOGLE_LOCATION=global uv run python scripts/verify_vertex_gcs_tool_result.py
"""

from __future__ import annotations

import asyncio
import os

GCS_BUCKET = 'pydantic-ai-test-files-vertex'


async def test_tool_result_with_gcs() -> None:
    from google.genai import Client
    from google.genai.types import Content, FunctionCall, FunctionResponse, Part, Tool, FunctionDeclaration, Schema

    project = os.environ['GOOGLE_PROJECT']
    location = os.environ.get('GOOGLE_LOCATION', 'global')
    client = Client(vertexai=True, project=project, location=location)

    gcs_uri = f'gs://{GCS_BUCKET}/test-files/kiwi.jpg'

    contents = [
        Content(role='user', parts=[Part(text='Use the get_image tool and describe what you see.')]),
        Content(role='model', parts=[Part(function_call=FunctionCall(name='get_image', args={}))]),
        Content(
            role='user',
            parts=[
                Part(
                    function_response=FunctionResponse(
                        name='get_image',
                        response={
                            'result': 'Here is the image',
                            'file': {'file_uri': gcs_uri, 'mime_type': 'image/jpeg'},
                        },
                    )
                ),
            ],
        ),
    ]

    tools = [
        Tool(function_declarations=[
            FunctionDeclaration(
                name='get_image',
                description='Gets an image',
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
        text = response.text[:200] if response.text else '(no text)'
        print(f'Tool result with gs:// in FunctionResponse: OK — {text}')
    except Exception as e:
        print(f'Tool result with gs:// in FunctionResponse: FAILED — {type(e).__name__}: {e}')

    # Also try: file_data part alongside function_response
    contents2 = [
        Content(role='user', parts=[Part(text='Use the get_image tool and describe what you see.')]),
        Content(role='model', parts=[Part(function_call=FunctionCall(name='get_image', args={}))]),
        Content(
            role='user',
            parts=[
                Part(
                    function_response=FunctionResponse(
                        name='get_image',
                        response={'result': 'Here is the image'},
                    )
                ),
                Part(file_data={'file_uri': gcs_uri, 'mime_type': 'image/jpeg'}),
            ],
        ),
    ]

    try:
        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents2,
            config={'tools': tools},
        )
        text = response.text[:200] if response.text else '(no text)'
        print(f'Tool result with gs:// as separate file_data part: OK — {text}')
    except Exception as e:
        print(f'Tool result with gs:// as separate file_data part: FAILED — {type(e).__name__}: {e}')


if __name__ == '__main__':
    asyncio.run(test_tool_result_with_gcs())
