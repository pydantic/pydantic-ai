"""Verify Bedrock + S3 access for uploaded_file tests.

Usage: .claude/skills/pytest-vcr/run-bedrock-tests.sh -m scripts/verify_bedrock_access.py
   or: source .env && AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... uv run python scripts/verify_bedrock_access.py
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    import boto3

    key_id = os.environ.get('AWS_ACCESS_KEY_ID', '')
    print(f'AWS_ACCESS_KEY_ID: {key_id[:8]}...' if key_id else 'AWS_ACCESS_KEY_ID: NOT SET')
    print(f'AWS_SECRET_ACCESS_KEY: {"set" if os.environ.get("AWS_SECRET_ACCESS_KEY") else "NOT SET"}')
    print(f'AWS_BEARER_TOKEN_BEDROCK: {"set" if os.environ.get("AWS_BEARER_TOKEN_BEDROCK") else "not set (good)"}')
    print()

    # 1. Check S3 access
    print('--- S3 Access ---')
    s3 = boto3.client('s3')
    bucket = 'pydantic-ai-test-files'
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix='test-files/', MaxKeys=10)
        files = [obj['Key'] for obj in resp.get('Contents', [])]
        print(f'  Bucket {bucket}: {len(files)} files found')
        for f in files:
            print(f'    {f}')
    except Exception as e:
        print(f'  Bucket {bucket}: FAILED - {e}')

    # 2. Check Bedrock model access
    print()
    print('--- Bedrock Model Access ---')
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    models = [
        'us.amazon.nova-2-lite-v1:0',
        'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    ]
    for model in models:
        try:
            resp = bedrock.converse(
                modelId=model,
                messages=[{'role': 'user', 'content': [{'text': 'Say "ok"'}]}],
                inferenceConfig={'maxTokens': 5},
            )
            text = resp['output']['message']['content'][0]['text']
            print(f'  {model}: OK ({text.strip()!r})')
        except Exception as e:
            print(f'  {model}: FAILED - {e}')

    # 3. Check S3 image in tool result (the problematic case)
    print()
    print('--- S3 Image in Tool Result (Nova) ---')
    try:
        resp = bedrock.converse(
            modelId='us.amazon.nova-2-lite-v1:0',
            messages=[
                {'role': 'user', 'content': [{'text': 'Call the tool'}]},
                {'role': 'assistant', 'content': [{'toolUse': {'toolUseId': 'test1', 'name': 'get_file', 'input': {}}}]},
                {
                    'role': 'user',
                    'content': [
                        {
                            'toolResult': {
                                'toolUseId': 'test1',
                                'content': [
                                    {
                                        'image': {
                                            'format': 'jpeg',
                                            'source': {
                                                's3Location': {'uri': 's3://pydantic-ai-test-files/test-files/kiwi.jpg'}
                                            },
                                        }
                                    }
                                ],
                                'status': 'success',
                            }
                        }
                    ],
                },
            ],
            toolConfig={
                'tools': [
                    {'toolSpec': {'name': 'get_file', 'inputSchema': {'json': {'type': 'object', 'properties': {}}}}}
                ]
            },
            inferenceConfig={'maxTokens': 20},
        )
        text = resp['output']['message']['content'][0]['text']
        print(f'  Result: OK ({text[:60]!r}...)')
    except Exception as e:
        print(f'  Result: FAILED - {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
