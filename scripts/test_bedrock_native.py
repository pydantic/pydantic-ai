"""Verify which Bedrock models accept documents/videos natively in toolResult.content.

The API spec (ToolResultContentBlock) lists document/video as valid members,
but Nova Pro rejects them. Testing with Claude to see if it's model-specific.
"""

from pathlib import Path

import boto3

ASSETS = Path(__file__).parent.parent / 'tests' / 'assets'

TOOL_CONFIG = {
    'tools': [
        {
            'toolSpec': {
                'name': 'get_file',
                'description': 'Gets a file',
                'inputSchema': {'json': {'type': 'object', 'properties': {}}},
            }
        }
    ]
}


def test_native(model_id: str, file_type: str) -> None:
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    tool_use_id = 'test-tool-123'

    if file_type == 'document':
        file_bytes = (ASSETS / 'dummy.pdf').read_bytes()
        file_block = {'document': {'name': 'Doc 1', 'format': 'pdf', 'source': {'bytes': file_bytes}}}
    elif file_type == 'video':
        file_bytes = (ASSETS / 'small_video.mp4').read_bytes()
        file_block = {'video': {'format': 'mp4', 'source': {'bytes': file_bytes}}}
    else:
        file_bytes = (ASSETS / 'kiwi.jpg').read_bytes()
        file_block = {'image': {'format': 'jpeg', 'source': {'bytes': file_bytes}}}

    messages = [
        {'role': 'user', 'content': [{'text': f'Use the get_file tool to get a {file_type}.'}]},
        {
            'role': 'assistant',
            'content': [{'toolUse': {'toolUseId': tool_use_id, 'name': 'get_file', 'input': {}}}],
        },
        {
            'role': 'user',
            'content': [
                {
                    'toolResult': {
                        'toolUseId': tool_use_id,
                        'status': 'success',
                        'content': [
                            {'text': f'Here is the {file_type}'},
                            file_block,
                        ],
                    }
                }
            ],
        },
    ]

    print(f'  {file_type:10s} ... ', end='', flush=True)
    try:
        response = client.converse(modelId=model_id, messages=messages, toolConfig=TOOL_CONFIG)
        output_text = response['output']['message']['content'][0].get('text', '')
        print(f'OK: {output_text[:120]}')
    except Exception as e:
        error_msg = str(e)
        # Extract the key part of the error
        if 'Malformed' in error_msg:
            print(f'REJECTED: {error_msg.split("errors: ")[1][:120] if "errors: " in error_msg else error_msg[:120]}')
        else:
            print(f'ERROR: {type(e).__name__}: {error_msg[:120]}')


if __name__ == '__main__':
    models = [
        'us.amazon.nova-pro-v1:0',
        'us.amazon.nova-lite-v1:0',
        'us.amazon.nova-2-lite-v1:0',
        'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'us.anthropic.claude-haiku-4-5-20251001-v1:0',
    ]
    for model_id in models:
        print(f'\n=== {model_id} ===')
        for ft in ('image', 'document', 'video'):
            test_native(model_id, ft)
