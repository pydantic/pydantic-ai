#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydantic-ai-slim[xai]",
# ]
# ///

"""MRE: demonstrate the bug on the published release.

Build a ModelResponse with [ThinkingPart, ToolCallPart] and feed it through
XaiModel._map_response_parts. On the buggy release this produces two separate
assistant messages instead of one.
"""

from pydantic_ai.messages import ThinkingPart, ToolCallPart
from pydantic_ai.models.xai import XaiModel
from pydantic_ai.providers.xai import XaiProvider


def main():
    provider = XaiProvider(api_key='dummy')
    model = XaiModel('grok-4-fast-non-reasoning', provider=provider)

    parts = [
        ThinkingPart(content='I need to use a tool', signature='sig-abc', provider_name='xai'),
        ToolCallPart(tool_name='my_tool', args={}, tool_call_id='call-1'),
    ]

    messages = model._map_response_parts(parts)
    print(f'Number of assistant messages: {len(messages)}')
    for i, msg in enumerate(messages):
        print(
            f'  Message {i}: '
            f'reasoning={msg.reasoning_content!r}, '
            f'encrypted={msg.encrypted_content!r}, '
            f'tool_calls={len(msg.tool_calls)}'
        )

    if len(messages) == 1 and messages[0].tool_calls:
        print('PASS: ThinkingPart and ToolCallPart are on the same assistant message.')
    else:
        print('BUG: ThinkingPart and ToolCallPart are on separate assistant messages.')


if __name__ == '__main__':
    main()
