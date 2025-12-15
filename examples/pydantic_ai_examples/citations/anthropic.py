"""Example: Getting citations from Anthropic Claude.

Shows how to access tool result citations from Claude models.

Run with:
    uv run -m pydantic_ai_examples.citations.anthropic

Requires ANTHROPIC_API_KEY environment variable. Citations typically come
from tool results like web searches.
"""

from __future__ import annotations as _annotations

from pydantic_ai import Agent, TextPart, ToolResultCitation


def main():
    """Get citations from Claude responses."""
    agent = Agent('anthropic:claude-3-5-sonnet-20241022')

    result = agent.run_sync(
        'What are the latest developments in AI? Use web search if needed.'
    )

    print('Response:', result.output)
    print()

    citations_found = False
    for message in result.new_messages():
        if message.role == 'assistant':
            for part in message.parts:
                if isinstance(part, TextPart) and part.citations:
                    citations_found = True
                    print(f'Found {len(part.citations)} citation(s):')
                    print()

                    for i, citation in enumerate(part.citations, 1):
                        if isinstance(citation, ToolResultCitation):
                            print(f'Citation {i}:')
                            print(f'  Tool Name: {citation.tool_name}')
                            print(f'  Tool Call ID: {citation.tool_call_id or "N/A"}')
                            if citation.citation_data:
                                print(f'  Citation Data: {citation.citation_data}')
                            print()

    if not citations_found:
        print('No citations found.')
        print('Citations typically appear when the model uses tools like web search.')


if __name__ == '__main__':
    main()
