"""Example: Getting citations from OpenAI Responses API.

Shows how to access citations from the Responses API format.

Run with:
    uv run -m pydantic_ai_examples.citations.openai_responses

Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations as _annotations

from pydantic_ai import Agent, TextPart, URLCitation


def main():
    """Get citations from Responses API."""
    agent = Agent('openai:gpt-4o')

    result = agent.run_sync(
        'What are the key features of Python 3.12? Provide sources.'
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
                        if isinstance(citation, URLCitation):
                            cited_text = part.content[
                                citation.start_index : citation.end_index
                            ]

                            print(f'Citation {i}:')
                            print(f'  Title: {citation.title or "N/A"}')
                            print(f'  URL: {citation.url}')
                            print(
                                f'  Text Range: {citation.start_index}-{citation.end_index}'
                            )
                            print(f'  Cited Text: "{cited_text}"')
                            print()

    if not citations_found:
        print('No citations found.')
        print('Citations only appear when the model includes them.')


if __name__ == '__main__':
    main()
