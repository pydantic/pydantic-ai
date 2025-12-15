"""Example: Getting citations from Google Gemini.

Shows how to access grounding citations from Gemini models.

Run with:
    uv run -m pydantic_ai_examples.citations.google

Requires GOOGLE_API_KEY environment variable. Citations are more likely
when grounding tools like Google Search are enabled.
"""

from __future__ import annotations as _annotations

from pydantic_ai import Agent, GroundingCitation, TextPart


def main():
    """Get citations from Gemini responses."""
    agent = Agent('google-gla:gemini-1.5-flash')

    result = agent.run_sync('What are the latest developments in AI? Provide sources.')

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
                        if isinstance(citation, GroundingCitation):
                            print(f'Citation {i}:')
                            if citation.citation_metadata:
                                print(
                                    f'  Citation Metadata: {citation.citation_metadata}'
                                )
                            if citation.grounding_metadata:
                                print(
                                    f'  Grounding Metadata: {citation.grounding_metadata}'
                                )
                            print()

    if not citations_found:
        print('No citations found.')
        print('Citations appear when grounding metadata is present.')


if __name__ == '__main__':
    main()
