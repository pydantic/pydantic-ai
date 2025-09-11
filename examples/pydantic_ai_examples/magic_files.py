"""Magic file inputs example.

Demonstrates provider-agnostic file inputs using MagicDocumentUrl and MagicBinaryContent.

Behavior summary:
- OpenAI: text/plain is converted to inline text with a clear BEGIN/END file delimiter.
- Other providers (e.g., Anthropic, Gemini): file/url parts are passed as-is when supported.
"""

from __future__ import annotations

from pydantic_ai import Agent, MagicBinaryContent, MagicDocumentUrl

# Load API keys from .env if available
try:  # pragma: no cover - example bootstrap
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    pass


def run_with_openai() -> None:
    agent = Agent('openai:gpt-4o')

    # Text file by URL → becomes inline text with a file delimiter on OpenAI
    txt_url = MagicDocumentUrl(
        url='https://raw.githubusercontent.com/pydantic/pydantic/main/README.md',
        filename='README.md',
        # media_type optional; inferred from extension if omitted
        media_type='text/plain',
    )

    # Binary text (bytes) → becomes inline text with a file delimiter on OpenAI
    txt_bytes = MagicBinaryContent(
        data=b'Hello from bytes',
        media_type='text/plain',
        filename='hello.txt',
    )

    # PDF by URL → remains a file part (base64 + strict MIME) on OpenAI
    pdf_url = MagicDocumentUrl(
        url='https://arxiv.org/pdf/2403.05530.pdf',
        filename='gemini-tech-report.pdf',
        media_type='application/pdf',
    )

    result = agent.run_sync(
        [
            'Summarize these inputs and mention their filenames:',
            txt_url,
            txt_bytes,
            pdf_url,
        ]
    )
    print('\n[OpenAI] Output:\n', result.output)


def run_with_anthropic() -> None:
    agent = Agent('anthropic:claude-3-5-sonnet-latest')

    txt_url = MagicDocumentUrl(
        url='https://raw.githubusercontent.com/pydantic/pydantic/main/README.md',
        filename='README.md',
        media_type='text/plain',
    )
    pdf_url = MagicDocumentUrl(
        url='https://arxiv.org/pdf/2403.05530.pdf',
        filename='gemini-tech-report.pdf',
        media_type='application/pdf',
    )

    # For non-OpenAI providers, Magic* pass through like their base classes when supported.
    result = agent.run_sync(
        [
            'Briefly summarize these documents:',
            txt_url,
            pdf_url,
        ]
    )
    print('\n[Anthropic] Output:\n', result.output)


if __name__ == '__main__':
    # Pick the snippets you want to try locally based on installed provider extras and API keys.
    # Comment/uncomment as needed.
    try:
        run_with_openai()
    except Exception as e:  # pragma: no cover - example only
        print('[OpenAI] Example skipped:', e)

    try:
        run_with_anthropic()
    except Exception as e:  # pragma: no cover - example only
        print('[Anthropic] Example skipped:', e)
