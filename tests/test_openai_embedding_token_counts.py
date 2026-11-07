"""OpenAI embedding token counting is local, so these tests need no API recording."""

import pytest

from pydantic_ai import Embedder

from .conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='OpenAI not installed')


@pytest.mark.parametrize('model_name', ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'])
@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('', 0),
        ('Hello, world!', 4),
        ('<|endoftext|>', 7),
        ('<|fim_prefix|>', 7),
        ('The document says <|endoftext|>.', 9),
    ],
)
async def test_count_tokens_treats_special_token_spellings_as_text(model_name: str, text: str, expected: int):
    """Literal marker spellings in documents are ordinary text, not control tokens."""
    embedder = Embedder(OpenAIEmbeddingModel(model_name, provider=OpenAIProvider(api_key='test-key')))
    assert await embedder.count_tokens(text) == expected
