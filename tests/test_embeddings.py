import os
from unittest.mock import patch

import pytest
from dirty_equals import IsList
from inline_snapshot import snapshot
from logfire.testing import CaptureLogfire

from pydantic_ai.embeddings import Embedder, infer_model

from .conftest import try_import

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
]

with try_import() as openai_imports_successful:
    from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as cohere_imports_successful:
    from pydantic_ai.embeddings.cohere import CohereEmbeddingModel
    from pydantic_ai.providers.cohere import CohereProvider

with try_import() as sentence_transformers_imports_successful:
    from pydantic_ai.embeddings.sentence_transformers import SentenceTransformerEmbeddingModel


@pytest.mark.skipif(not openai_imports_successful, reason='OpenAI not installed')
class TestOpenAI:
    async def test_infer_model(self, openai_api_key: str):
        with patch.dict(os.environ, {'OPENAI_API_KEY': openai_api_key}):
            model = infer_model('openai:text-embedding-3-small')
        assert isinstance(model, OpenAIEmbeddingModel)
        assert model.model_name == 'text-embedding-3-small'
        assert model.system == 'openai'
        assert model.base_url == 'https://api.openai.com/v1/'

    async def test_infer_model_azure(self, openai_api_key: str):
        with patch.dict(
            os.environ,
            {
                'AZURE_OPENAI_API_KEY': 'azure-openai-api-key',
                'AZURE_OPENAI_ENDPOINT': 'https://project-id.openai.azure.com/',
                'OPENAI_API_VERSION': '2023-03-15-preview',
            },
        ):
            model = infer_model('azure:text-embedding-3-small')
        assert isinstance(model, OpenAIEmbeddingModel)
        assert model.model_name == 'text-embedding-3-small'
        assert model.system == 'azure'
        assert 'azure.com' in model.base_url

    async def test_query(self, openai_api_key: str):
        model = OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key=openai_api_key))
        embedder = Embedder(model)
        embeddings = await embedder.embed_query('Hello, world!')
        assert embeddings == IsList(-0.019143931567668915, -0.025292053818702698, -0.0017211713129654527, length=1536)

    async def test_documents(self, openai_api_key: str):
        model = OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key=openai_api_key))
        embedder = Embedder(model)
        embeddings = await embedder.embed_documents(['hello', 'world'])
        assert embeddings == IsList(
            IsList(0.016751619055867195, -0.055799614638090134, 0.005647437181323767, length=1536),
            IsList(-0.010633519850671291, -0.03604777529835701, 0.03019288368523121, length=1536),
            length=2,
        )


@pytest.mark.skipif(not cohere_imports_successful, reason='Cohere not installed')
class TestCohere:
    async def test_infer_model(self, co_api_key: str):
        with patch.dict(os.environ, {'CO_API_KEY': co_api_key}):
            model = infer_model('cohere:embed-v4.0')
        assert isinstance(model, CohereEmbeddingModel)
        assert model.model_name == 'embed-v4.0'
        assert model.system == 'cohere'
        assert model.base_url == 'https://api.cohere.com'
        assert isinstance(model._provider, CohereProvider)  # type: ignore[reportAttributeAccess]

    async def test_query(self, co_api_key: str):
        model = CohereEmbeddingModel('embed-v4.0', provider=CohereProvider(api_key=co_api_key))
        embedder = Embedder(model)
        embeddings = await embedder.embed_query('Hello, world!')
        assert embeddings == IsList(-0.016547563, 0.026550192, 0.0044764862, length=1536)

    async def test_documents(self, co_api_key: str):
        model = CohereEmbeddingModel('embed-v4.0', provider=CohereProvider(api_key=co_api_key))
        embedder = Embedder(model)
        embeddings = await embedder.embed_documents(['hello', 'world'])
        assert embeddings == IsList(
            IsList(0.015943069, 0.013248466, 0.0024139155, length=1536),
            IsList(-0.0060736495, -0.015005487, 0.00033246286, length=1536),
            length=2,
        )


@pytest.mark.skipif(not sentence_transformers_imports_successful, reason='SentenceTransformers not installed')
class TestSentenceTransformers:
    async def test_infer_model(self):
        model = infer_model('sentence-transformers:all-MiniLM-L6-v2')
        assert isinstance(model, SentenceTransformerEmbeddingModel)
        assert model.model_name == 'all-MiniLM-L6-v2'
        assert model.system == 'sentence-transformers'
        assert model.base_url is None

    async def test_query(self):
        model = SentenceTransformerEmbeddingModel('all-MiniLM-L6-v2')
        embedder = Embedder(model)
        embeddings = await embedder.embed_query('Hello, world!')
        assert embeddings == IsList(
            snapshot(-0.03817721828818321),
            snapshot(0.032911062240600586),
            snapshot(-0.0054594180546700954),
            length=384,
        )

    async def test_documents(self):
        model = SentenceTransformerEmbeddingModel('all-MiniLM-L6-v2')
        embedder = Embedder(model)
        embeddings = await embedder.embed_documents(['hello', 'world'])
        assert embeddings == IsList(
            IsList(
                snapshot(-0.06277172267436981),
                snapshot(0.05495882406830788),
                snapshot(0.052164893597364426),
                length=384,
            ),
            IsList(
                snapshot(-0.030238090083003044),
                snapshot(0.03164675831794739),
                snapshot(-0.06337423622608185),
                length=384,
            ),
            length=2,
        )


@pytest.mark.skipif(not openai_imports_successful, reason='OpenAI not installed')
async def test_instrumentation(openai_api_key: str, capfire: CaptureLogfire):
    model = OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key=openai_api_key))
    embedder = Embedder(model, instrument=True)
    await embedder.embed_query('Hello, world!')

    assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'embed text-embedding-3-small',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'gen_ai.operation.name': 'embed',
                    'gen_ai.system': 'openai',
                    'gen_ai.request.model': 'text-embedding-3-small',
                    'server.address': 'api.openai.com',
                    'gen_ai.embedding.input_type': 'query',
                    'gen_ai.embedding.num_inputs': 1,
                    'gen_ai.prompt': 'Hello, world!',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'embedding_settings': {'type': 'object'},
                            'gen_ai.prompt': {'type': ['string', 'array']},
                        },
                    },
                    'logfire.span_type': 'span',
                    'logfire.msg': 'embed text-embedding-3-small',
                    'gen_ai.embedding.dimension': 1536,
                    'gen_ai.embedding.num_outputs': 1,
                    'gen_ai.response.model': 'text-embedding-3-small',
                },
            }
        ]
    )
