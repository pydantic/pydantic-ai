import os
from collections.abc import Iterator
from decimal import Decimal
from typing import Any, get_args
from unittest.mock import patch

import pytest
from inline_snapshot import snapshot
from logfire.testing import CaptureLogfire

from pydantic_ai.embeddings import Embedder, EmbeddingResult, KnownEmbeddingModelName, infer_model
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsList, try_import

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
]

with try_import() as openai_imports_successful:
    from pydantic_ai.embeddings.openai import LatestOpenAIEmbeddingModelNames, OpenAIEmbeddingModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as cohere_imports_successful:
    from pydantic_ai.embeddings.cohere import CohereEmbeddingModel, LatestCohereEmbeddingModelNames
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
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(
                    IsList(
                        snapshot(-0.019193023443222046),
                        snapshot(-0.025299284607172012),
                        snapshot(-0.0016930076526477933),
                        length=1536,
                    ),
                    length=1,
                ),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(input_tokens=4),
                model_name='text-embedding-3-small',
                timestamp=IsDatetime(),
                provider_name='openai',
            )
        )
        assert result.cost().total_price == snapshot(Decimal('8E-8'))

    async def test_documents(self, openai_api_key: str):
        model = OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key=openai_api_key))
        embedder = Embedder(model)
        result = await embedder.embed_documents(['hello', 'world'])
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(
                    IsList(
                        snapshot(0.01681816205382347),
                        snapshot(-0.05579638481140137),
                        snapshot(0.005661087576299906),
                        length=1536,
                    ),
                    IsList(
                        snapshot(-0.010592407546937466),
                        snapshot(-0.03599696233868599),
                        snapshot(0.030227113515138626),
                        length=1536,
                    ),
                    length=2,
                ),
                inputs=['hello', 'world'],
                input_type='document',
                usage=RequestUsage(input_tokens=2),
                model_name='text-embedding-3-small',
                timestamp=IsDatetime(),
                provider_name='openai',
            )
        )
        assert result.cost().total_price == snapshot(Decimal('4E-8'))

    async def test_max_input_tokens(self, openai_api_key: str):
        model = OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key=openai_api_key))
        embedder = Embedder(model)
        max_input_tokens = await embedder.max_input_tokens()
        assert max_input_tokens == snapshot(8192)

    async def test_count_tokens(self, openai_api_key: str):
        model = OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key=openai_api_key))
        embedder = Embedder(model)
        count = await embedder.count_tokens('Hello, world!')
        assert count == snapshot(4)


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
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(
                    IsList(snapshot(-0.018445116), snapshot(0.008921167), snapshot(-0.0011377502), length=1536),
                    length=1,
                ),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(input_tokens=4),
                model_name='embed-v4.0',
                timestamp=IsDatetime(),
                provider_name='cohere',
                provider_response_id='0728b136-9b30-4fb5-bf9a-2c7cf36d51d3',
            )
        )

    async def test_documents(self, co_api_key: str):
        model = CohereEmbeddingModel('embed-v4.0', provider=CohereProvider(api_key=co_api_key))
        embedder = Embedder(model)
        result = await embedder.embed_documents(['hello', 'world'])
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(
                    IsList(snapshot(0.015943069), snapshot(0.013248466), snapshot(0.0024139155), length=1536),
                    IsList(snapshot(-0.0060736495), snapshot(-0.015005487), snapshot(0.00033246286), length=1536),
                    length=2,
                ),
                inputs=['hello', 'world'],
                input_type='document',
                usage=RequestUsage(input_tokens=2),
                model_name='embed-v4.0',
                timestamp=IsDatetime(),
                provider_name='cohere',
                provider_response_id='199299d7-f43d-45af-903c-347fff81bbe4',
            )
        )

    async def test_max_input_tokens(self, co_api_key: str):
        model = CohereEmbeddingModel('embed-v4.0', provider=CohereProvider(api_key=co_api_key))
        embedder = Embedder(model)
        max_input_tokens = await embedder.max_input_tokens()
        assert max_input_tokens == snapshot(128000)

    async def test_count_tokens(self, co_api_key: str):
        model = CohereEmbeddingModel('embed-v4.0', provider=CohereProvider(api_key=co_api_key))
        embedder = Embedder(model)
        count = await embedder.count_tokens('Hello, world!')
        assert count == snapshot(4)


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
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(
                    IsList(
                        snapshot(-0.03817721828818321),
                        snapshot(0.032911062240600586),
                        snapshot(-0.0054594180546700954),
                        length=384,
                    ),
                    length=1,
                ),
                inputs=['Hello, world!'],
                input_type='query',
                model_name='all-MiniLM-L6-v2',
                timestamp=IsDatetime(),
                provider_name='sentence-transformers',
            )
        )

    async def test_documents(self):
        model = SentenceTransformerEmbeddingModel('all-MiniLM-L6-v2')
        embedder = Embedder(model)
        result = await embedder.embed_documents(['hello', 'world'])
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(
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
                ),
                inputs=['hello', 'world'],
                input_type='document',
                model_name='all-MiniLM-L6-v2',
                timestamp=IsDatetime(),
                provider_name='sentence-transformers',
            )
        )

    async def test_max_input_tokens(self):
        model = SentenceTransformerEmbeddingModel('all-MiniLM-L6-v2')
        embedder = Embedder(model)
        max_input_tokens = await embedder.max_input_tokens()
        assert max_input_tokens == snapshot(256)

    async def test_count_tokens(self):
        model = SentenceTransformerEmbeddingModel('all-MiniLM-L6-v2')
        embedder = Embedder(model)
        count = await embedder.count_tokens('Hello, world!')
        assert count == snapshot(6)


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
                    'gen_ai.usage.input_tokens': 4,
                    'gen_ai.embedding.dimension': 1536,
                    'operation.cost': 8e-08,
                    'gen_ai.embedding.num_outputs': 1,
                    'gen_ai.response.model': 'text-embedding-3-small',
                },
            }
        ]
    )


@pytest.mark.skipif(
    not openai_imports_successful() or not cohere_imports_successful(),
    reason='some embedding package was not installed',
)
def test_known_embedding_model_names():  # pragma: lax no cover
    # Coverage seems to be misbehaving..?
    def get_model_names(model_name_type: Any) -> Iterator[str]:
        for arg in get_args(model_name_type):
            if isinstance(arg, str):
                yield arg
            else:
                yield from get_model_names(arg)

    openai_names = [f'openai:{n}' for n in get_model_names(LatestOpenAIEmbeddingModelNames)]
    cohere_names = [f'cohere:{n}' for n in get_model_names(LatestCohereEmbeddingModelNames)]

    generated_names = sorted(openai_names + cohere_names)

    known_embedding_model_names = sorted(get_args(KnownEmbeddingModelName.__value__))
    assert generated_names == known_embedding_model_names
