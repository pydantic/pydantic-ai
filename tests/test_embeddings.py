import os
from collections.abc import Iterator
from decimal import Decimal
from typing import Any, get_args
from unittest.mock import patch

import pytest
from inline_snapshot import snapshot
from logfire.testing import CaptureLogfire

from pydantic_ai.embeddings import Embedder, EmbeddingResult, KnownEmbeddingModelName, infer_embedding_model
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsFloat, IsInt, IsList, try_import

pytestmark = [
    pytest.mark.anyio,
]

with try_import() as openai_imports_successful:
    from pydantic_ai.embeddings.openai import LatestOpenAIEmbeddingModelNames, OpenAIEmbeddingModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as cohere_imports_successful:
    from pydantic_ai.embeddings.cohere import CohereEmbeddingModel, LatestCohereEmbeddingModelNames
    from pydantic_ai.providers.cohere import CohereProvider

with try_import() as sentence_transformers_imports_successful:
    from sentence_transformers import SentenceTransformer

    from pydantic_ai.embeddings.sentence_transformers import SentenceTransformerEmbeddingModel


@pytest.mark.skipif(not openai_imports_successful, reason='OpenAI not installed')
@pytest.mark.vcr
class TestOpenAI:
    @pytest.fixture
    def embedder(self, openai_api_key: str) -> Embedder:
        return Embedder(OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key=openai_api_key)))

    async def test_infer_model(self, openai_api_key: str):
        with patch.dict(os.environ, {'OPENAI_API_KEY': openai_api_key}):
            model = infer_embedding_model('openai:text-embedding-3-small')
        assert isinstance(model, OpenAIEmbeddingModel)
        assert model.model_name == 'text-embedding-3-small'
        assert model.system == 'openai'
        assert model.base_url == 'https://api.openai.com/v1/'

    async def test_infer_model_azure(self):
        with patch.dict(
            os.environ,
            {
                'AZURE_OPENAI_API_KEY': 'azure-openai-api-key',
                'AZURE_OPENAI_ENDPOINT': 'https://project-id.openai.azure.com/',
                'OPENAI_API_VERSION': '2023-03-15-preview',
            },
        ):
            model = infer_embedding_model('azure:text-embedding-3-small')
        assert isinstance(model, OpenAIEmbeddingModel)
        assert model.model_name == 'text-embedding-3-small'
        assert model.system == 'azure'
        assert 'azure.com' in model.base_url

    async def test_query(self, embedder: Embedder):
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1536), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(input_tokens=4),
                model_name='text-embedding-3-small',
                timestamp=IsDatetime(),
                provider_name='openai',
            )
        )
        assert result.cost().total_price == snapshot(Decimal('8E-8'))

    async def test_documents(self, embedder: Embedder):
        result = await embedder.embed_documents(['hello', 'world'])
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1536), length=2),
                inputs=['hello', 'world'],
                input_type='document',
                usage=RequestUsage(input_tokens=2),
                model_name='text-embedding-3-small',
                timestamp=IsDatetime(),
                provider_name='openai',
            )
        )
        assert result.cost().total_price == snapshot(Decimal('4E-8'))

    async def test_max_input_tokens(self, embedder: Embedder):
        max_input_tokens = await embedder.max_input_tokens()
        assert max_input_tokens == snapshot(8192)

    async def test_count_tokens(self, embedder: Embedder):
        count = await embedder.count_tokens('Hello, world!')
        assert count == snapshot(4)

    async def test_embed_error(self, openai_api_key: str):
        model = OpenAIEmbeddingModel('nonexistent', provider=OpenAIProvider(api_key=openai_api_key))
        embedder = Embedder(model)
        with pytest.raises(ModelHTTPError, match='model_not_found'):
            await embedder.embed_query('Hello, world!')

    async def test_instrumentation(self, openai_api_key: str, capfire: CaptureLogfire):
        model = OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key=openai_api_key))
        embedder = Embedder(model, instrument=True)
        await embedder.embed_query('Hello, world!')

        assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
            [
                {
                    'name': 'embeddings text-embedding-3-small',
                    'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                    'parent': None,
                    'start_time': 1000000000,
                    'end_time': 2000000000,
                    'attributes': {
                        'gen_ai.operation.name': 'embeddings',
                        'gen_ai.provider.name': 'openai',
                        'gen_ai.request.model': 'text-embedding-3-small',
                        'input_type': 'query',
                        'server.address': 'api.openai.com',
                        'inputs_count': 1,
                        'inputs': ['Hello, world!'],
                        'logfire.json_schema': {
                            'type': 'object',
                            'properties': {
                                'input_type': {'type': 'string'},
                                'inputs_count': {'type': 'integer'},
                                'embedding_settings': {'type': 'object'},
                                'inputs': {'type': ['array']},
                                'embeddings': {'type': 'array'},
                            },
                        },
                        'logfire.span_type': 'span',
                        'logfire.msg': 'embeddings text-embedding-3-small',
                        'gen_ai.usage.input_tokens': 4,
                        'operation.cost': 8e-08,
                        'gen_ai.response.model': 'text-embedding-3-small',
                        'gen_ai.embeddings.dimension.count': 1536,
                    },
                }
            ]
        )

        assert capfire.get_collected_metrics() == snapshot(
            [
                {
                    'name': 'gen_ai.client.token.usage',
                    'description': 'Measures number of input and output tokens used',
                    'unit': '{token}',
                    'data': {
                        'data_points': [
                            {
                                'attributes': {
                                    'gen_ai.provider.name': 'openai',
                                    'gen_ai.operation.name': 'embeddings',
                                    'gen_ai.request.model': 'text-embedding-3-small',
                                    'gen_ai.response.model': 'text-embedding-3-small',
                                    'gen_ai.token.type': 'input',
                                },
                                'start_time_unix_nano': IsInt(),
                                'time_unix_nano': IsInt(),
                                'count': 1,
                                'sum': 4,
                                'scale': 20,
                                'zero_count': 0,
                                'positive': {'offset': 2097151, 'bucket_counts': [1]},
                                'negative': {'offset': 0, 'bucket_counts': [0]},
                                'flags': 0,
                                'min': 4,
                                'max': 4,
                                'exemplars': [],
                            }
                        ],
                        'aggregation_temporality': 1,
                    },
                },
                {
                    'name': 'operation.cost',
                    'description': 'Monetary cost',
                    'unit': '{USD}',
                    'data': {
                        'data_points': [
                            {
                                'attributes': {
                                    'gen_ai.provider.name': 'openai',
                                    'gen_ai.operation.name': 'embeddings',
                                    'gen_ai.request.model': 'text-embedding-3-small',
                                    'gen_ai.response.model': 'text-embedding-3-small',
                                    'gen_ai.token.type': 'input',
                                },
                                'start_time_unix_nano': IsInt(),
                                'time_unix_nano': IsInt(),
                                'count': 1,
                                'sum': 8e-08,
                                'scale': 20,
                                'zero_count': 0,
                                'positive': {'offset': -24720625, 'bucket_counts': [1]},
                                'negative': {'offset': 0, 'bucket_counts': [0]},
                                'flags': 0,
                                'min': 8e-08,
                                'max': 8e-08,
                                'exemplars': [],
                            }
                        ],
                        'aggregation_temporality': 1,
                    },
                },
            ]
        )


@pytest.mark.skipif(not cohere_imports_successful, reason='Cohere not installed')
@pytest.mark.vcr
class TestCohere:
    async def test_infer_model(self, co_api_key: str):
        with patch.dict(os.environ, {'CO_API_KEY': co_api_key}):
            model = infer_embedding_model('cohere:embed-v4.0')
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
                embeddings=IsList(IsList(IsFloat(), length=1536), length=2),
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

    async def test_embed_error(self, co_api_key: str):
        model = CohereEmbeddingModel('nonexistent', provider=CohereProvider(api_key=co_api_key))
        embedder = Embedder(model)
        with pytest.raises(ModelHTTPError, match='not found,'):
            await embedder.embed_query('Hello, world!')


@pytest.mark.skipif(not sentence_transformers_imports_successful, reason='SentenceTransformers not installed')
class TestSentenceTransformers:
    @pytest.fixture(scope='session')
    def stsb_bert_tiny_model(self):
        model = SentenceTransformer('sentence-transformers-testing/stsb-bert-tiny-safetensors')
        model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
        return model

    @pytest.fixture
    def embedder(self, stsb_bert_tiny_model: Any) -> Embedder:
        return Embedder(SentenceTransformerEmbeddingModel(stsb_bert_tiny_model))

    async def test_infer_model(self):
        model = infer_embedding_model('sentence-transformers:all-MiniLM-L6-v2')
        assert isinstance(model, SentenceTransformerEmbeddingModel)
        assert model.model_name == 'all-MiniLM-L6-v2'
        assert model.system == 'sentence-transformers'
        assert model.base_url is None

    async def test_query(self, embedder: Embedder):
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=128), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                model_name='sentence-transformers-testing/stsb-bert-tiny-safetensors',
                timestamp=IsDatetime(),
                provider_name='sentence-transformers',
            )
        )

    async def test_documents(self, embedder: Embedder):
        result = await embedder.embed_documents(['hello', 'world'])
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=128), length=2),
                inputs=['hello', 'world'],
                input_type='document',
                model_name='sentence-transformers-testing/stsb-bert-tiny-safetensors',
                timestamp=IsDatetime(),
                provider_name='sentence-transformers',
            )
        )

    async def test_max_input_tokens(self, embedder: Embedder):
        max_input_tokens = await embedder.max_input_tokens()
        assert max_input_tokens == snapshot(512)

    async def test_count_tokens(self, embedder: Embedder):
        count = await embedder.count_tokens('Hello, world!')
        assert count == snapshot(6)


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

    known_model_names = sorted(get_args(KnownEmbeddingModelName.__value__))
    if generated_names != known_model_names:
        errors: list[str] = []
        missing_names = set(generated_names) - set(known_model_names)
        if missing_names:
            errors.append(f'Missing names: {missing_names}')
        extra_names = set(known_model_names) - set(generated_names)
        if extra_names:
            errors.append(f'Extra names: {extra_names}')
        raise AssertionError('\n'.join(errors))
