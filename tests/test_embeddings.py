from __future__ import annotations

import os
from collections.abc import Iterator
from decimal import Decimal
from typing import Any, get_args
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inline_snapshot import snapshot

from pydantic_ai.embeddings import (
    Embedder,
    EmbeddingResult,
    EmbeddingSettings,
    InstrumentedEmbeddingModel,
    KnownEmbeddingModelName,
    TestEmbeddingModel,
    infer_embedding_model,
)
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsFloat, IsInt, IsList, IsStr, try_import

pytestmark = [
    pytest.mark.anyio,
]

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire

with try_import() as openai_imports_successful:
    from pydantic_ai.embeddings.openai import LatestOpenAIEmbeddingModelNames, OpenAIEmbeddingModel
    from pydantic_ai.providers.gateway import GATEWAY_BASE_URL
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as cohere_imports_successful:
    from pydantic_ai.embeddings.cohere import (
        CohereEmbeddingModel,
        CohereEmbeddingSettings,
        LatestCohereEmbeddingModelNames,
    )
    from pydantic_ai.providers.cohere import CohereProvider

with try_import() as google_imports_successful:
    from pydantic_ai.embeddings.google import (
        GoogleEmbeddingModel,
        GoogleEmbeddingSettings,
        LatestGoogleGLAEmbeddingModelNames,
        LatestGoogleVertexEmbeddingModelNames,
    )
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as voyageai_imports_successful:
    from pydantic_ai.embeddings.voyageai import (
        LatestVoyageAIEmbeddingModelNames,
        VoyageAIEmbeddingModel,
        VoyageAIEmbeddingSettings,
    )
    from pydantic_ai.providers.voyageai import VoyageAIProvider

with try_import() as sentence_transformers_imports_successful:
    from sentence_transformers import SentenceTransformer

    from pydantic_ai.embeddings.sentence_transformers import SentenceTransformerEmbeddingModel


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
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

        assert await model.max_input_tokens() is None
        with pytest.raises(UserError, match='Counting tokens is not supported for non-OpenAI embedding models'):
            await model.count_tokens('Hello, world!')

    async def test_infer_model_gateway(self):
        with patch.dict(
            os.environ,
            {'PYDANTIC_AI_GATEWAY_API_KEY': 'test-api-key', 'PYDANTIC_AI_GATEWAY_BASE_URL': GATEWAY_BASE_URL},
        ):
            model = infer_embedding_model('gateway/openai:text-embedding-3-small')
        assert isinstance(model, OpenAIEmbeddingModel)
        assert model.model_name == 'text-embedding-3-small'
        assert model.system == 'openai'
        assert 'gateway.pydantic.dev' in model.base_url

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

    async def test_response_with_no_usage(self):
        mock_client = AsyncMock()
        mock_embedding_item = MagicMock()
        mock_embedding_item.embedding = [0.1, 0.2, 0.3]

        mock_response = MagicMock()
        mock_response.data = [mock_embedding_item]
        mock_response.usage = None
        mock_response.model = 'test-model'

        mock_client.embeddings.create.return_value = mock_response

        provider = OpenAIProvider(openai_client=mock_client)
        model = OpenAIEmbeddingModel('test-model', provider=provider)

        result = await model.embed('test', input_type='query')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=[[0.1, 0.2, 0.3]],
                inputs=['test'],
                input_type='query',
                model_name='test-model',
                provider_name='openai',
                timestamp=IsDatetime(),
            )
        )

    @pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
    async def test_instrumentation(self, openai_api_key: str, capfire: CaptureLogfire):
        model = OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key=openai_api_key))
        embedder = Embedder(model, instrument=True)
        await embedder.embed_query('Hello, world!', settings={'dimensions': 128})

        spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
        span = next(span for span in spans if 'embeddings' in span['name'])

        assert span == snapshot(
            {
                'name': 'embeddings text-embedding-3-small',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': IsInt(),
                'end_time': IsInt(),
                'attributes': {
                    'gen_ai.operation.name': 'embeddings',
                    'gen_ai.provider.name': 'openai',
                    'gen_ai.request.model': 'text-embedding-3-small',
                    'input_type': 'query',
                    'server.address': 'api.openai.com',
                    'inputs_count': 1,
                    'embedding_settings': {'dimensions': 128},
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
                    'gen_ai.embeddings.dimension.count': 128,
                },
            }
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


@pytest.mark.skipif(not cohere_imports_successful(), reason='Cohere not installed')
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
        assert result.cost().total_price == snapshot(Decimal('4.8E-7'))

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
        assert result.cost().total_price == snapshot(Decimal('2.4E-7'))

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

    async def test_query_with_cohere_truncate(self, co_api_key: str):
        model = CohereEmbeddingModel('embed-v4.0', provider=CohereProvider(api_key=co_api_key))
        embedder = Embedder(model)
        settings: CohereEmbeddingSettings = {'cohere_truncate': 'END'}
        result = await embedder.embed_query('Hello, world!', settings=settings)
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1536), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(input_tokens=4),
                model_name='embed-v4.0',
                timestamp=IsDatetime(),
                provider_name='cohere',
                provider_response_id=IsStr(),
            )
        )

    async def test_query_with_truncate(self, co_api_key: str):
        model = CohereEmbeddingModel('embed-v4.0', provider=CohereProvider(api_key=co_api_key))
        embedder = Embedder(model)
        result = await embedder.embed_query('Hello, world!', settings={'truncate': True})
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1536), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(input_tokens=4),
                model_name='embed-v4.0',
                timestamp=IsDatetime(),
                provider_name='cohere',
                provider_response_id=IsStr(),
            )
        )


@pytest.mark.skipif(not voyageai_imports_successful(), reason='VoyageAI not installed')
@pytest.mark.vcr
class TestVoyageAI:
    async def test_infer_model(self, voyage_api_key: str):
        with patch.dict(os.environ, {'VOYAGE_API_KEY': voyage_api_key}):
            model = infer_embedding_model('voyageai:voyage-3.5')
        assert isinstance(model, VoyageAIEmbeddingModel)
        assert model.model_name == 'voyage-3.5'
        assert model.system == 'voyageai'
        assert model.base_url == 'https://api.voyageai.com/v1'
        assert isinstance(model._provider, VoyageAIProvider)  # type: ignore[reportAttributeAccess]

    async def test_query(self, voyage_api_key: str):
        model = VoyageAIEmbeddingModel('voyage-3.5', provider=VoyageAIProvider(api_key=voyage_api_key))
        embedder = Embedder(model)
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(input_tokens=3),
                model_name='voyage-3.5',
                timestamp=IsDatetime(),
                provider_name='voyageai',
            )
        )

    async def test_query_voyage_4(self, voyage_api_key: str):
        model = VoyageAIEmbeddingModel('voyage-4', provider=VoyageAIProvider(api_key=voyage_api_key))
        embedder = Embedder(model)
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(input_tokens=3),
                model_name='voyage-4',
                timestamp=IsDatetime(),
                provider_name='voyageai',
            )
        )

    async def test_documents(self, voyage_api_key: str):
        model = VoyageAIEmbeddingModel('voyage-3.5', provider=VoyageAIProvider(api_key=voyage_api_key))
        embedder = Embedder(model)
        result = await embedder.embed_documents(['hello', 'world'])
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=2),
                inputs=['hello', 'world'],
                input_type='document',
                usage=RequestUsage(),
                model_name='voyage-3.5',
                timestamp=IsDatetime(),
                provider_name='voyageai',
            )
        )

    async def test_max_input_tokens(self, voyage_api_key: str):
        model = VoyageAIEmbeddingModel('voyage-3.5', provider=VoyageAIProvider(api_key=voyage_api_key))
        embedder = Embedder(model)
        max_input_tokens = await embedder.max_input_tokens()
        assert max_input_tokens == snapshot(32000)

    async def test_embed_error(self, voyage_api_key: str):
        model = VoyageAIEmbeddingModel('nonexistent', provider=VoyageAIProvider(api_key=voyage_api_key))
        embedder = Embedder(model)
        with pytest.raises(ModelAPIError, match='not supported'):
            await embedder.embed_query('Hello, world!')

    async def test_query_with_truncate(self, voyage_api_key: str):
        model = VoyageAIEmbeddingModel('voyage-3.5', provider=VoyageAIProvider(api_key=voyage_api_key))
        embedder = Embedder(model)
        result = await embedder.embed_query('Hello, world!', settings={'truncate': True})
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(input_tokens=3),
                model_name='voyage-3.5',
                timestamp=IsDatetime(),
                provider_name='voyageai',
            )
        )

    async def test_query_with_voyageai_input_type(self, voyage_api_key: str):
        model = VoyageAIEmbeddingModel('voyage-3.5', provider=VoyageAIProvider(api_key=voyage_api_key))
        embedder = Embedder(model)
        settings: VoyageAIEmbeddingSettings = {'voyageai_input_type': 'none'}
        result = await embedder.embed_query('Hello, world!', settings=settings)
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(input_tokens=4),
                model_name='voyage-3.5',
                timestamp=IsDatetime(),
                provider_name='voyageai',
            )
        )


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
@pytest.mark.vcr
class TestGoogle:
    @pytest.fixture
    def embedder(self, gemini_api_key: str) -> Embedder:
        return Embedder(GoogleEmbeddingModel('gemini-embedding-001', provider=GoogleProvider(api_key=gemini_api_key)))

    async def test_infer_model_gla(self, gemini_api_key: str):
        with patch.dict(os.environ, {'GOOGLE_API_KEY': gemini_api_key}):
            model = infer_embedding_model('google-gla:gemini-embedding-001')
        assert isinstance(model, GoogleEmbeddingModel)
        assert model.model_name == 'gemini-embedding-001'
        assert model.system == 'google-gla'
        assert 'generativelanguage.googleapis.com' in model.base_url

    async def test_infer_model_vertex(self):
        # Vertex AI requires project setup, so we just test the model creation
        # without actually calling the API
        with patch.dict(
            os.environ,
            {
                'GOOGLE_API_KEY': 'mock-api-key',
            },
        ):
            model = infer_embedding_model('google-vertex:gemini-embedding-001')
        assert isinstance(model, GoogleEmbeddingModel)
        assert model.model_name == 'gemini-embedding-001'
        assert model.system == 'google-vertex'

    async def test_model_with_string_provider(self, gemini_api_key: str):
        with patch.dict(os.environ, {'GOOGLE_API_KEY': gemini_api_key}):
            model = GoogleEmbeddingModel('gemini-embedding-001', provider='google-gla')
        assert isinstance(model, GoogleEmbeddingModel)
        assert model.model_name == 'gemini-embedding-001'
        assert model.system == 'google-gla'

    async def test_query(self, embedder: Embedder):
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=3072), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(),
                model_name='gemini-embedding-001',
                timestamp=IsDatetime(),
                provider_name='google-gla',
            )
        )

    async def test_documents(self, embedder: Embedder):
        result = await embedder.embed_documents(['hello', 'world'])
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=3072), length=2),
                inputs=['hello', 'world'],
                input_type='document',
                usage=RequestUsage(),
                model_name='gemini-embedding-001',
                timestamp=IsDatetime(),
                provider_name='google-gla',
            )
        )

    async def test_query_with_dimensions(self, embedder: Embedder):
        result = await embedder.embed_query('Hello, world!', settings={'dimensions': 768})
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=768), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(),
                model_name='gemini-embedding-001',
                timestamp=IsDatetime(),
                provider_name='google-gla',
            )
        )

    async def test_max_input_tokens(self, embedder: Embedder):
        max_input_tokens = await embedder.max_input_tokens()
        assert max_input_tokens == snapshot(2048)

    async def test_count_tokens(self, embedder: Embedder):
        count = await embedder.count_tokens('Hello, world!')
        assert count == snapshot(5)

    async def test_embed_error(self, gemini_api_key: str):
        model = GoogleEmbeddingModel('nonexistent-model', provider=GoogleProvider(api_key=gemini_api_key))
        embedder = Embedder(model)
        with pytest.raises(ModelHTTPError, match='not found'):
            await embedder.embed_query('Hello, world!')

    async def test_count_tokens_error(self, gemini_api_key: str):
        model = GoogleEmbeddingModel('nonexistent-model', provider=GoogleProvider(api_key=gemini_api_key))
        embedder = Embedder(model)
        with pytest.raises(ModelHTTPError, match='not found'):
            await embedder.count_tokens('Hello, world!')

    async def test_query_with_task_type(self, embedder: Embedder):
        result = await embedder.embed_query(
            'Hello, world!', settings=GoogleEmbeddingSettings(google_task_type='RETRIEVAL_QUERY')
        )
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=3072), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(),
                model_name='gemini-embedding-001',
                timestamp=IsDatetime(),
                provider_name='google-gla',
            )
        )

    @pytest.mark.skipif(
        not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
    )
    @pytest.mark.vcr()
    async def test_vertex_query(
        self, allow_model_requests: None, vertex_provider: GoogleProvider
    ):  # pragma: lax no cover
        model = GoogleEmbeddingModel('gemini-embedding-001', provider=vertex_provider)
        embedder = Embedder(model)
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=3072), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(input_tokens=4),
                model_name='gemini-embedding-001',
                timestamp=IsDatetime(),
                provider_name='google-vertex',
            )
        )

    @pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
    async def test_instrumentation(self, gemini_api_key: str, capfire: CaptureLogfire):
        model = GoogleEmbeddingModel('gemini-embedding-001', provider=GoogleProvider(api_key=gemini_api_key))
        embedder = Embedder(model, instrument=True)
        await embedder.embed_query('Hello, world!', settings={'dimensions': 768})

        spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
        span = next(span for span in spans if 'embeddings' in span['name'])

        assert span == snapshot(
            {
                'name': 'embeddings gemini-embedding-001',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': IsInt(),
                'end_time': IsInt(),
                'attributes': {
                    'gen_ai.operation.name': 'embeddings',
                    'gen_ai.provider.name': 'google-gla',
                    'gen_ai.request.model': 'gemini-embedding-001',
                    'input_type': 'query',
                    'server.address': 'generativelanguage.googleapis.com',
                    'inputs_count': 1,
                    'embedding_settings': {'dimensions': 768},
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
                    'logfire.msg': 'embeddings gemini-embedding-001',
                    'gen_ai.response.model': 'gemini-embedding-001',
                    'operation.cost': 0.0,
                    'gen_ai.embeddings.dimension.count': 768,
                },
            }
        )


@pytest.mark.skipif(not sentence_transformers_imports_successful(), reason='SentenceTransformers not installed')
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
    not openai_imports_successful()
    or not cohere_imports_successful()
    or not google_imports_successful()
    or not voyageai_imports_successful(),
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
    google_gla_names = [f'google-gla:{n}' for n in get_model_names(LatestGoogleGLAEmbeddingModelNames)]
    google_vertex_names = [f'google-vertex:{n}' for n in get_model_names(LatestGoogleVertexEmbeddingModelNames)]
    voyageai_names = [f'voyageai:{n}' for n in get_model_names(LatestVoyageAIEmbeddingModelNames)]

    generated_names = sorted(openai_names + cohere_names + google_gla_names + google_vertex_names + voyageai_names)

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


def test_infer_model_error():
    with pytest.raises(ValueError, match='You must provide a provider prefix when specifying an embedding model name'):
        infer_embedding_model('nonexistent')


async def test_instrument_all():
    model = TestEmbeddingModel()
    embedder = Embedder(model)

    def get_model():
        return embedder._get_model()  # pyright: ignore[reportPrivateUsage]

    Embedder.instrument_all(False)
    assert get_model() is model

    Embedder.instrument_all()
    m = get_model()
    assert isinstance(m, InstrumentedEmbeddingModel)
    assert m.wrapped is model
    assert m.instrumentation_settings.event_mode == InstrumentationSettings().event_mode

    assert m.model_name == model.model_name
    assert m.system == model.system
    assert m.base_url == model.base_url
    assert m.settings == model.settings

    assert (await m.embed('Hello, world!', input_type='query')).embeddings == (
        await model.embed('Hello, world!', input_type='query')
    ).embeddings
    assert await m.max_input_tokens() == await model.max_input_tokens()
    assert await m.count_tokens('Hello, world!') == await model.count_tokens('Hello, world!')

    options = InstrumentationSettings(version=1, event_mode='logs')
    Embedder.instrument_all(options)
    m = get_model()
    assert isinstance(m, InstrumentedEmbeddingModel)
    assert m.wrapped is model
    assert m.instrumentation_settings is options

    Embedder.instrument_all(False)
    assert get_model() is model


def test_override():
    model = TestEmbeddingModel()
    embedder = Embedder(model)

    model2 = TestEmbeddingModel()

    with embedder.override(model=model2):
        assert embedder._get_model() is model2  # pyright: ignore[reportPrivateUsage]

    with embedder.override():
        assert embedder._get_model() is model  # pyright: ignore[reportPrivateUsage]

    assert embedder._get_model() is model  # pyright: ignore[reportPrivateUsage]


def test_sync():
    model = TestEmbeddingModel()
    embedder = Embedder(model)

    result = embedder.embed_query_sync('Hello, world!')
    assert isinstance(result, EmbeddingResult)

    result = embedder.embed_documents_sync(['hello', 'world'])
    assert isinstance(result, EmbeddingResult)

    result = embedder.embed_sync('Hello, world!', input_type='query')
    assert isinstance(result, EmbeddingResult)

    result = embedder.max_input_tokens_sync()
    assert isinstance(result, int)

    result = embedder.count_tokens_sync('Hello, world!')
    assert isinstance(result, int)


async def test_settings():
    model_settings: EmbeddingSettings = {'dimensions': 128, 'from_model': True}  # pyright: ignore[reportAssignmentType]
    model = TestEmbeddingModel(settings=model_settings)
    assert model.settings == model_settings
    await Embedder(model).embed_query('Hello, world!')
    assert model.last_settings == snapshot({'dimensions': 128, 'from_model': True})

    embedder_settings: EmbeddingSettings = {'dimensions': 256, 'from_embedder': True}  # pyright: ignore[reportAssignmentType]
    embedder = Embedder(model, settings=embedder_settings)
    await embedder.embed_query('Hello, world!')
    assert model.last_settings == snapshot({'dimensions': 256, 'from_model': True, 'from_embedder': True})

    embed_settings: EmbeddingSettings = {'dimensions': 512, 'from_embed': True}  # pyright: ignore[reportAssignmentType]
    await embedder.embed_query('Hello, world!', settings=embed_settings)
    assert model.last_settings == snapshot(
        {'dimensions': 512, 'from_model': True, 'from_embedder': True, 'from_embed': True}
    )


def test_result():
    result = EmbeddingResult(
        embeddings=[[-1.0], [-0.5], [0.0], [0.5], [1.0]],
        inputs=['a', 'b', 'c', 'd', 'e'],
        input_type='document',
        model_name='test',
        timestamp=IsDatetime(),
        provider_name='test',
    )
    assert result[0] == result['a'] == snapshot([-1.0])
    assert result[1] == result['b'] == snapshot([-0.5])
    assert result[2] == result['c'] == snapshot([0.0])
    assert result[3] == result['d'] == snapshot([0.5])
    assert result[4] == result['e'] == snapshot([1.0])


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_limited_instrumentation(capfire: CaptureLogfire):
    model = TestEmbeddingModel()
    embedder = Embedder(model, instrument=InstrumentationSettings(include_content=False))
    await embedder.embed_query('Hello, world!')

    assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'embeddings test',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': IsInt(),
                'end_time': IsInt(),
                'attributes': {
                    'gen_ai.operation.name': 'embeddings',
                    'gen_ai.provider.name': 'test',
                    'gen_ai.request.model': 'test',
                    'input_type': 'query',
                    'inputs_count': 1,
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'input_type': {'type': 'string'},
                            'inputs_count': {'type': 'integer'},
                            'embedding_settings': {'type': 'object'},
                        },
                    },
                    'logfire.span_type': 'span',
                    'logfire.msg': 'embeddings test',
                    'gen_ai.usage.input_tokens': 2,
                    'gen_ai.response.model': 'test',
                    'gen_ai.embeddings.dimension.count': 8,
                    'gen_ai.response.id': IsStr(),
                },
            }
        ]
    )
