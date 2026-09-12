from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, get_args
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import anyio
import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pytest_mock import MockerFixture

import pydantic_ai.models

from ._inline_snapshot import snapshot

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup as ExceptionGroup  # pragma: lax no cover
else:
    ExceptionGroup = ExceptionGroup  # pragma: lax no cover

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

from .conftest import IsDatetime, IsFloat, IsInt, IsList, IsStr, TestEnv, try_import

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.usefixtures('allow_model_requests'),
]

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire

with try_import() as openai_imports_successful:
    from pydantic_ai.embeddings.openai import LatestOpenAIEmbeddingModelNames, OpenAIEmbeddingModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as cohere_imports_successful:
    from pydantic_ai.embeddings.cohere import (
        CohereEmbeddingModel,
        CohereEmbeddingSettings,
        LatestCohereEmbeddingModelNames,
    )
    from pydantic_ai.providers.cohere import CohereProvider

with try_import() as bedrock_imports_successful:
    from botocore.exceptions import ClientError

    from pydantic_ai.embeddings.bedrock import (
        BedrockEmbeddingModel,
        BedrockEmbeddingSettings,
        LatestBedrockEmbeddingModelNames,
    )
    from pydantic_ai.providers.bedrock import BedrockProvider

with try_import() as google_imports_successful:
    from pydantic_ai.embeddings.google import (
        GoogleEmbeddingModel,
        GoogleEmbeddingSettings,
        LatestGoogleGLAEmbeddingModelNames,
        LatestGoogleVertexEmbeddingModelNames,
    )
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

with try_import() as voyageai_imports_successful:
    from pydantic_ai.embeddings.voyageai import (
        LatestVoyageAIEmbeddingModelNames,
        VoyageAIEmbeddingModel,
        VoyageAIEmbeddingSettings,
    )
    from pydantic_ai.providers.voyageai import VoyageAIProvider

with try_import() as sentence_transformers_imports_successful:
    import torch
    from sentence_transformers import SentenceTransformer

    import pydantic_ai.embeddings.sentence_transformers as sentence_transformers_module
    from pydantic_ai.embeddings.sentence_transformers import (
        SentenceTransformerEmbeddingModel,
        SentenceTransformersEmbeddingSettings,
    )


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_embedding_model_blocks_requests_when_disabled():
    model = OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key='test-key'))

    with pydantic_ai.models.override_allow_model_requests(False):
        with pytest.raises(RuntimeError, match='Model requests are not allowed'):
            await model.embed('hello', input_type='query')


@pytest.mark.skipif(not cohere_imports_successful(), reason='cohere not installed')
async def test_cohere_embedding_model_blocks_requests_when_disabled():
    model = CohereEmbeddingModel('embed-v4.0', provider=CohereProvider(api_key='test-key'))

    with pydantic_ai.models.override_allow_model_requests(False):
        with pytest.raises(RuntimeError, match='Model requests are not allowed'):
            await model.embed('hello', input_type='query')


@pytest.mark.skipif(not google_imports_successful(), reason='google not installed')
async def test_google_embedding_model_blocks_requests_when_disabled():
    model = GoogleEmbeddingModel('gemini-embedding-001', provider=GoogleProvider(api_key='test-key'))

    with pydantic_ai.models.override_allow_model_requests(False):
        with pytest.raises(RuntimeError, match='Model requests are not allowed'):
            await model.embed('hello', input_type='query')


@pytest.mark.skipif(not bedrock_imports_successful(), reason='bedrock not installed')
async def test_bedrock_embedding_model_blocks_requests_when_disabled():
    model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=BedrockProvider(bedrock_client=MagicMock()))

    with pydantic_ai.models.override_allow_model_requests(False):
        with pytest.raises(RuntimeError, match='Model requests are not allowed'):
            await model.embed('hello', input_type='query')


@pytest.mark.skipif(not voyageai_imports_successful(), reason='voyageai not installed')
async def test_voyageai_embedding_model_blocks_requests_when_disabled():
    model = VoyageAIEmbeddingModel('voyage-4', provider=VoyageAIProvider(api_key='test-key'))

    with pydantic_ai.models.override_allow_model_requests(False):
        with pytest.raises(RuntimeError, match='Model requests are not allowed'):
            await model.embed('hello', input_type='query')


@pytest.mark.skipif(not google_imports_successful(), reason='google not installed')
async def test_google_embedding_model_blocks_count_tokens_when_disabled():
    model = GoogleEmbeddingModel('gemini-embedding-001', provider=GoogleProvider(api_key='test-key'))

    with pydantic_ai.models.override_allow_model_requests(False):
        with pytest.raises(RuntimeError, match='Model requests are not allowed'):
            await model.count_tokens('hello')


@pytest.mark.skipif(not cohere_imports_successful(), reason='cohere not installed')
async def test_cohere_embedding_model_blocks_count_tokens_when_disabled():
    model = CohereEmbeddingModel('embed-v4.0', provider=CohereProvider(api_key='test-key'))

    with pydantic_ai.models.override_allow_model_requests(False):
        with pytest.raises(RuntimeError, match='Model requests are not allowed'):
            await model.count_tokens('hello')


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_embedder_blocks_requests_when_disabled():
    """Pins the guard on the public `Embedder` surface, which is what issue #6763 reports as leaking.

    A guard that fires before the request is made can't be exercised through a VCR recording.
    The per-model tests above prove each `embed()` guards; this proves the wrapper chain
    (`Embedder` -> `InstrumentedEmbeddingModel` / `WrapperEmbeddingModel` -> concrete model)
    surfaces the `RuntimeError` rather than swallowing or wrapping it.
    """
    embedder = Embedder(OpenAIEmbeddingModel('text-embedding-3-small', provider=OpenAIProvider(api_key='test-key')))

    with pydantic_ai.models.override_allow_model_requests(False):
        with pytest.raises(RuntimeError, match='Model requests are not allowed'):
            await embedder.embed_query('hello')


async def test_test_embedding_model_is_exempt_from_request_guard():
    """`ALLOW_MODEL_REQUESTS`'s docstring promises `TestEmbeddingModel` is unaffected; pin that promise.

    Without this, adding the guard to `TestEmbeddingModel.embed` would break every user's test
    suite while the whole file still passed, since the module-level `allow_model_requests`
    fixture keeps the flag on for every other test here.
    """
    embedder = Embedder(TestEmbeddingModel())

    with pydantic_ai.models.override_allow_model_requests(False):
        result = await embedder.embed_query('hello')

    assert result.embeddings == snapshot([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])


async def test_test_embedding_model_counts_blank_input_as_zero_tokens():
    """Blank input reports no tokens on both methods.

    The estimator behind them is shared with the other test models, and it counts blank text as one
    token; `TestEmbeddingModel` guards that, so a blank input contributes nothing to the reported
    usage instead of one phantom token per empty string in the batch.
    """
    model = TestEmbeddingModel()

    assert await model.count_tokens('') == snapshot(0)
    result = await model.embed(['', 'hi there'], input_type='document')
    assert result.usage.input_tokens == snapshot(2)


STSB_BERT_TINY_MODEL = 'sentence-transformers-testing/stsb-bert-tiny-safetensors'
# Pinned so a warm HF cache is served without revalidating files against the Hub.
# Keep in sync with the HF cache keys and warmup commands in .github/workflows/ci.yml;
# `test_stsb_model_pin_matches_ci` guards against drift.
STSB_BERT_TINY_REVISION = 'f3cb857cba53019a20df283396bcca179cf051a4'


def _hf_hub_unavailable(exc: BaseException) -> bool:
    """Whether an exception from a SentenceTransformer load means the HF Hub is
    unavailable (worth skipping the real-model smoke tests) rather than a genuine
    integration defect or a bad model pin (which must fail loudly). Shapes verified
    against real outage and bad-pin probes.
    """
    if isinstance(exc, httpx.TransportError):
        # Transport-level httpx errors (e.g. connect timeouts) escape
        # huggingface_hub 1.x unwrapped, and the only httpx traffic in a model
        # load is Hub traffic. Status-level httpx errors are not outage proof.
        return True
    if isinstance(exc, RuntimeError):
        # After a connection error, huggingface_hub 1.x retries on an httpx client
        # it just closed, surfacing httpx's RuntimeError instead of its own error.
        return 'client has been closed' in str(exc)
    if type(exc).__module__.startswith('huggingface_hub'):
        # huggingface_hub errors are OSError subclasses. Ones carrying an HTTP
        # response are an outage only for transient statuses; 401/403/404 mean a
        # bad pin or an auth problem. Ones without a response (e.g.
        # LocalEntryNotFoundError) mean the Hub couldn't be reached at all.
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        return status is None or status == 429 or status >= 500
    # transformers wraps connection failures in a plain OSError: "We couldn't
    # connect to 'https://huggingface.co' to load the files [...]". Other OSError
    # shapes (corrupt cache, invalid model identifier) must propagate.
    return "couldn't connect" in str(exc).lower()


def test_hf_hub_unavailable_classifier():
    """Exercises the fixture's outage-classification guard on synthetic exceptions:
    no HTTP is involved (hence no VCR), just shapes captured from real probes."""

    class FakeResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code

    class FakeHubError(OSError):
        def __init__(self, msg: str, status_code: int | None = None):
            super().__init__(msg)
            self.response = FakeResponse(status_code) if status_code is not None else None

    FakeHubError.__module__ = 'huggingface_hub.errors'

    # Hub unavailability: skip the real-model smoke tests.
    assert _hf_hub_unavailable(httpx.ConnectTimeout('timed out'))
    assert _hf_hub_unavailable(RuntimeError('Cannot send a request, as the client has been closed.'))
    assert _hf_hub_unavailable(FakeHubError('504 Server Error', 504))
    assert _hf_hub_unavailable(FakeHubError('429 Too Many Requests', 429))
    assert _hf_hub_unavailable(FakeHubError('cannot find the requested files in the disk cache'))
    assert _hf_hub_unavailable(OSError("We couldn't connect to 'https://huggingface.co' to load the files"))
    # Anything else fails loudly: bad pins, auth problems, local defects.
    assert not _hf_hub_unavailable(FakeHubError('401 Unauthorized', 401))
    assert not _hf_hub_unavailable(FakeHubError('404 Repository Not Found', 404))
    assert not _hf_hub_unavailable(
        httpx.HTTPStatusError('404', request=httpx.Request('GET', 'https://x'), response=httpx.Response(404))
    )
    assert not _hf_hub_unavailable(RuntimeError('CUDA error: device-side assert triggered'))
    assert not _hf_hub_unavailable(OSError('Unable to load weights from pytorch checkpoint file'))
    assert not _hf_hub_unavailable(
        OSError(
            "xyz is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'"
        )
    )


def test_stsb_model_pin_matches_ci():
    """Parses CI configuration only, no network or VCR involved: drift between this
    pin and ci.yml silently reintroduces per-run Hub downloads, because the stale
    cache key keeps exact-hitting, so CI never caches the new revision."""
    ci_yml = Path(__file__).parent.parent / '.github' / 'workflows' / 'ci.yml'
    if not ci_yml.is_file():  # pragma: lax no cover
        pytest.skip('not running from a repo checkout')
    stsb_lines = [line for line in ci_yml.read_text().splitlines() if 'stsb-bert-tiny' in line]
    cache_keys = [line for line in stsb_lines if 'key:' in line]
    warmups = [line for line in stsb_lines if 'snapshot_download' in line]
    assert len(cache_keys) >= 2, 'expected a model cache key in both test jobs in ci.yml'
    assert len(warmups) >= 2, 'expected a model warmup command in both test jobs in ci.yml'
    assert all(STSB_BERT_TINY_MODEL in line for line in warmups)
    for line in cache_keys + warmups:
        assert STSB_BERT_TINY_REVISION in line, f'stale or missing revision pin in ci.yml: {line.strip()}'
    stray = {sha for line in stsb_lines for sha in re.findall(r'[0-9a-f]{40}', line)} - {STSB_BERT_TINY_REVISION}
    assert not stray, f'stale revision pins left in ci.yml: {stray}'


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
        assert urlparse(model.base_url).hostname == 'project-id.openai.azure.com'

        assert await model.max_input_tokens() is None
        with pytest.raises(UserError, match='Counting tokens is not supported for non-OpenAI embedding models'):
            await model.count_tokens('Hello, world!')

    async def test_infer_model_gateway(self):
        with patch.dict(
            os.environ,
            {
                'PYDANTIC_AI_GATEWAY_API_KEY': 'test-api-key',
                'PYDANTIC_AI_GATEWAY_BASE_URL': 'https://gateway.pydantic.dev/proxy',
            },
        ):
            model = infer_embedding_model('gateway/openai:text-embedding-3-small')
        assert isinstance(model, OpenAIEmbeddingModel)
        assert model.model_name == 'text-embedding-3-small'
        assert model.system == 'openai'
        assert urlparse(model.base_url).hostname == 'gateway.pydantic.dev'

    async def test_infer_model_vllm(self):
        with patch.dict(os.environ, {'VLLM_BASE_URL': 'http://localhost:8000/v1'}):
            model = infer_embedding_model('vllm:intfloat/e5-mistral-7b-instruct')
        assert isinstance(model, OpenAIEmbeddingModel)
        assert model.model_name == 'intfloat/e5-mistral-7b-instruct'
        assert model.system == 'vllm'
        assert model.base_url == 'http://localhost:8000/v1/'

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
                    'embeddings': [
                        [
                            -0.05322972685098648,
                            -0.0702020674943924,
                            -0.004702363163232803,
                            0.05225988104939461,
                            -0.09385143220424652,
                            -0.05457259342074394,
                            -0.058265477418899536,
                            0.14308986067771912,
                            -0.08907679468393326,
                            -0.08430216461420059,
                            -0.005837738048285246,
                            -0.0805719792842865,
                            -0.006830899510532618,
                            -0.08728630840778351,
                            0.028647813946008682,
                            0.051364634186029434,
                            -0.12801991403102875,
                            0.11496427655220032,
                            0.0011458658846095204,
                            0.11421823501586914,
                            0.14883434772491455,
                            0.005152316763997078,
                            0.012608022429049015,
                            0.02764066495001316,
                            0.13257074356079102,
                            0.0060242475010454655,
                            -0.027118438854813576,
                            0.10675787180662155,
                            0.002327868016436696,
                            -0.14450733363628387,
                            0.14174699783325195,
                            -0.09019584953784943,
                            -0.03890582174062729,
                            -0.0350077785551548,
                            0.036928821355104446,
                            0.05155114457011223,
                            0.004410942550748587,
                            -0.0023628384806215763,
                            -0.03543674945831299,
                            -0.08236246556043625,
                            -0.012393536977469921,
                            -0.042524099349975586,
                            0.07124651968479156,
                            0.02517874352633953,
                            -0.10228165239095688,
                            0.0561019703745842,
                            -0.11294997483491898,
                            -0.007241219747811556,
                            0.09855146706104279,
                            0.13443583250045776,
                            -0.09332920610904694,
                            -0.006844887975603342,
                            0.04797016829252243,
                            0.21083000302314758,
                            0.0025645014829933643,
                            -0.11847064644098282,
                            0.02316444367170334,
                            0.2106807976961136,
                            -0.13115327060222626,
                            0.041889969259500504,
                            0.03953995183110237,
                            0.06859808415174484,
                            0.028256144374608994,
                            -0.0026507622096687555,
                            0.038383595645427704,
                            -0.027920428663492203,
                            -0.057258326560258865,
                            -0.004299037158489227,
                            -0.032433949410915375,
                            0.13712157309055328,
                            0.002949176821857691,
                            0.10481817275285721,
                            -0.05356544256210327,
                            0.016263602301478386,
                            -0.12854214012622833,
                            -0.012663975358009338,
                            -0.06199565902352333,
                            0.024451356381177902,
                            -0.0736711397767067,
                            -0.1507740467786789,
                            -0.1225738525390625,
                            0.05360274761915207,
                            -0.08579423278570175,
                            -0.16755987703800201,
                            -0.0525209940969944,
                            0.0122536551207304,
                            -0.0663599744439125,
                            -0.03329189494252205,
                            -0.04584396257996559,
                            0.006136152893304825,
                            -0.022287849336862564,
                            0.04241219535470009,
                            -0.041554249823093414,
                            0.03470936417579651,
                            0.04021138325333595,
                            -0.07669258862733841,
                            -0.0629655048251152,
                            0.1572645604610443,
                            0.16935035586357117,
                            -0.1368977576494217,
                            0.04058440402150154,
                            -0.07587194442749023,
                            0.03679826483130455,
                            -0.05714642256498337,
                            0.08281008899211884,
                            0.062480583786964417,
                            0.022698169574141502,
                            -0.11287537217140198,
                            -0.0737084373831749,
                            0.01263599842786789,
                            -0.3461610972881317,
                            0.0262418445199728,
                            0.08825615793466568,
                            -0.08616725355386734,
                            0.015993164852261543,
                            0.08571963012218475,
                            0.14868514239788055,
                            -0.10966741293668747,
                            0.06497980654239655,
                            -0.14928196370601654,
                            0.02413429133594036,
                            -0.03200497850775719,
                            0.05419957637786865,
                            -0.11608333140611649,
                            -0.11951509863138199,
                            -0.1191420778632164,
                            0.016207650303840637,
                            0.15935346484184265,
                        ]
                    ],
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


@pytest.mark.skipif(not bedrock_imports_successful(), reason='Bedrock not installed')
@pytest.mark.vcr
class TestBedrock:
    async def test_infer_model(self):
        with patch.dict(
            os.environ,
            {
                'AWS_ACCESS_KEY_ID': 'test-access-key',
                'AWS_SECRET_ACCESS_KEY': 'test-secret-key',
                'AWS_DEFAULT_REGION': 'us-east-1',
            },
        ):
            model = infer_embedding_model('bedrock:amazon.titan-embed-text-v2:0')
        assert isinstance(model, BedrockEmbeddingModel)
        assert model.model_name == 'amazon.titan-embed-text-v2:0'
        assert model.system == 'bedrock'
        assert model.base_url == 'https://bedrock-runtime.us-east-1.amazonaws.com'

    async def test_titan_v1_minimal(self, bedrock_provider: BedrockProvider):
        """Test Titan V1 with default settings (fixed 1536 dimensions)."""
        model = BedrockEmbeddingModel('amazon.titan-embed-text-v1', provider=bedrock_provider)
        embedder = Embedder(model)
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1536), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                model_name='amazon.titan-embed-text-v1',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=4),
            )
        )

    async def test_titan_v2_minimal(self, bedrock_provider: BedrockProvider):
        """Test Titan V2 with default settings (1024 dimensions, normalize=True)."""
        model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=bedrock_provider)
        embedder = Embedder(model)
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                model_name='amazon.titan-embed-text-v2:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=5),
            )
        )

    async def test_titan_v2_with_dimensions(self, bedrock_provider: BedrockProvider):
        """Test Titan V2 with custom dimensions setting."""

        model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(dimensions=256))
        result = await embedder.embed_query('Test embedding dimensions')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=256), length=1),
                inputs=['Test embedding dimensions'],
                input_type='query',
                model_name='amazon.titan-embed-text-v2:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=4),
            )
        )

    async def test_titan_v2_with_normalize_false(self, bedrock_provider: BedrockProvider):
        """Test Titan V2 with normalize=False (override default)."""

        model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(bedrock_titan_normalize=False))
        result = await embedder.embed_query('Test normalization disabled')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=1),
                inputs=['Test normalization disabled'],
                input_type='query',
                model_name='amazon.titan-embed-text-v2:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=4),
            )
        )

    async def test_titan_v2_with_normalize_true(self, bedrock_provider: BedrockProvider):
        """Test Titan V2 with explicit normalize=True setting."""

        model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(bedrock_titan_normalize=True))
        result = await embedder.embed_query('Test explicit normalization')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=1),
                inputs=['Test explicit normalization'],
                input_type='query',
                model_name='amazon.titan-embed-text-v2:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=4),
            )
        )

    async def test_titan_v2_multiple_texts(self, bedrock_provider: BedrockProvider):
        """Test Titan V2 document embedding (multiple texts, sequential requests)."""
        model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=bedrock_provider)
        # Use max_concurrency=1 to ensure deterministic request order for VCRpy
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(bedrock_max_concurrency=1))
        result = await embedder.embed_documents(['hello', 'world'])
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=2),
                inputs=['hello', 'world'],
                input_type='document',
                model_name='amazon.titan-embed-text-v2:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=4),
            )
        )

    @pytest.mark.parametrize('max_concurrency', [0, -1])
    async def test_titan_v2_rejects_invalid_max_concurrency(
        self, bedrock_provider: BedrockProvider, max_concurrency: int
    ):
        model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(bedrock_max_concurrency=max_concurrency))

        with (
            anyio.fail_after(1),
            pytest.raises(UserError, match=f'bedrock_max_concurrency must be >= 1, got {max_concurrency}'),
        ):
            await embedder.embed_query('hello')

    async def test_cohere_v3_minimal(self, bedrock_provider: BedrockProvider):
        """Test Cohere V3 with default settings (1024 dimensions, truncate=NONE)."""
        model = BedrockEmbeddingModel('cohere.embed-english-v3', provider=bedrock_provider)
        embedder = Embedder(model)
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                model_name='cohere.embed-english-v3',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=4),
                provider_response_id=IsStr(),
            )
        )

    async def test_cohere_v3_with_input_type(self, bedrock_provider: BedrockProvider):
        """Test Cohere V3 with custom input_type setting."""

        model = BedrockEmbeddingModel('cohere.embed-english-v3', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(bedrock_cohere_input_type='classification'))
        result = await embedder.embed_query('Test input type setting')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=1),
                inputs=['Test input type setting'],
                input_type='query',
                model_name='cohere.embed-english-v3',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=4),
                provider_response_id=IsStr(),
            )
        )

    async def test_cohere_v3_with_truncate(self, bedrock_provider: BedrockProvider):
        """Test Cohere V3 with custom truncate setting."""

        model = BedrockEmbeddingModel('cohere.embed-english-v3', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(bedrock_cohere_truncate='END'))
        result = await embedder.embed_query('Test truncation setting')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=1),
                inputs=['Test truncation setting'],
                input_type='query',
                model_name='cohere.embed-english-v3',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=5),
                provider_response_id=IsStr(),
            )
        )

    async def test_cohere_v3_with_base_truncate(self, bedrock_provider: BedrockProvider):
        """Test Cohere V3 with base truncate=True setting (maps to END)."""

        model = BedrockEmbeddingModel('cohere.embed-multilingual-v3', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(truncate=True))
        result = await embedder.embed_query('Test base truncate setting')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=1),
                inputs=['Test base truncate setting'],
                input_type='query',
                model_name='cohere.embed-multilingual-v3',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=6),
                provider_response_id=IsStr(),
            )
        )

    async def test_cohere_v4_minimal(self, bedrock_provider: BedrockProvider):
        """Test Cohere V4 with default settings (1536 dimensions, truncate=NONE)."""
        model = BedrockEmbeddingModel('cohere.embed-v4:0', provider=bedrock_provider)
        embedder = Embedder(model)
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1536), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                model_name='cohere.embed-v4:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=4),
                provider_response_id=IsStr(),
            )
        )

    async def test_cohere_v4_with_dimensions(self, bedrock_provider: BedrockProvider):
        """Test Cohere V4 with custom dimensions setting."""

        model = BedrockEmbeddingModel('cohere.embed-v4:0', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(dimensions=512))
        result = await embedder.embed_query('Test dimensions setting')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=512), length=1),
                inputs=['Test dimensions setting'],
                input_type='query',
                model_name='cohere.embed-v4:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=3),
                provider_response_id=IsStr(),
            )
        )

    async def test_cohere_v4_with_max_tokens(self, bedrock_provider: BedrockProvider):
        """Test Cohere V4 with max_tokens setting."""

        model = BedrockEmbeddingModel('cohere.embed-v4:0', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(bedrock_cohere_max_tokens=256))
        result = await embedder.embed_query('Test max tokens setting')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1536), length=1),
                inputs=['Test max tokens setting'],
                input_type='query',
                model_name='cohere.embed-v4:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=4),
                provider_response_id=IsStr(),
            )
        )

    async def test_cohere_v4_with_input_type(self, bedrock_provider: BedrockProvider):
        """Test Cohere V4 with custom input_type setting."""

        model = BedrockEmbeddingModel('cohere.embed-v4:0', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(bedrock_cohere_input_type='clustering'))
        result = await embedder.embed_query('Test input type setting')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1536), length=1),
                inputs=['Test input type setting'],
                input_type='query',
                model_name='cohere.embed-v4:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=4),
                provider_response_id=IsStr(),
            )
        )

    async def test_cohere_v4_with_truncate(self, bedrock_provider: BedrockProvider):
        """Test Cohere V4 with custom truncate setting."""

        model = BedrockEmbeddingModel('cohere.embed-v4:0', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(bedrock_cohere_truncate='END'))
        result = await embedder.embed_query('Test truncation setting')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1536), length=1),
                inputs=['Test truncation setting'],
                input_type='query',
                model_name='cohere.embed-v4:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=4),
                provider_response_id=IsStr(),
            )
        )

    async def test_cohere_v4_with_truncate_start(self, bedrock_provider: BedrockProvider):
        """Test Cohere V4 with truncate START setting."""

        model = BedrockEmbeddingModel('cohere.embed-v4:0', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(bedrock_cohere_truncate='START'))
        result = await embedder.embed_query('Test truncation start setting')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1536), length=1),
                inputs=['Test truncation start setting'],
                input_type='query',
                model_name='cohere.embed-v4:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=5),
                provider_response_id=IsStr(),
            )
        )

    async def test_cohere_v4_truncate_priority(self, bedrock_provider: BedrockProvider):
        """Test that bedrock_cohere_truncate takes precedence over base truncate."""

        model = BedrockEmbeddingModel('cohere.embed-v4:0', provider=bedrock_provider)
        # Both settings provided - model-specific should win (START over END from truncate=True)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(bedrock_cohere_truncate='START', truncate=True))
        result = await embedder.embed_query('Test truncate priority')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1536), length=1),
                inputs=['Test truncate priority'],
                input_type='query',
                model_name='cohere.embed-v4:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=3),
                provider_response_id=IsStr(),
            )
        )

    async def test_cohere_v4_batch_documents(self, bedrock_provider: BedrockProvider):
        """Test Cohere V4 batch embedding (multiple texts in single request)."""
        model = BedrockEmbeddingModel('cohere.embed-v4:0', provider=bedrock_provider)
        embedder = Embedder(model)
        result = await embedder.embed_documents(['hello', 'world'])
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1536), length=2),
                inputs=['hello', 'world'],
                input_type='document',
                model_name='cohere.embed-v4:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=2),
                provider_response_id=IsStr(),
            )
        )

    async def test_nova_minimal(self, bedrock_provider: BedrockProvider):
        """Test Nova with default settings (3072 dimensions, truncate=NONE)."""
        model = BedrockEmbeddingModel('amazon.nova-2-multimodal-embeddings-v1:0', provider=bedrock_provider)
        embedder = Embedder(model)
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=3072), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                model_name='amazon.nova-2-multimodal-embeddings-v1:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=19),
            )
        )

    async def test_nova_with_dimensions(self, bedrock_provider: BedrockProvider):
        """Test Nova with custom dimensions setting."""

        model = BedrockEmbeddingModel('amazon.nova-2-multimodal-embeddings-v1:0', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(dimensions=256))
        result = await embedder.embed_query('Test Nova dimensions')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=256), length=1),
                inputs=['Test Nova dimensions'],
                input_type='query',
                model_name='amazon.nova-2-multimodal-embeddings-v1:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=18),
            )
        )

    async def test_nova_with_truncate(self, bedrock_provider: BedrockProvider):
        """Test Nova with custom truncate setting."""

        model = BedrockEmbeddingModel('amazon.nova-2-multimodal-embeddings-v1:0', provider=bedrock_provider)
        embedder = Embedder(
            model,
            settings=BedrockEmbeddingSettings(bedrock_nova_truncate='END'),
        )
        result = await embedder.embed_query('Test Nova truncate')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=3072), length=1),
                inputs=['Test Nova truncate'],
                input_type='query',
                model_name='amazon.nova-2-multimodal-embeddings-v1:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=19),
            )
        )

    async def test_nova_with_truncate_start(self, bedrock_provider: BedrockProvider):
        """Test Nova with truncate START setting."""

        model = BedrockEmbeddingModel('amazon.nova-2-multimodal-embeddings-v1:0', provider=bedrock_provider)
        embedder = Embedder(
            model,
            settings=BedrockEmbeddingSettings(bedrock_nova_truncate='START'),
        )
        result = await embedder.embed_query('Test Nova truncate start')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=3072), length=1),
                inputs=['Test Nova truncate start'],
                input_type='query',
                model_name='amazon.nova-2-multimodal-embeddings-v1:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=20),
            )
        )

    async def test_nova_with_base_truncate(self, bedrock_provider: BedrockProvider):
        """Test Nova with base truncate=True setting (maps to END)."""

        model = BedrockEmbeddingModel('amazon.nova-2-multimodal-embeddings-v1:0', provider=bedrock_provider)
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(truncate=True))
        result = await embedder.embed_query('Test base truncate')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=3072), length=1),
                inputs=['Test base truncate'],
                input_type='query',
                model_name='amazon.nova-2-multimodal-embeddings-v1:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=19),
            )
        )

    async def test_nova_with_embedding_purpose(self, bedrock_provider: BedrockProvider):
        """Test Nova with custom embedding_purpose setting."""

        model = BedrockEmbeddingModel('amazon.nova-2-multimodal-embeddings-v1:0', provider=bedrock_provider)
        embedder = Embedder(
            model,
            settings=BedrockEmbeddingSettings(bedrock_nova_embedding_purpose='TEXT_RETRIEVAL'),
        )
        result = await embedder.embed_query('Test Nova settings')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=3072), length=1),
                inputs=['Test Nova settings'],
                input_type='query',
                model_name='amazon.nova-2-multimodal-embeddings-v1:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=22),
            )
        )

    async def test_nova_multiple_texts(self, bedrock_provider: BedrockProvider):
        """Test Nova document embedding (multiple texts, sequential requests)."""
        model = BedrockEmbeddingModel('amazon.nova-2-multimodal-embeddings-v1:0', provider=bedrock_provider)
        # Use max_concurrency=1 to ensure deterministic request order for VCRpy
        embedder = Embedder(model, settings=BedrockEmbeddingSettings(bedrock_max_concurrency=1))
        result = await embedder.embed_documents(['hello', 'world'])
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=3072), length=2),
                inputs=['hello', 'world'],
                input_type='document',
                model_name='amazon.nova-2-multimodal-embeddings-v1:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=6),
            )
        )

    async def test_titan_v1_max_input_tokens(self, bedrock_provider: BedrockProvider):
        model = BedrockEmbeddingModel('amazon.titan-embed-text-v1', provider=bedrock_provider)
        embedder = Embedder(model)
        max_input_tokens = await embedder.max_input_tokens()
        assert max_input_tokens == snapshot(8192)

    async def test_titan_v2_max_input_tokens(self, bedrock_provider: BedrockProvider):
        model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=bedrock_provider)
        embedder = Embedder(model)
        max_input_tokens = await embedder.max_input_tokens()
        assert max_input_tokens == snapshot(8192)

    async def test_cohere_v3_max_input_tokens(self, bedrock_provider: BedrockProvider):
        model = BedrockEmbeddingModel('cohere.embed-english-v3', provider=bedrock_provider)
        embedder = Embedder(model)
        max_input_tokens = await embedder.max_input_tokens()
        assert max_input_tokens == snapshot(512)

    async def test_cohere_v4_max_input_tokens(self, bedrock_provider: BedrockProvider):
        model = BedrockEmbeddingModel('cohere.embed-v4:0', provider=bedrock_provider)
        embedder = Embedder(model)
        max_input_tokens = await embedder.max_input_tokens()
        assert max_input_tokens == snapshot(128000)

    async def test_nova_max_input_tokens(self, bedrock_provider: BedrockProvider):
        model = BedrockEmbeddingModel('amazon.nova-2-multimodal-embeddings-v1:0', provider=bedrock_provider)
        embedder = Embedder(model)
        max_input_tokens = await embedder.max_input_tokens()
        assert max_input_tokens == snapshot(8192)

    async def test_base_url_property(self, bedrock_provider: BedrockProvider):
        """Test that base_url property returns the endpoint URL."""
        model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=bedrock_provider)
        assert model.base_url is not None
        assert isinstance(model.base_url, str)

    async def test_regional_prefix_model_name(self, bedrock_provider: BedrockProvider):
        """Test model with regional prefix (e.g., us.amazon.titan-embed-text-v2:0) is handled correctly."""
        model = BedrockEmbeddingModel('us.amazon.titan-embed-text-v2:0', provider=bedrock_provider)
        # Model name preserves the regional prefix
        assert model.model_name == 'us.amazon.titan-embed-text-v2:0'
        # But handler uses normalized name (without prefix)
        assert model._handler.model_name == 'amazon.titan-embed-text-v2:0'  # pyright: ignore[reportPrivateUsage]
        # max_input_tokens() works correctly with regional prefix
        max_tokens = await model.max_input_tokens()
        assert max_tokens == snapshot(8192)

    async def test_regional_prefix_embed(self, bedrock_provider: BedrockProvider):
        """Test embedding with a regional prefix model ID using Cohere v4.

        Cross-region inference profiles are supported for Cohere models on Bedrock.
        """
        model = BedrockEmbeddingModel('us.cohere.embed-v4:0', provider=bedrock_provider)
        embedder = Embedder(model)
        result = await embedder.embed_query('Hello from regional endpoint!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1536), length=1),
                inputs=['Hello from regional endpoint!'],
                input_type='query',
                model_name='us.cohere.embed-v4:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=5),
                provider_response_id=IsStr(),
            )
        )

    async def test_inference_profile_embed(self, bedrock_provider: BedrockProvider):
        # When re-recording, set AWS_ACCOUNT_ID to your real account ID
        account_id = os.getenv('AWS_ACCOUNT_ID', '123456789012')
        inference_profile_arn = f'arn:aws:bedrock:us-east-1:{account_id}:application-inference-profile/otnfa2ysixqd'
        settings: BedrockEmbeddingSettings = {'bedrock_inference_profile': inference_profile_arn}
        model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=bedrock_provider, settings=settings)

        result = await model.embed('Hello, world!', input_type='document')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=1024), length=1),
                inputs=['Hello, world!'],
                input_type='document',
                model_name='amazon.titan-embed-text-v2:0',
                provider_name='bedrock',
                timestamp=IsDatetime(),
                usage=RequestUsage(input_tokens=5),
            )
        )

    async def test_unsupported_model_error(self, bedrock_provider: BedrockProvider):
        with pytest.raises(UserError, match='Unsupported Bedrock embedding model'):
            BedrockEmbeddingModel('unsupported.model', provider=bedrock_provider)

    async def test_unknown_model_max_tokens_returns_none(self, bedrock_provider: BedrockProvider):
        """Test that unknown models with valid prefixes return None for max_input_tokens."""
        model = BedrockEmbeddingModel('amazon.titan-embed-text-v99:0', provider=bedrock_provider)
        assert await model.max_input_tokens() is None

    def test_model_with_string_provider(self, bedrock_provider: BedrockProvider):
        """Test BedrockEmbeddingModel can be created with string provider."""
        with patch('pydantic_ai.embeddings.bedrock.infer_provider', return_value=bedrock_provider) as mock_infer:
            model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider='bedrock')
            mock_infer.assert_called_once_with('bedrock')
            assert model.model_name == 'amazon.titan-embed-text-v2:0'

    async def test_client_error_with_status_code(self, bedrock_provider: BedrockProvider):
        """Test error handling when ClientError is raised with HTTP status code.

        ResponseMetadata.HTTPHeaders is the nested dict that BedrockEmbeddingModel extracts
        via metadata.get('HTTPHeaders') — verify it reaches ModelHTTPError.headers unchanged.
        """
        model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=bedrock_provider)

        error_response = {
            'Error': {'Code': 'ValidationException', 'Message': 'Invalid input'},
            'ResponseMetadata': {
                'HTTPStatusCode': 400,
                'HTTPHeaders': {'retry-after': '5', 'x-amzn-requestid': 'req-abc'},
            },
        }
        with patch.object(
            model.client,
            'invoke_model',
            side_effect=ClientError(error_response, 'InvokeModel'),  # pyright: ignore[reportArgumentType]
        ):
            with pytest.raises(ExceptionGroup) as exc_info:
                await model.embed(['test'], input_type='query')
            assert len(exc_info.value.exceptions) == 1
            exc = exc_info.value.exceptions[0]
            assert isinstance(exc, ModelHTTPError)
            assert exc.status_code == 400
            assert exc.headers is not None
            assert exc.headers.get('retry-after') == '5'
            assert exc.headers.get('x-amzn-requestid') == 'req-abc'

    async def test_client_error_without_status_code(self, bedrock_provider: BedrockProvider):
        """Test error handling when ClientError is raised without HTTP status code."""
        model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=bedrock_provider)

        error_response = {
            'Error': {'Code': 'UnknownError', 'Message': 'Something went wrong'},
            'ResponseMetadata': {},  # No HTTPStatusCode
        }
        with patch.object(
            model.client,
            'invoke_model',
            side_effect=ClientError(error_response, 'InvokeModel'),  # pyright: ignore[reportArgumentType]
        ):
            with pytest.raises(ExceptionGroup) as exc_info:
                await model.embed(['test'], input_type='query')
            assert len(exc_info.value.exceptions) == 1
            assert isinstance(exc_info.value.exceptions[0], ModelAPIError)

    async def test_count_tokens_not_implemented(self, bedrock_provider: BedrockProvider):
        """Test that count_tokens raises NotImplementedError (Bedrock doesn't support it)."""
        model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=bedrock_provider)
        embedder = Embedder(model)
        with pytest.raises(NotImplementedError):
            await embedder.count_tokens('Hello, world!')

    @pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
    async def test_instrumentation(self, bedrock_provider: BedrockProvider, capfire: CaptureLogfire):
        model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=bedrock_provider)
        embedder = Embedder(model, instrument=True)
        await embedder.embed_query('Hello, world!', settings={'dimensions': 256})

        spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
        span = next(span for span in spans if 'embeddings' in span['name'])

        assert span == snapshot(
            {
                'name': 'embeddings amazon.titan-embed-text-v2:0',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': IsInt(),
                'end_time': IsInt(),
                'attributes': {
                    'gen_ai.operation.name': 'embeddings',
                    'gen_ai.provider.name': 'bedrock',
                    'gen_ai.request.model': 'amazon.titan-embed-text-v2:0',
                    'input_type': 'query',
                    'server.address': 'bedrock-runtime.us-east-1.amazonaws.com',
                    'inputs_count': 1,
                    'embedding_settings': {'dimensions': 256},
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
                    'logfire.msg': 'embeddings amazon.titan-embed-text-v2:0',
                    'gen_ai.usage.input_tokens': 5,
                    'gen_ai.response.model': 'amazon.titan-embed-text-v2:0',
                    'operation.cost': 5e-07,
                    'gen_ai.embeddings.dimension.count': 256,
                    'embeddings': [
                        [
                            -0.09060307592153549,
                            0.19755451381206512,
                            0.029295168817043304,
                            -0.0407390259206295,
                            0.03198820352554321,
                            -0.07254625856876373,
                            0.07156474143266678,
                            0.021547332406044006,
                            0.04041390120983124,
                            0.020400185137987137,
                            0.029309740290045738,
                            0.023030338808894157,
                            -0.00732302526012063,
                            -0.09935697168111801,
                            -0.028770674020051956,
                            -0.01739199459552765,
                            0.017438577488064766,
                            -0.09611490368843079,
                            0.07079621404409409,
                            -0.059979796409606934,
                            -0.019924761727452278,
                            0.019159488379955292,
                            -0.0021209935657680035,
                            0.011791412718594074,
                            -0.11812614649534225,
                            -0.003097908105701208,
                            0.005962706170976162,
                            -0.013459030538797379,
                            0.06573699414730072,
                            -0.0030917737167328596,
                            0.18248212337493896,
                            0.08134309202432632,
                            0.002889336086809635,
                            -0.02139703556895256,
                            0.05786033719778061,
                            0.15323910117149353,
                            0.08070510625839233,
                            -0.006874058861285448,
                            0.07243584841489792,
                            -0.04509449750185013,
                            0.05067840591073036,
                            -0.01804717257618904,
                            -0.004620024003088474,
                            0.02507771924138069,
                            0.01773475669324398,
                            0.07827892154455185,
                            0.024611501023173332,
                            0.05785113573074341,
                            0.08990681916475296,
                            0.007683425676077604,
                            0.05162157863378525,
                            -0.008477839641273022,
                            -0.07484668493270874,
                            0.07796300202608109,
                            -0.019157953560352325,
                            0.01868560165166855,
                            0.03604615479707718,
                            0.03901984170079231,
                            -0.06384757906198502,
                            -0.013228987343609333,
                            -0.04396767541766167,
                            0.026188058778643608,
                            0.0625225305557251,
                            0.006156709045171738,
                            -0.03721477463841438,
                            -0.14584091305732727,
                            -0.06491497159004211,
                            -0.051186032593250275,
                            -0.06784112006425858,
                            0.08590101450681686,
                            0.03569649159908295,
                            -0.00468673650175333,
                            -0.05168292298913002,
                            -0.05307851731777191,
                            -0.012795739807188511,
                            -0.0366227962076664,
                            0.038486141711473465,
                            -0.010759863071143627,
                            -0.04883039370179176,
                            0.17056283354759216,
                            -0.033978838473558426,
                            -0.03620564937591553,
                            0.027411887422204018,
                            -0.06762641668319702,
                            -0.024319345131516457,
                            -0.08622612804174423,
                            0.06366661190986633,
                            -0.03024294599890709,
                            -0.0016179669182747602,
                            -0.04859728366136551,
                            -0.033322449773550034,
                            0.06800215691328049,
                            0.00525570847094059,
                            0.04471416398882866,
                            0.1069345772266388,
                            -0.11152469366788864,
                            0.07652139663696289,
                            -0.1129494234919548,
                            -0.08297716081142426,
                            -0.028996111825108528,
                            -0.054041627794504166,
                            0.014081680215895176,
                            0.06624002754688263,
                            0.025899739935994148,
                            0.07912241667509079,
                            0.014882228337228298,
                            -0.044965676963329315,
                            0.07860098779201508,
                            -0.02821703441441059,
                            0.0021194599103182554,
                            0.046871963888406754,
                            0.049051232635974884,
                            0.03404612839221954,
                            -0.024462737143039703,
                            0.04415592551231384,
                            -0.11784128099679947,
                            0.10309746116399765,
                            0.050725944340229034,
                            0.13046947121620178,
                            0.07651526480913162,
                            -0.041904572397470474,
                            -0.003067235928028822,
                            -0.014202835969626904,
                            0.010428601875901222,
                            0.057942382991313934,
                            -0.021722164005041122,
                            -0.05459219589829445,
                            0.010652509517967701,
                            -0.021011333912611008,
                            0.03867477551102638,
                            0.09031321853399277,
                            -0.028267646208405495,
                            0.05359687656164169,
                            -0.014423675835132599,
                            0.04842245206236839,
                            -0.013590922579169273,
                            -0.07219238579273224,
                            0.01238549780100584,
                            -0.052477337419986725,
                            -0.04114390164613724,
                            -0.02811121568083763,
                            0.000969246553722769,
                            0.07565030455589294,
                            -0.15310414135456085,
                            0.08023888617753983,
                            -0.017510849982500076,
                            -0.031463705003261566,
                            -0.02159947343170643,
                            -0.04565043747425079,
                            -0.00662982976064086,
                            -0.15926314890384674,
                            0.027329647913575172,
                            -0.010661711916327477,
                            0.031040426343679428,
                            -0.009848893620073795,
                            0.07134083658456802,
                            0.03514745831489563,
                            0.08713556826114655,
                            -0.0377546064555645,
                            0.0766686275601387,
                            -0.0481356643140316,
                            0.0018832827918231487,
                            0.12125396728515625,
                            -0.035317301750183105,
                            0.07488963007926941,
                            -0.07252171635627747,
                            -0.02362385019659996,
                            -0.02880747802555561,
                            0.06559687107801437,
                            0.03381320834159851,
                            -0.04347116872668266,
                            0.0006456531118601561,
                            -0.035015564411878586,
                            0.09682036936283112,
                            -0.01940946839749813,
                            0.128380686044693,
                            0.019004592671990395,
                            -0.06542490422725677,
                            -0.1337069571018219,
                            -0.02495809830725193,
                            0.019561486318707466,
                            -0.020059721544384956,
                            -0.08843453973531723,
                            -0.08971051126718521,
                            0.02468511275947094,
                            -0.13424371182918549,
                            0.045431893318891525,
                            -0.06309917569160461,
                            -0.0286878552287817,
                            -0.03678842633962631,
                            -0.049784302711486816,
                            -0.022281935438513756,
                            0.03574863448739052,
                            -0.08971664309501648,
                            0.012232135981321335,
                            0.042407602071762085,
                            -0.11800269782543182,
                            -0.024039460346102715,
                            0.01734521985054016,
                            0.09017059952020645,
                            -0.028598906472325325,
                            -0.053897466510534286,
                            0.05603226646780968,
                            0.05059712007641792,
                            -0.04617110267281532,
                            0.03818095102906227,
                            -0.010658644139766693,
                            -0.007039306219667196,
                            0.07277937233448029,
                            -0.017655007541179657,
                            -0.012112514115869999,
                            -0.03988020122051239,
                            0.01731147989630699,
                            -0.030819585546851158,
                            0.058645546436309814,
                            -0.011903942562639713,
                            0.07016915082931519,
                            -0.13065505027770996,
                            -0.09879566729068756,
                            0.03908271715044975,
                            -0.026194192469120026,
                            0.10055319219827652,
                            -0.050931449979543686,
                            -0.0004662198480218649,
                            0.01080893911421299,
                            -0.024402352049946785,
                            0.022439897060394287,
                            -0.06716325879096985,
                            0.0021900064311921597,
                            0.03293290734291077,
                            -0.0325479730963707,
                            0.014446680434048176,
                            -0.037684060633182526,
                            0.00859132781624794,
                            0.017724020406603813,
                            -0.02108187973499298,
                            -0.04605761170387268,
                            0.04302104935050011,
                            0.004255789797753096,
                            0.007060776464641094,
                            -0.004876905120909214,
                            -0.10156230628490448,
                            -0.015161346644163132,
                            0.11803336441516876,
                            -0.026813777163624763,
                            0.051384635269641876,
                            -0.03564434498548508,
                            0.045358285307884216,
                            -0.09531895071268082,
                            -0.27399003505706787,
                            0.07576993107795715,
                            -0.005306317936629057,
                            0.02496729977428913,
                            0.07582207024097443,
                            0.03562901169061661,
                            0.009741540998220444,
                        ]
                    ],
                },
            }
        )


@dataclass
class _GoogleTaskPrefixCase:
    id: str
    model_name: str
    input_type: Literal['query', 'document']
    inputs: list[str]
    settings: GoogleEmbeddingSettings
    expected_texts: list[str]
    expected_task_type: str | None
    expected_warning: str | None = None


# The `GoogleEmbeddingSettings(...)` calls are only evaluated when the `google` extra is installed.
_GOOGLE_TASK_PREFIX_CASES: list[_GoogleTaskPrefixCase] = (
    [
        _GoogleTaskPrefixCase(
            id='default-query',
            model_name='gemini-embedding-2',
            input_type='query',
            inputs=['Hello, world!'],
            settings=GoogleEmbeddingSettings(),
            expected_texts=['task: search result | query: Hello, world!'],
            expected_task_type=None,
        ),
        _GoogleTaskPrefixCase(
            id='asymmetric-query',
            model_name='gemini-embedding-2',
            input_type='query',
            inputs=['Hello, world!'],
            settings=GoogleEmbeddingSettings(google_task='question answering'),
            expected_texts=['task: question answering | query: Hello, world!'],
            expected_task_type=None,
        ),
        _GoogleTaskPrefixCase(
            id='asymmetric-document-with-title',
            model_name='gemini-embedding-2',
            input_type='document',
            inputs=['hello'],
            settings=GoogleEmbeddingSettings(google_task='search result', google_title='Greeting'),
            expected_texts=['title: Greeting | text: hello'],
            expected_task_type=None,
        ),
        _GoogleTaskPrefixCase(
            id='asymmetric-document-no-title',
            model_name='gemini-embedding-2',
            input_type='document',
            inputs=['hello', 'world'],
            settings=GoogleEmbeddingSettings(),
            expected_texts=['title: none | text: hello', 'title: none | text: world'],
            expected_task_type=None,
        ),
        _GoogleTaskPrefixCase(
            id='symmetric-query',
            model_name='gemini-embedding-2',
            input_type='query',
            inputs=['hello'],
            settings=GoogleEmbeddingSettings(google_task='classification'),
            expected_texts=['task: classification | query: hello'],
            expected_task_type=None,
        ),
        _GoogleTaskPrefixCase(
            id='symmetric-document-ignores-title',
            model_name='gemini-embedding-2',
            input_type='document',
            inputs=['hello'],
            settings=GoogleEmbeddingSettings(google_task='clustering', google_title='ignored'),
            expected_texts=['task: clustering | query: hello'],
            expected_task_type=None,
        ),
        _GoogleTaskPrefixCase(
            id='symmetric-sentence-similarity-ignores-title',
            model_name='gemini-embedding-2',
            input_type='document',
            inputs=['hello'],
            settings=GoogleEmbeddingSettings(google_task='sentence similarity', google_title='ignored'),
            expected_texts=['task: sentence similarity | query: hello'],
            expected_task_type=None,
        ),
        _GoogleTaskPrefixCase(
            id='asymmetric-document-empty-title',
            model_name='gemini-embedding-2',
            input_type='document',
            inputs=['hello'],
            settings=GoogleEmbeddingSettings(google_title=''),
            expected_texts=['title: none | text: hello'],
            expected_task_type=None,
        ),
        _GoogleTaskPrefixCase(
            id='raw-passthrough',
            model_name='gemini-embedding-2',
            input_type='document',
            inputs=['title: custom | text: hello'],
            settings=GoogleEmbeddingSettings(google_task='raw'),
            expected_texts=['title: custom | text: hello'],
            expected_task_type=None,
        ),
        _GoogleTaskPrefixCase(
            id='task-type-ignored-on-embedding-2',
            model_name='gemini-embedding-2',
            input_type='query',
            inputs=['hello'],
            settings=GoogleEmbeddingSettings(google_task='classification', google_task_type='RETRIEVAL_QUERY'),
            expected_texts=['task: classification | query: hello'],
            expected_task_type=None,
            expected_warning='`google_task_type` is not supported by `gemini-embedding-2`',
        ),
        _GoogleTaskPrefixCase(
            id='task-ignored-on-other-model',
            model_name='gemini-embedding-2-preview',
            input_type='query',
            inputs=['hello'],
            settings=GoogleEmbeddingSettings(google_task='classification'),
            expected_texts=['hello'],
            expected_task_type='RETRIEVAL_QUERY',
            expected_warning='`google_task` is only supported by `gemini-embedding-2`',
        ),
    ]
    if google_imports_successful()
    else []
)


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
@pytest.mark.vcr
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in _GOOGLE_TASK_PREFIX_CASES])
async def test_google_task_prefix(case: _GoogleTaskPrefixCase, gemini_api_key: str, monkeypatch: pytest.MonkeyPatch):
    """`google_task` builds the right text prefix (and `task_type`) for `gemini-embedding-2`.

    Spies on `embed_content` to assert the exact text sent to the API for each
    (task, input_type, title) combination, plus the warn-and-ignore behavior when
    `google_task`/`google_task_type` are used on the wrong model.
    """
    provider = GoogleProvider(api_key=gemini_api_key)
    model = GoogleEmbeddingModel(case.model_name, provider=provider)
    embedder = Embedder(model)

    captured: dict[str, Any] = {}
    real_embed_content = provider.client.aio.models.embed_content

    async def spy(**kwargs: Any) -> Any:
        captured['contents'] = kwargs['contents']
        captured['config'] = kwargs['config']
        return await real_embed_content(**kwargs)

    monkeypatch.setattr(provider.client.aio.models, 'embed_content', spy)

    async def run() -> EmbeddingResult:
        if case.input_type == 'query':
            return await embedder.embed_query(case.inputs, settings=case.settings)
        return await embedder.embed_documents(case.inputs, settings=case.settings)

    if case.expected_warning is not None:
        with pytest.warns(UserWarning, match=case.expected_warning):
            result = await run()
    else:
        result = await run()

    sent_texts = [part.text for content in captured['contents'] for part in content.parts]
    assert sent_texts == case.expected_texts
    assert captured['config'].task_type == case.expected_task_type
    assert captured['config'].title is None
    assert len(result.embeddings) == len(case.inputs)
    # The prefix is applied internally; the user gets their original (non-prefixed) text back.
    assert result.inputs == case.inputs


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
@pytest.mark.skipif(
    not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
)
@pytest.mark.vcr
async def test_google_task_prefix_vertex(
    allow_model_requests: None, vertex_provider: GoogleCloudProvider, monkeypatch: pytest.MonkeyPatch
):  # pragma: lax no cover
    """`google_task` builds the same `gemini-embedding-2` prefix against Google Cloud (Vertex) as against the Gemini API."""
    model = GoogleEmbeddingModel('gemini-embedding-2', provider=vertex_provider)
    embedder = Embedder(model)

    captured: dict[str, Any] = {}
    real_embed_content = vertex_provider.client.aio.models.embed_content

    async def spy(**kwargs: Any) -> Any:
        captured['contents'] = kwargs['contents']
        captured['config'] = kwargs['config']
        return await real_embed_content(**kwargs)

    monkeypatch.setattr(vertex_provider.client.aio.models, 'embed_content', spy)

    result = await embedder.embed_query(
        'Hello, world!', settings=GoogleEmbeddingSettings(google_task='question answering')
    )

    sent_texts = [part.text for content in captured['contents'] for part in content.parts]
    assert sent_texts == ['task: question answering | query: Hello, world!']
    assert captured['config'].task_type is None
    assert captured['config'].title is None
    assert len(result.embeddings) == 1


@pytest.mark.skipif(not google_imports_successful(), reason='Google not installed')
@pytest.mark.vcr
class TestGoogle:
    @pytest.fixture
    def embedder(self, gemini_api_key: str) -> Embedder:
        return Embedder(
            GoogleEmbeddingModel('gemini-embedding-2-preview', provider=GoogleProvider(api_key=gemini_api_key))
        )

    async def test_infer_model_google(self, gemini_api_key: str):
        with patch.dict(os.environ, {'GOOGLE_API_KEY': gemini_api_key}):
            model = infer_embedding_model('google:gemini-embedding-001')
        assert isinstance(model, GoogleEmbeddingModel)
        assert model.model_name == 'gemini-embedding-001'
        assert model.system == 'google'
        assert urlparse(model.base_url).hostname == 'generativelanguage.googleapis.com'

    async def test_infer_model_google_cloud(self, env: TestEnv):
        for name in {
            'GOOGLE_APPLICATION_CREDENTIALS',
            'GOOGLE_CLOUD_PROJECT',
            'GOOGLE_CLOUD_LOCATION',
            'GEMINI_API_KEY',
        }:
            env.remove(name)
        env.set('GOOGLE_API_KEY', 'mock-api-key')
        model = infer_embedding_model('google-cloud:gemini-embedding-001')
        assert isinstance(model, GoogleEmbeddingModel)
        assert model.model_name == 'gemini-embedding-001'
        assert model.system == 'google-cloud'

    async def test_model_with_string_provider(self, gemini_api_key: str):
        with patch.dict(os.environ, {'GOOGLE_API_KEY': gemini_api_key}):
            model = GoogleEmbeddingModel('gemini-embedding-001', provider='google')
        assert isinstance(model, GoogleEmbeddingModel)
        assert model.model_name == 'gemini-embedding-001'
        assert model.system == 'google'

    async def test_query(self, embedder: Embedder):
        result = await embedder.embed_query('Hello, world!')
        assert result == snapshot(
            EmbeddingResult(
                embeddings=IsList(IsList(IsFloat(), length=3072), length=1),
                inputs=['Hello, world!'],
                input_type='query',
                usage=RequestUsage(),
                model_name='gemini-embedding-2-preview',
                timestamp=IsDatetime(),
                provider_name='google',
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
                model_name='gemini-embedding-2-preview',
                timestamp=IsDatetime(),
                provider_name='google',
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
                model_name='gemini-embedding-2-preview',
                timestamp=IsDatetime(),
                provider_name='google',
            )
        )

    async def test_max_input_tokens(self, embedder: Embedder):
        max_input_tokens = await embedder.max_input_tokens()
        assert max_input_tokens == snapshot(8192)

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

    async def test_embed_error_no_http_response(self, gemini_api_key: str, mocker: MockerFixture):
        """An APIError with response=None (no HTTP response object) yields headers=None on ModelHTTPError.

        This exercises the defensive `e.response is not None else None` branch in the embed
        path of GoogleEmbeddingModel — the branch that handles non-HTTP errors where the SDK
        raises an APIError without attaching an httpx.Response.
        """
        from google.genai import errors

        model = GoogleEmbeddingModel('gemini-embedding-2-preview', provider=GoogleProvider(api_key=gemini_api_key))
        error_without_response = errors.APIError(503, {'error': {'code': 503, 'message': 'Unavailable'}})
        mocker.patch.object(model._client.aio.models, 'embed_content', side_effect=error_without_response)  # pyright: ignore[reportPrivateUsage]

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.embed(['test'], input_type='query')

        assert exc_info.value.status_code == 503
        assert exc_info.value.headers is None

    async def test_count_tokens_error_no_http_response(self, gemini_api_key: str, mocker: MockerFixture):
        """Same as above for the count_tokens path."""
        from google.genai import errors

        model = GoogleEmbeddingModel('gemini-embedding-2-preview', provider=GoogleProvider(api_key=gemini_api_key))
        error_without_response = errors.APIError(503, {'error': {'code': 503, 'message': 'Unavailable'}})
        mocker.patch.object(model._client.aio.models, 'count_tokens', side_effect=error_without_response)  # pyright: ignore[reportPrivateUsage]

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.count_tokens('test')

        assert exc_info.value.status_code == 503
        assert exc_info.value.headers is None

    async def test_embed_error_with_http_response(self, gemini_api_key: str, mocker: MockerFixture):
        """An APIError with a real httpx.Response propagates its headers to ModelHTTPError.

        The positive path `headers=dict(e.response.headers) if e.response is not None else None`
        converts httpx.Headers to a plain lowercased dict. Verify the value reaches
        ModelHTTPError.headers so a wrong-attribute regression (e.g. swapping response for None)
        would be caught.
        """
        import httpx
        from google.genai import errors

        model = GoogleEmbeddingModel('gemini-embedding-2-preview', provider=GoogleProvider(api_key=gemini_api_key))
        req = httpx.Request('POST', 'https://generativelanguage.googleapis.com/v1beta/models')
        resp = httpx.Response(429, headers={'retry-after': '10', 'x-goog-request-id': 'rid-1'}, request=req)
        error_with_response = errors.APIError(429, {'error': {'code': 429, 'message': 'Rate limited'}})
        error_with_response.response = resp
        mocker.patch.object(model._client.aio.models, 'embed_content', side_effect=error_with_response)  # pyright: ignore[reportPrivateUsage]

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.embed(['test'], input_type='query')

        exc = exc_info.value
        assert exc.status_code == 429
        assert exc.headers is not None
        assert exc.headers.get('retry-after') == '10'
        assert exc.headers.get('x-goog-request-id') == 'rid-1'

    async def test_embed_error_low_status_code(self, gemini_api_key: str, mocker: MockerFixture):
        """An APIError with code < 400 is re-raised verbatim, not wrapped in ModelHTTPError.

        GoogleEmbeddingModel only wraps errors with status_code >= 400. A code below
        400 is a non-HTTP-error signal from the SDK; the original exception propagates.
        This covers the `raise` (else) branch of `if (status_code := e.code) >= 400`.
        """
        from google.genai import errors

        model = GoogleEmbeddingModel('gemini-embedding-2-preview', provider=GoogleProvider(api_key=gemini_api_key))
        low_code_error = errors.APIError(0, {'error': {'code': 0, 'message': 'Unknown'}})
        mocker.patch.object(model._client.aio.models, 'embed_content', side_effect=low_code_error)  # pyright: ignore[reportPrivateUsage]

        with pytest.raises(errors.APIError) as exc_info:
            await model.embed(['test'], input_type='query')

        assert exc_info.value is low_code_error

    async def test_count_tokens_error_low_status_code(self, gemini_api_key: str, mocker: MockerFixture):
        """Same as test_embed_error_low_status_code for the count_tokens path."""
        from google.genai import errors

        model = GoogleEmbeddingModel('gemini-embedding-2-preview', provider=GoogleProvider(api_key=gemini_api_key))
        low_code_error = errors.APIError(0, {'error': {'code': 0, 'message': 'Unknown'}})
        mocker.patch.object(model._client.aio.models, 'count_tokens', side_effect=low_code_error)  # pyright: ignore[reportPrivateUsage]

        with pytest.raises(errors.APIError) as exc_info:
            await model.count_tokens('test')

        assert exc_info.value is low_code_error

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
                model_name='gemini-embedding-2-preview',
                timestamp=IsDatetime(),
                provider_name='google',
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
                provider_name='google-cloud',
            )
        )

    @pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
    async def test_instrumentation(self, gemini_api_key: str, capfire: CaptureLogfire):
        model = GoogleEmbeddingModel('gemini-embedding-2-preview', provider=GoogleProvider(api_key=gemini_api_key))
        embedder = Embedder(model, instrument=True)
        await embedder.embed_query('Hello, world!', settings={'dimensions': 768})

        spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
        span = next(span for span in spans if 'embeddings' in span['name'])

        assert span == snapshot(
            {
                'name': 'embeddings gemini-embedding-2-preview',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': IsInt(),
                'end_time': IsInt(),
                'attributes': {
                    'gen_ai.operation.name': 'embeddings',
                    'gen_ai.provider.name': 'google',
                    'gen_ai.request.model': 'gemini-embedding-2-preview',
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
                    'logfire.msg': 'embeddings gemini-embedding-2-preview',
                    'gen_ai.response.model': 'gemini-embedding-2-preview',
                    'gen_ai.embeddings.dimension.count': 768,
                    'embeddings': [
                        [
                            -0.039718185,
                            0.0070917504,
                            0.04710465,
                            -0.01439842,
                            -0.032200605,
                            0.024094693,
                            -0.010970279,
                            0.02872452,
                            -0.010583709,
                            -0.08625138,
                            -0.0063364585,
                            0.01887979,
                            -0.021125995,
                            0.00435056,
                            0.05707527,
                            0.018849764,
                            0.018166717,
                            -0.0286141,
                            -0.03673057,
                            -0.014347317,
                            -0.0039356677,
                            0.014675356,
                            -0.0075492137,
                            0.035733655,
                            -0.008793042,
                            0.04962065,
                            0.025404112,
                            -0.0069676614,
                            -0.047568157,
                            0.23192443,
                            0.010093859,
                            -0.017558968,
                            -0.015465944,
                            -0.041906275,
                            0.02873053,
                            -0.018706022,
                            0.0046569617,
                            -0.019730637,
                            -0.01292914,
                            -0.04229858,
                            0.026582966,
                            0.010542063,
                            0.020673444,
                            0.010464244,
                            0.015809909,
                            -0.012530256,
                            -0.046968244,
                            -0.02345551,
                            0.021388978,
                            -0.0010235927,
                            -0.02773519,
                            0.02532855,
                            -0.000462879,
                            -0.04641633,
                            -0.03061325,
                            -0.006679063,
                            -0.034734864,
                            -0.005253075,
                            -0.023992425,
                            0.021487711,
                            0.034217577,
                            -0.018886117,
                            0.025997177,
                            0.033568352,
                            -0.012795402,
                            0.009265727,
                            0.028410094,
                            -0.010798237,
                            0.01884387,
                            0.008468412,
                            0.030363921,
                            0.024800597,
                            0.0057984754,
                            -0.052960366,
                            -0.02335996,
                            0.0069483602,
                            0.009702113,
                            -0.03183237,
                            0.021668525,
                            -0.0044478346,
                            -0.005872878,
                            0.0033597185,
                            0.008562815,
                            -0.005600276,
                            -0.023837758,
                            -0.00831205,
                            -0.013365214,
                            0.03389827,
                            0.014578596,
                            -0.015586147,
                            0.0021254362,
                            -0.011313204,
                            -0.003489946,
                            0.018111814,
                            -0.010687574,
                            0.0097742295,
                            0.0031727394,
                            -0.008046762,
                            0.015921036,
                            -0.0053140237,
                            -0.027457967,
                            0.009619769,
                            -0.050177447,
                            -0.009322343,
                            0.027861806,
                            0.03497894,
                            0.017372763,
                            -0.017192954,
                            0.014178947,
                            -0.0063463063,
                            -0.00053655886,
                            -0.35211203,
                            -0.03261482,
                            -0.018916233,
                            0.019432215,
                            -0.06190374,
                            -0.009335508,
                            0.01914503,
                            0.00484078,
                            0.03086304,
                            0.023285542,
                            0.046894588,
                            0.01586852,
                            -0.032378677,
                            0.011936534,
                            0.028885048,
                            0.030906744,
                            -0.025834473,
                            -0.031206204,
                            -0.006359555,
                            0.00335148,
                            0.05258607,
                            0.017001793,
                            -0.021232156,
                            -0.015524483,
                            -0.014554567,
                            -0.023691524,
                            -0.04296526,
                            0.055506714,
                            -0.022089561,
                            -0.0013124781,
                            0.01848247,
                            -0.012746861,
                            -0.006956526,
                            0.01142064,
                            0.004620205,
                            -0.0021637836,
                            -0.0047564777,
                            -0.04220708,
                            -0.0008800746,
                            0.028327422,
                            -0.01506599,
                            -0.008554804,
                            0.036477096,
                            0.027212359,
                            -0.01927298,
                            -0.0131601095,
                            0.005962288,
                            0.025952693,
                            -0.01552238,
                            0.027871216,
                            -0.0045060944,
                            -0.037848838,
                            -0.005343479,
                            -0.008562321,
                            0.0789249,
                            0.008916387,
                            0.0021779865,
                            -0.025126694,
                            0.0068234983,
                            -0.0220768,
                            0.037357222,
                            -0.007670817,
                            -0.0024561433,
                            -0.008693516,
                            0.007712375,
                            -0.018511057,
                            -0.0024187707,
                            3.8923972e-05,
                            0.008710769,
                            1.5122982e-06,
                            0.019026315,
                            0.00038850043,
                            -0.034555256,
                            -0.033012778,
                            0.0036981506,
                            0.0039528096,
                            -0.027452607,
                            -0.0057473024,
                            0.00030588498,
                            0.041277274,
                            -0.004404056,
                            -0.009171699,
                            0.016077898,
                            0.018406542,
                            0.0122451335,
                            0.0065869265,
                            -0.03325715,
                            0.0031982455,
                            -0.038740937,
                            -0.008824045,
                            0.017574212,
                            -0.006540467,
                            -0.035467267,
                            -0.013357835,
                            -0.017471377,
                            -0.022087116,
                            -0.020092279,
                            -0.022217928,
                            0.015064528,
                            0.053085286,
                            0.005801194,
                            -0.009982443,
                            -0.019088073,
                            0.021847837,
                            0.017901996,
                            0.019352717,
                            0.079207845,
                            -0.04335041,
                            -0.0052789343,
                            0.05320919,
                            0.04627506,
                            0.014458461,
                            0.017407324,
                            -0.051489998,
                            0.0032934865,
                            0.014288595,
                            -0.010135598,
                            -0.006224999,
                            0.024853924,
                            -0.0030578393,
                            -0.018449089,
                            0.03429361,
                            -0.015636684,
                            0.00919899,
                            0.022262411,
                            -0.009200845,
                            0.05408544,
                            0.011237957,
                            0.06809289,
                            -0.024023388,
                            0.008849299,
                            0.009711797,
                            0.019844323,
                            -0.018000651,
                            -0.016385049,
                            0.012468136,
                            -0.0023727433,
                            0.014378499,
                            -0.028540535,
                            -0.025778761,
                            -0.0071643274,
                            0.0018286612,
                            -0.006208821,
                            0.028918855,
                            0.04557822,
                            0.042765845,
                            -0.0011696913,
                            -0.012811224,
                            0.0035377173,
                            0.014354801,
                            -0.035951484,
                            0.020528223,
                            -0.005355377,
                            0.031006813,
                            0.013665832,
                            -0.010941785,
                            0.008625167,
                            0.008132226,
                            0.06138893,
                            0.027802056,
                            0.0045530074,
                            0.01672978,
                            -0.008494048,
                            0.013599865,
                            0.022728283,
                            -0.019952137,
                            0.0072484966,
                            -0.020345319,
                            -0.038401946,
                            -0.013421613,
                            0.0037297935,
                            -0.014992606,
                            -0.026964154,
                            0.007908023,
                            0.020773768,
                            -0.022224572,
                            -0.0023568014,
                            0.015372207,
                            0.0071050115,
                            0.027023513,
                            -0.029069506,
                            0.01239473,
                            0.008238166,
                            0.015439753,
                            0.0035322723,
                            -0.01686047,
                            0.0035522743,
                            -0.023494868,
                            -0.01744252,
                            -0.038724273,
                            -0.007844626,
                            -0.0008073593,
                            -0.031811237,
                            0.007262426,
                            0.048724655,
                            0.018961968,
                            -0.015134065,
                            0.019078458,
                            0.0064050616,
                            -0.028675204,
                            -0.024842802,
                            -0.06528752,
                            0.0040062354,
                            -0.024066366,
                            -0.0024558394,
                            0.01304531,
                            -0.024845209,
                            0.0013522166,
                            -0.2151237,
                            -0.0032345597,
                            0.009205375,
                            -0.047540195,
                            0.010263713,
                            -0.009950665,
                            0.035051547,
                            -0.009835741,
                            -0.023357121,
                            -0.026387766,
                            0.020865431,
                            -0.010684087,
                            0.0008897444,
                            -0.0048271306,
                            0.0003864332,
                            -0.010216144,
                            -0.010752211,
                            0.01204882,
                            0.012218168,
                            -0.0032389981,
                            0.031441547,
                            -0.023674348,
                            -0.3684222,
                            -0.025399694,
                            0.0011839865,
                            0.031524897,
                            -0.03859212,
                            -0.0009058105,
                            0.009053854,
                            -0.0009867137,
                            0.06955363,
                            -0.01603583,
                            0.019266773,
                            0.024365652,
                            -0.010622137,
                            -0.012839141,
                            -0.0033559306,
                            -0.030351954,
                            -0.031793445,
                            0.014498899,
                            -0.01684399,
                            0.0152310245,
                            -0.005884764,
                            0.03919016,
                            0.019014737,
                            0.01896184,
                            0.0030873166,
                            -0.037844185,
                            0.042336885,
                            0.038030304,
                            -0.018450303,
                            -0.034548126,
                            0.03574147,
                            0.051009174,
                            0.029431224,
                            0.009230895,
                            0.0035172948,
                            -0.0043552704,
                            0.03118658,
                            0.020804023,
                            0.00086036127,
                            -0.027324405,
                            0.027556578,
                            0.013465706,
                            -0.010402242,
                            0.07202653,
                            0.03863592,
                            0.0049685836,
                            -0.0040400266,
                            0.032852028,
                            -0.014997189,
                            -0.0031374989,
                            0.06648203,
                            0.0025776841,
                            0.0041916245,
                            -0.015081957,
                            0.0061184163,
                            0.039823398,
                            -0.04882981,
                            -0.0021786094,
                            -0.0111819375,
                            -0.04007043,
                            -0.020118162,
                            0.02469064,
                            0.02424022,
                            -0.006638552,
                            0.036903173,
                            0.043675564,
                            -0.027565528,
                            -0.018154671,
                            0.014449511,
                            0.00060017296,
                            0.00028639263,
                            0.013484467,
                            -0.02450472,
                            0.030932492,
                            -0.014354374,
                            0.0036955953,
                            0.011471794,
                            -0.019299483,
                            0.0056127743,
                            0.03212357,
                            0.015467451,
                            0.016087374,
                            -0.037531022,
                            0.012665385,
                            -0.043193664,
                            -0.00012406892,
                            0.025179362,
                            0.0055069216,
                            0.0063957963,
                            -0.002517324,
                            -0.016555985,
                            0.020920094,
                            0.011394156,
                            0.014619687,
                            -0.038390182,
                            0.0439744,
                            -0.011482927,
                            0.009825423,
                            -0.004160354,
                            0.020368997,
                            0.03491698,
                            -0.0034704967,
                            0.0021255706,
                            0.010282736,
                            0.015724726,
                            0.045868374,
                            0.024184162,
                            -0.0134936115,
                            0.0044668233,
                            0.0012478447,
                            -0.013371312,
                            -0.0085783675,
                            0.01731196,
                            -0.00016034678,
                            0.024397139,
                            0.042506926,
                            -0.0017551893,
                            0.0019976632,
                            0.040093794,
                            -0.037392195,
                            -0.01965574,
                            -0.018734004,
                            0.05009869,
                            0.011303273,
                            -0.032823566,
                            0.014475997,
                            -0.012336681,
                            -0.011140458,
                            0.000971855,
                            -0.012117022,
                            -0.0381897,
                            0.034103855,
                            0.00014202195,
                            0.045798495,
                            0.034632802,
                            -0.024270294,
                            0.01950017,
                            -0.0072281132,
                            0.02100011,
                            -0.0096343625,
                            0.028804017,
                            0.0009235383,
                            0.016021773,
                            0.030110111,
                            -0.028152337,
                            0.012108085,
                            -0.017204683,
                            -0.02713978,
                            0.0075973463,
                            0.027334211,
                            -0.036430154,
                            -0.048451085,
                            0.00040952556,
                            0.021199103,
                            -0.00603127,
                            0.03748226,
                            0.005657242,
                            0.015573037,
                            0.026719728,
                            -0.013274318,
                            0.017131511,
                            -0.02267349,
                            0.0505698,
                            0.005665451,
                            -0.011916767,
                            0.008605462,
                            -0.3563205,
                            0.027140133,
                            -0.039547928,
                            0.0076093213,
                            0.018632373,
                            0.03625323,
                            0.00023610593,
                            0.025215266,
                            -0.012984429,
                            -0.0066881296,
                            -0.0063353865,
                            -0.022111852,
                            -0.06008927,
                            -0.0047209873,
                            0.027827973,
                            -0.0027215409,
                            0.0032449246,
                            -0.0150848925,
                            -0.0034249797,
                            -0.024533842,
                            0.013539898,
                            -0.012121895,
                            0.009372096,
                            0.03403872,
                            0.028711831,
                            -0.011447683,
                            -0.019878032,
                            -0.005147342,
                            0.0025633152,
                            0.003246234,
                            -0.020177472,
                            -0.0004185361,
                            -0.017794114,
                            -0.0025014188,
                            -0.029792536,
                            0.045846906,
                            -0.008731828,
                            0.054890458,
                            0.017330898,
                            -0.02630449,
                            0.0359795,
                            0.011844687,
                            0.0386938,
                            -0.018248042,
                            -8.227386e-06,
                            -0.01210922,
                            0.01745892,
                            -0.11931442,
                            0.01669014,
                            -0.023421858,
                            -0.003701275,
                            -0.027903575,
                            -0.012578226,
                            0.0044612056,
                            0.006826556,
                            0.0142461695,
                            -0.025385372,
                            -0.060161296,
                            -0.023865903,
                            0.018704308,
                            -0.0039240117,
                            -0.016665233,
                            0.00393121,
                            0.02005786,
                            -0.011810783,
                            0.001141533,
                            -0.029942162,
                            0.23249598,
                            0.010486518,
                            -0.03267631,
                            -0.016784158,
                            0.009184899,
                            0.017583698,
                            -0.06940357,
                            0.020634983,
                            0.0098996125,
                            -0.0038242722,
                            -0.008168517,
                            -0.009534711,
                            -0.013586409,
                            -0.007924422,
                            0.029386882,
                            0.03626113,
                            0.021722483,
                            0.021595681,
                            0.02295795,
                            0.020884829,
                            -0.01831911,
                            0.010178437,
                            -9.788741e-05,
                            0.0410476,
                            0.027140489,
                            -0.009013686,
                            -0.007957704,
                            0.0073299655,
                            0.013554522,
                            0.03406621,
                            0.0086602885,
                            -0.004206749,
                            -0.00726234,
                            0.0026538165,
                            0.031023094,
                            0.023193588,
                            -0.00366463,
                            0.0079810675,
                            -0.015076812,
                            -0.00048952753,
                            -0.017639315,
                            -0.013687591,
                            -0.00878102,
                            0.025409997,
                            -0.012098391,
                            -0.0010359066,
                            0.030746698,
                            0.04868354,
                            0.0048578926,
                            -0.011650512,
                            0.01949775,
                            0.0038707622,
                            -0.03291529,
                            -0.013797636,
                            -0.008437529,
                            0.017869594,
                            -0.019448569,
                            0.0072278082,
                            0.008521242,
                            0.0016531241,
                            0.01663008,
                            -0.022253664,
                            -0.034466095,
                            -0.045577172,
                            0.024964234,
                            0.0014992359,
                            0.043162532,
                            -0.017614044,
                            -0.018294306,
                            0.011179284,
                            0.027475428,
                            -0.042453974,
                            -0.058339074,
                            0.003713978,
                            -0.008902841,
                            -0.023268076,
                            -0.013861255,
                            -0.022229131,
                            -0.011027949,
                            -0.02439343,
                            -0.0065350975,
                            -0.0008731903,
                            0.01248304,
                            -0.0136078475,
                            -0.0029668908,
                            -0.020759333,
                            -0.012891743,
                            -0.022692317,
                            -0.0015963655,
                            -0.011657416,
                            0.026364863,
                            0.022736192,
                            -0.022828251,
                            -0.016368246,
                            -0.025395468,
                            -0.002188065,
                            -0.014938493,
                            -0.020719102,
                            0.02591615,
                            0.007264857,
                            0.0210535,
                            -0.038810164,
                            0.023895852,
                            0.020885067,
                            0.0069733155,
                            -0.021368885,
                            0.015740661,
                            0.035886213,
                            0.038927548,
                            0.0037099477,
                            -0.016085878,
                            -0.0035345098,
                            0.020996144,
                            0.008206138,
                            -0.022769658,
                            -0.016938368,
                            0.031417664,
                            0.010775718,
                            -0.025795154,
                            -0.028287001,
                            -0.009441712,
                            0.018597499,
                            0.0072308434,
                            -0.008797003,
                            0.03939681,
                            0.020777853,
                            0.05956594,
                            0.03252825,
                            -0.033422485,
                            0.020982997,
                            -0.014228513,
                            -0.019492384,
                            0.018416215,
                            -0.03916733,
                            -0.0069780224,
                            -0.0152080925,
                            0.020285297,
                            -0.04185614,
                            -0.03160447,
                            -0.021927007,
                            -0.02512525,
                            0.005409058,
                            0.02780378,
                            -0.0014170316,
                            -0.03552618,
                            -0.008779973,
                            -0.023295147,
                            0.015702697,
                            -0.0018499467,
                            0.027807372,
                            -0.0016777726,
                            -0.028280752,
                            0.001880646,
                            0.021517469,
                            -0.021920834,
                            0.0042870604,
                            0.0016790206,
                            -0.013460576,
                            0.088902965,
                            0.014514668,
                            -0.05025462,
                            -0.004923846,
                            -0.010813537,
                            -0.031683378,
                            0.028117152,
                            -0.004611733,
                            -0.0134712625,
                            0.024858989,
                            -0.007158381,
                            -0.027374245,
                            -0.040916014,
                            0.013493191,
                            0.014128484,
                            -0.002084427,
                            -0.0015982436,
                            -0.009864484,
                            0.040343057,
                            -0.02681094,
                            0.016389828,
                            0.018494258,
                            -0.010489675,
                            0.020642819,
                            0.02707498,
                            -0.026973462,
                            -0.014743578,
                            -0.007373948,
                            -0.015010584,
                            0.0464746,
                            0.011341716,
                            -0.025844352,
                            -0.017612807,
                            -0.009791927,
                            -0.0016559958,
                            0.017630955,
                        ]
                    ],
                },
            }
        )


@pytest.mark.skipif(not sentence_transformers_imports_successful(), reason='SentenceTransformers not installed')
class TestSentenceTransformers:
    def _load_stsb_bert_tiny_model(self):
        # The pinned commit revision lets huggingface_hub serve every model file
        # straight from a warm cache without revalidating it against the Hub.
        # Construction still fires a few metadata requests (model card, repo tree,
        # adapter probe); those tolerate Hub HTTP errors, and connection-level
        # failures fall through to the skip below. CI warms the cache out-of-band
        # (see ci.yml); a cold cache downloads the model here on first use.
        try:
            model = SentenceTransformer(STSB_BERT_TINY_MODEL, revision=STSB_BERT_TINY_REVISION)
        except (OSError, RuntimeError, httpx.HTTPError) as e:
            # Skip only when the Hub is unavailable (see `_hf_hub_unavailable`) so a
            # HF outage never reds the whole suite; anything else (bad pin, auth
            # problem, corrupt cache, dependency mismatch) fails loudly.
            if not _hf_hub_unavailable(e):
                raise
            pytest.skip(f'sentence-transformers test model unavailable (HF Hub): {e}')
        # If the model card didn't load, `model_id` is unset and the model reports
        # its name as 'unknown'. Set it explicitly so the reported name is
        # deterministic (a no-op when the card did load).
        model.model_card_data.model_id = STSB_BERT_TINY_MODEL
        model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing
        return model

    @pytest.fixture(scope='session')
    def stsb_bert_tiny_model(self):
        return self._load_stsb_bert_tiny_model()

    def test_model_unavailable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(f'{__name__}.SentenceTransformer', MagicMock(side_effect=httpx.ConnectTimeout('offline')))
        skip = MagicMock(side_effect=RuntimeError('skipped'))
        monkeypatch.setattr(pytest, 'skip', skip)

        with pytest.raises(RuntimeError, match='skipped'):
            self._load_stsb_bert_tiny_model()
        skip.assert_called_once_with('sentence-transformers test model unavailable (HF Hub): offline')

    @pytest.fixture
    def embedder(self, stsb_bert_tiny_model: Any) -> Embedder:
        return Embedder(SentenceTransformerEmbeddingModel(stsb_bert_tiny_model))

    async def test_embed_is_exempt_from_request_guard(self, embedder: Embedder):
        """`ALLOW_MODEL_REQUESTS`'s docstring promises this model is unaffected; pin that promise.

        Inference is local, so there's no provider call to block and no recording to make.
        """
        with pydantic_ai.models.override_allow_model_requests(False):
            result = await embedder.embed_query('hello')

        assert result.embeddings

    async def test_infer_model(self):
        model = infer_embedding_model('sentence-transformers:all-MiniLM-L6-v2')
        assert isinstance(model, SentenceTransformerEmbeddingModel)
        assert model.model_name == 'all-MiniLM-L6-v2'
        assert model.system == 'sentence-transformers'
        assert model.base_url is None

    async def test_adapter_without_downloaded_model(self, monkeypatch: pytest.MonkeyPatch):
        """VCR cannot exercise a local adapter, so cover it without a downloaded model."""
        st_model = MagicMock(spec=SentenceTransformer)
        st_model.model_card_data = MagicMock()
        st_model.model_card_data.model_id = 'test-model'
        st_model.encode_query.return_value.tolist.return_value = [[0.1, 0.2]]
        st_model.encode_document.return_value.tolist.return_value = [[0.3, 0.4], [0.5, 0.6]]
        st_model.get_max_seq_length.return_value = 512
        st_model.tokenize.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}

        def preserve_model(model: SentenceTransformer) -> SentenceTransformer:
            return model

        monkeypatch.setattr(sentence_transformers_module, 'deepcopy', preserve_model)

        model = SentenceTransformerEmbeddingModel(
            st_model,
            settings=SentenceTransformersEmbeddingSettings(
                dimensions=2,
                sentence_transformers_batch_size=2,
                sentence_transformers_device='cpu',
                sentence_transformers_normalize_embeddings=True,
            ),
        )
        embedder = Embedder(model)

        query_result = await embedder.embed_query('hello')
        assert query_result == snapshot(
            EmbeddingResult(
                embeddings=[[0.1, 0.2]],
                inputs=['hello'],
                input_type='query',
                model_name='test-model',
                provider_name='sentence-transformers',
                timestamp=IsDatetime(),
            )
        )
        st_model.encode_query.assert_called_once_with(
            ['hello'],
            show_progress_bar=False,
            convert_to_numpy=True,
            convert_to_tensor=False,
            device='cpu',
            normalize_embeddings=True,
            truncate_dim=2,
            batch_size=2,
        )

        documents_result = await embedder.embed_documents(['hello', 'world'])
        assert documents_result == snapshot(
            EmbeddingResult(
                embeddings=[[0.3, 0.4], [0.5, 0.6]],
                inputs=['hello', 'world'],
                input_type='document',
                model_name='test-model',
                provider_name='sentence-transformers',
                timestamp=IsDatetime(),
            )
        )
        st_model.encode_document.assert_called_once_with(
            ['hello', 'world'],
            show_progress_bar=False,
            convert_to_numpy=True,
            convert_to_tensor=False,
            device='cpu',
            normalize_embeddings=True,
            truncate_dim=2,
            batch_size=2,
        )
        assert await embedder.max_input_tokens() == 512
        assert await embedder.count_tokens('hello') == 3

        loaded_model = MagicMock(spec=SentenceTransformer)
        loaded_model.get_max_seq_length.return_value = 256
        constructor = MagicMock(return_value=loaded_model)
        monkeypatch.setattr(sentence_transformers_module, 'SentenceTransformer', constructor)

        lazy_embedder = Embedder(SentenceTransformerEmbeddingModel('lazy-model'))
        assert await lazy_embedder.max_input_tokens() == 256
        constructor.assert_called_once_with('lazy-model')

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
    or not voyageai_imports_successful()
    or not bedrock_imports_successful(),
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
    google_names = [f'google:{n}' for n in get_model_names(LatestGoogleGLAEmbeddingModelNames)]
    google_cloud_names = [f'google-cloud:{n}' for n in get_model_names(LatestGoogleVertexEmbeddingModelNames)]
    voyageai_names = [f'voyageai:{n}' for n in get_model_names(LatestVoyageAIEmbeddingModelNames)]
    bedrock_names = [f'bedrock:{n}' for n in get_model_names(LatestBedrockEmbeddingModelNames)]

    generated_names = sorted(
        openai_names + cohere_names + google_names + google_cloud_names + voyageai_names + bedrock_names
    )

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
    assert m.instrumentation_settings.version == InstrumentationSettings().version

    assert m.model_name == model.model_name
    assert m.system == model.system
    assert m.base_url == model.base_url
    assert m.settings == model.settings

    assert (await m.embed('Hello, world!', input_type='query')).embeddings == (
        await model.embed('Hello, world!', input_type='query')
    ).embeddings
    assert await m.max_input_tokens() == await model.max_input_tokens()
    assert await m.count_tokens('Hello, world!') == await model.count_tokens('Hello, world!')

    options = InstrumentationSettings(version=5)
    Embedder.instrument_all(options)
    m = get_model()
    assert isinstance(m, InstrumentedEmbeddingModel)
    assert m.wrapped is model
    assert m.instrumentation_settings is options

    Embedder.instrument_all(False)
    assert get_model() is model


class ExplicitPortEmbeddingModel(TestEmbeddingModel):
    @property
    def base_url(self) -> str:
        return 'https://example.com:8000/v1'


class MalformedPortEmbeddingModel(TestEmbeddingModel):
    @property
    def base_url(self) -> str:
        return 'https://example.com:notaport/v1'


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
@pytest.mark.parametrize(
    'model_type,expected_server_attributes',
    [
        pytest.param(
            ExplicitPortEmbeddingModel,
            snapshot({'server.address': 'example.com', 'server.port': 8000}),
            id='explicit-port',
        ),
        pytest.param(MalformedPortEmbeddingModel, snapshot({}), id='malformed-port'),
    ],
)
async def test_instrumented_embedding_model_server_attributes(
    model_type: type[TestEmbeddingModel], expected_server_attributes: dict[str, str | int], capfire: CaptureLogfire
):
    """A `base_url` whose port isn't an integer omits the server attributes instead of failing the request.

    `urlparse` accepts the URL and only raises when `hostname`/`port` are read, so this is a unit test:
    no real provider produces a `base_url` that survives client construction and fails at attribute-building.
    """
    model = InstrumentedEmbeddingModel(model_type(), InstrumentationSettings())

    await model.embed('Hello, world!', input_type='query')

    [span] = capfire.exporter.exported_spans_as_dict()
    assert {k: v for k, v in span['attributes'].items() if k.startswith('server.')} == expected_server_attributes


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


def test_instrumented_span_exports_embeddings_when_content_included():
    """With include_content=True the exported span carries the embeddings its declared json_schema promises."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    embedder = Embedder(
        TestEmbeddingModel(dimensions=3),
        instrument=InstrumentationSettings(tracer_provider=tracer_provider, include_content=True),
    )
    result = embedder.embed_query_sync('hello')

    assert result.embeddings == [[1.0, 1.0, 1.0]]

    (span,) = span_exporter.get_finished_spans()
    attributes = span.attributes or {}
    json_schema = json.loads(str(attributes['logfire.json_schema']))
    assert json_schema['properties']['embeddings'] == {'type': 'array'}
    assert attributes['embeddings'] == json.dumps(result.embeddings)
    assert attributes['inputs'] == '["hello"]'
