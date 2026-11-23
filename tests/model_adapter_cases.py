"""One case per model class: how the suite builds it over a transport a test controls, and which of its entry points
call the provider's SDK.

Every concrete `Model`, `EmbeddingModel` and `ImageGenerationModel` class in `pydantic_ai.models`,
`pydantic_ai.embeddings` and `pydantic_ai.images` has an entry in `CASES`, keyed by class name:
`test_every_model_class_has_a_case` in `tests/test_model_adapter_conformance.py` fails until a new one gets one. The
entry is a `ModelAdapterCase` that builds the class over one of the transports in `tests/model_adapter_transports.py`,
or `Exempt(reason)` for a class with no provider SDK of its own: a wrapper, which `tests/test_wrapper_parity.py`
checks instead, a test double, or local inference.

The entry points are the base class's public coroutines and async context managers (`ENTRY_POINTS`). A case
classifies each one the class overrides as `CALLS_SDK`, `NoIO` or `NotSupported`; one the class inherits unchanged
raises the base's `NotImplementedError`. `sdk_cells()` lists every (class, entry point) that calls the SDK, for the
suites that run over every adapter: `tests/test_model_adapter_conformance.py` sends each one every transport fault
a provider can fail with. `tests/models/test_model_settings_support.py` builds its classes with the builds here.

This module imports no provider SDK at module level outside `try_import`, so it loads on every install lane.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache
from typing import Literal, NamedTuple, TypeGuard, get_type_hints

import httpx
import httpx2

from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.embeddings import EmbeddingModel
from pydantic_ai.exceptions import UserError
from pydantic_ai.images import ImageGenerationModel
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.batch import Batch, BatchRequest, BatchStatus
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.system_one import SystemOneProvider
from pydantic_ai.tools import ToolDefinition

from .capability_cases import caused_by_missing_dependency
from .conftest import try_import
from .manifests import Exempt
from .model_adapter_transports import (
    AdapterModel,
    BotocoreTransport,
    Httpx2Transport,
    HuggingFaceTransport,
    LegacyHttpxTransport,
    Transport,
    XaiTransport,
)

with try_import():
    from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
    from pydantic_ai.images.openai import OpenAIImageGenerationModel
    from pydantic_ai.models.bedrock_mantle import BedrockMantleChatModel, BedrockMantleResponsesModel
    from pydantic_ai.models.cerebras import CerebrasModel
    from pydantic_ai.models.crusoe import CrusoeModel
    from pydantic_ai.models.github_copilot import GitHubCopilotModel
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openai_codex import OpenAICodexModel
    from pydantic_ai.models.openai_decisions import OpenAIDecisionsModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.models.snowflake import SnowflakeModel
    from pydantic_ai.models.zai import ZaiModel
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider
    from pydantic_ai.providers.cerebras import CerebrasProvider
    from pydantic_ai.providers.crusoe import CrusoeProvider
    from pydantic_ai.providers.github_copilot import GitHubCopilotProvider
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    from pydantic_ai.providers.snowflake import SnowflakeProvider
    from pydantic_ai.providers.zai import ZaiProvider

with try_import():
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import():
    from pydantic_ai.embeddings.google import GoogleEmbeddingModel
    from pydantic_ai.images.google import GoogleImageGenerationModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.google_decisions import GoogleDecisionsModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import():
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

with try_import():
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

with try_import():
    from pydantic_ai.embeddings.cohere import CohereEmbeddingModel
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider

with try_import():
    from botocore.client import BaseClient

    from pydantic_ai.embeddings.bedrock import BedrockEmbeddingModel
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

with try_import():
    from huggingface_hub import AsyncInferenceClient

    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

with try_import():
    from xai_sdk import AsyncClient as AsyncXaiClient

    from pydantic_ai.images.xai import XaiImageGenerationModel
    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

with try_import():
    from pydantic_ai.models.typesafe import TypeSafeModel
    from pydantic_ai.providers.typesafe import TypeSafeProvider


ADAPTER_PACKAGES = ('pydantic_ai.models', 'pydantic_ai.embeddings', 'pydantic_ai.images')
"""The packages whose model classes `CASES` covers. A wrapper defined elsewhere, like a durable execution engine's, is
covered by `tests/test_wrapper_parity.py`."""

BASES: tuple[type[Model], type[EmbeddingModel], type[ImageGenerationModel]] = (
    Model,
    EmbeddingModel,
    ImageGenerationModel,
)


def _is_entry_point(member: object) -> bool:
    """A coroutine function, or an `@asynccontextmanager` around an async generator like `Model.request_stream`."""
    return inspect.iscoroutinefunction(member) or inspect.isasyncgenfunction(getattr(member, '__wrapped__', None))


ENTRY_POINTS: dict[type[AdapterModel], tuple[str, ...]] = {
    base: tuple(name for name, member in vars(base).items() if not name.startswith('_') and _is_entry_point(member))
    for base in BASES
}
"""Each base's public coroutines and async context managers, the calls through which a model reaches its provider."""


def base_of(model_class: type[AdapterModel]) -> type[AdapterModel]:
    return next(base for base in BASES if issubclass(model_class, base))


CALLS_SDK: Literal['calls_sdk'] = 'calls_sdk'
"""The entry point sends a request through the provider's SDK."""


@dataclass(frozen=True)
class NoIO:
    """The entry point answers without the provider, like a token count a local tokenizer makes."""

    reason: str
    invoke: bool = True
    """Whether a test may call it: one that does slow local work, like loading a tokenizer, is classified unchecked."""


@dataclass(frozen=True)
class NotSupported:
    """The entry point raises `raises` before sending anything, for this build."""

    raises: type[Exception]
    reason: str


EntryPointKind = Literal['calls_sdk'] | NoIO | NotSupported

INHERITED: dict[tuple[type[AdapterModel], str], EntryPointKind] = {
    (Model, 'cancel_suspended_response'): NoIO('the base class has no suspended responses to cancel'),
}
"""What an entry point a class inherits unchanged does, where the base class doesn't raise `NotImplementedError`."""


@dataclass(frozen=True)
class ModelAdapterCase:
    """How to build one model class over a transport a test controls, and how each of its entry points behaves."""

    transport: Transport
    entry_points: Mapping[str, EntryPointKind]
    """How each entry point the class overrides behaves for this build. One it inherits raises `NotImplementedError`."""
    parameters: ModelRequestParameters = field(default_factory=ModelRequestParameters)
    """What the requests the entry points are called with carry: a decision model answers only an output tool."""

    @property
    def reason(self) -> str:
        return f'built over a `{type(self.transport).__name__}`'

    def kind(self, model_class: type[AdapterModel], entry_point: str) -> EntryPointKind:
        if (kind := self.entry_points.get(entry_point)) is not None:
            return kind
        return INHERITED.get(
            (base_of(model_class), entry_point), NotSupported(NotImplementedError, 'the base class raises')
        )


class SdkCell(NamedTuple):
    """One entry point of one model class that sends a request through the provider's SDK."""

    model_class: type[AdapterModel]
    entry_point: str
    case: ModelAdapterCase

    @property
    def id(self) -> str:
        return f'{self.model_class.__name__}.{self.entry_point}'


def _is_model_class(obj: object) -> TypeGuard[type[AdapterModel]]:
    return isinstance(obj, type) and not inspect.isabstract(obj) and issubclass(obj, BASES)


@cache
def shipped_model_classes() -> tuple[dict[str, type[AdapterModel]], tuple[str, ...]]:
    """Every concrete model class in `ADAPTER_PACKAGES`, by name, and the modules the walk skipped.

    Walks the packages rather than reading an export list, so a class that is never re-exported is still covered.
    Modules whose optional dependency group is not installed are skipped: the class couldn't be used either.
    """
    found: dict[str, type[AdapterModel]] = {}
    skipped: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for package_name in ADAPTER_PACKAGES:
            package = importlib.import_module(package_name)
            for module_info in pkgutil.walk_packages(package.__path__, f'{package_name}.'):
                try:
                    module = importlib.import_module(module_info.name)
                except Exception as exc:
                    if not caused_by_missing_dependency(exc):
                        raise
                    skipped.append(module_info.name)
                    continue
                for obj in vars(module).values():
                    if _is_model_class(obj) and obj.__module__.startswith('pydantic_ai.'):
                        existing = found.setdefault(obj.__name__, obj)
                        assert existing is obj, f'two model classes are named {obj.__name__}'
    return found, tuple(skipped)


PROBE_KEY = 'probe-key'


def openai_chat(client: httpx2.AsyncClient) -> OpenAIChatModel:
    return OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=PROBE_KEY, http_client=client))


def openai_responses(client: httpx2.AsyncClient) -> OpenAIResponsesModel:
    return OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=PROBE_KEY, http_client=client))


def openai_codex(client: httpx2.AsyncClient) -> OpenAICodexModel:
    return OpenAICodexModel(
        'gpt-5.6-luna',
        provider=OpenAICodexProvider(
            credentials=OpenAICodexCredentials(access_token=PROBE_KEY, refresh_token=PROBE_KEY, account_id='probe'),
            http_client=client,
        ),
    )


def cerebras(client: httpx2.AsyncClient) -> CerebrasModel:
    return CerebrasModel('llama3.1-8b', provider=CerebrasProvider(api_key=PROBE_KEY, http_client=client))


def crusoe(client: httpx2.AsyncClient) -> CrusoeModel:
    return CrusoeModel('openai/gpt-oss-120b', provider=CrusoeProvider(api_key=PROBE_KEY, http_client=client))


def github_copilot(client: httpx2.AsyncClient) -> GitHubCopilotModel:
    # A GPT id: the Anthropic ids that reject sampling settings drop `temperature`/`top_p` before the wire.
    return GitHubCopilotModel('gpt-5.4', provider=GitHubCopilotProvider(api_key=PROBE_KEY, http_client=client))


def ollama(client: httpx2.AsyncClient) -> OllamaModel:
    return OllamaModel(
        'llama3.2', provider=OllamaProvider(base_url='http://probe/v1', api_key=PROBE_KEY, http_client=client)
    )


def openrouter(client: httpx2.AsyncClient) -> OpenRouterModel:
    return OpenRouterModel('openai/gpt-4o', provider=OpenRouterProvider(api_key=PROBE_KEY, http_client=client))


def snowflake(client: httpx2.AsyncClient) -> SnowflakeModel:
    return SnowflakeModel(
        'llama3.1-70b', provider=SnowflakeProvider(account='probe', token=PROBE_KEY, http_client=client)
    )


def zai(client: httpx2.AsyncClient) -> ZaiModel:
    return ZaiModel('glm-4.6', provider=ZaiProvider(api_key=PROBE_KEY, http_client=client))


def bedrock_mantle_chat(client: httpx2.AsyncClient) -> BedrockMantleChatModel:
    # `gpt-oss-safeguard-*` are the only Mantle models served on the Chat Completions interface.
    return BedrockMantleChatModel(
        'openai.gpt-oss-safeguard-20b',
        provider=BedrockMantleProvider(api_key=PROBE_KEY, region_name='us-east-1', http_client=client),
    )


def bedrock_mantle_responses(client: httpx2.AsyncClient) -> BedrockMantleResponsesModel:
    # GPT-5.4 keeps reasoning off by default, so sampling params are not dropped before the wire.
    return BedrockMantleResponsesModel(
        'openai.gpt-5.4',
        provider=BedrockMantleProvider(api_key=PROBE_KEY, region_name='us-east-1', http_client=client),
    )


def openai_decisions(client: httpx2.AsyncClient) -> OpenAIDecisionsModel:
    return OpenAIDecisionsModel('gpt-6-luna', provider=OpenAIProvider(api_key=PROBE_KEY, http_client=client))


def anthropic(client: httpx2.AsyncClient) -> AnthropicModel:
    return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=PROBE_KEY, http_client=client))


def groq(client: httpx.AsyncClient) -> GroqModel:
    return GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key=PROBE_KEY, http_client=client))


def mistral(client: httpx2.AsyncClient) -> MistralModel:
    return MistralModel('mistral-large-latest', provider=MistralProvider(api_key=PROBE_KEY, http_client=client))


def cohere(client: httpx.AsyncClient) -> CohereModel:
    return CohereModel('command-r-plus', provider=CohereProvider(api_key=PROBE_KEY, http_client=client))


def google(client: httpx2.AsyncClient) -> GoogleModel:
    return GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=PROBE_KEY, http_client=client))


def google_decisions(client: httpx2.AsyncClient) -> GoogleDecisionsModel:
    return GoogleDecisionsModel('gemini-decision-1', provider=GoogleProvider(api_key=PROBE_KEY, http_client=client))


def typesafe(client: httpx2.AsyncClient) -> TypeSafeModel:
    return TypeSafeModel('jev-latest', provider=TypeSafeProvider(api_key=PROBE_KEY, http_client=client))


def system_one(client: httpx2.AsyncClient) -> SystemOneModel:
    return SystemOneModel(
        'clm-latest', provider=SystemOneProvider(base_url='http://localhost:8700', http_client=client)
    )


def bedrock_converse() -> BedrockConverseModel:
    return BedrockConverseModel(
        'anthropic.claude-sonnet-4-5-20250929-v1:0',
        provider=BedrockProvider(aws_access_key_id=PROBE_KEY, aws_secret_access_key=PROBE_KEY, region_name='us-east-1'),
    )


def huggingface(hf_client: AsyncInferenceClient) -> HuggingFaceModel:
    return HuggingFaceModel('probe-model', provider=HuggingFaceProvider(api_key='hf_probe', hf_client=hf_client))


def xai(client: AsyncXaiClient) -> XaiModel:
    return XaiModel('grok-4', provider=XaiProvider(xai_client=client))


def _openai_embeddings(client: httpx2.AsyncClient) -> OpenAIEmbeddingModel:
    return OpenAIEmbeddingModel(
        'text-embedding-3-small', provider=OpenAIProvider(api_key=PROBE_KEY, http_client=client)
    )


def _cohere_embeddings(client: httpx.AsyncClient) -> CohereEmbeddingModel:
    return CohereEmbeddingModel('embed-v4.0', provider=CohereProvider(api_key=PROBE_KEY, http_client=client))


def _google_embeddings(client: httpx2.AsyncClient) -> GoogleEmbeddingModel:
    return GoogleEmbeddingModel('gemini-embedding-001', provider=GoogleProvider(api_key=PROBE_KEY, http_client=client))


def _bedrock_embeddings() -> BedrockEmbeddingModel:
    return BedrockEmbeddingModel(
        'amazon.titan-embed-text-v2:0',
        provider=BedrockProvider(aws_access_key_id=PROBE_KEY, aws_secret_access_key=PROBE_KEY, region_name='us-east-1'),
    )


def _openai_images(client: httpx2.AsyncClient) -> OpenAIImageGenerationModel:
    return OpenAIImageGenerationModel('gpt-image-2', provider=OpenAIProvider(api_key=PROBE_KEY, http_client=client))


def _google_images(client: httpx2.AsyncClient) -> GoogleImageGenerationModel:
    return GoogleImageGenerationModel(
        'gemini-2.5-flash-image', provider=GoogleProvider(api_key=PROBE_KEY, http_client=client)
    )


def _xai_images(client: AsyncXaiClient) -> XaiImageGenerationModel:
    return XaiImageGenerationModel('grok-imagine-image', provider=XaiProvider(xai_client=client))


def _boto3_client(model: AdapterModel) -> BaseClient:
    assert isinstance(model, BedrockConverseModel | BedrockEmbeddingModel)
    return model.client


DECISION_PARAMETERS = ModelRequestParameters(
    output_mode='tool',
    output_tools=[
        ToolDefinition(
            name='final_result', parameters_json_schema={'type': 'object', 'properties': {'ok': {'type': 'boolean'}}}
        )
    ],
    allow_text_output=False,
)
"""The one request shape a decision model accepts: an output tool and no function tools."""

_CHAT: dict[str, EntryPointKind] = {'request': CALLS_SDK, 'request_stream': CALLS_SDK}
_BATCH: dict[str, EntryPointKind] = {
    'batch_encode': NoIO('encodes the request as the batch API takes it'),
    'batch_create': CALLS_SDK,
    'batch_status': CALLS_SDK,
    'batch_results': CALLS_SDK,
    'batch_cancel': CALLS_SDK,
}
_NO_BATCH_API = NotSupported(UserError, 'the provider serves no OpenAI Batch API')
_NO_BATCHES: dict[str, EntryPointKind] = {
    name: _NO_BATCH_API for name in ('batch_encode', 'batch_create', 'batch_status', 'batch_results', 'batch_cancel')
}
_RESPONSES: dict[str, EntryPointKind] = {
    **_CHAT,
    **_BATCH,
    'count_tokens': CALLS_SDK,
    'compact_messages': CALLS_SDK,
    'cancel_suspended_response': CALLS_SDK,
}

_WRAPPER = Exempt(
    'forwards to the model it wraps; `tests/test_wrapper_parity.py` checks that it forwards each entry point'
)

CASES: dict[str, ModelAdapterCase | Exempt] = {
    'OpenAIChatModel': ModelAdapterCase(Httpx2Transport(openai_chat), _CHAT),
    'CerebrasModel': ModelAdapterCase(Httpx2Transport(cerebras), _CHAT),
    'CrusoeModel': ModelAdapterCase(Httpx2Transport(crusoe), _CHAT),
    'GitHubCopilotModel': ModelAdapterCase(Httpx2Transport(github_copilot), _CHAT),
    'OllamaModel': ModelAdapterCase(Httpx2Transport(ollama), _CHAT),
    'OpenRouterModel': ModelAdapterCase(Httpx2Transport(openrouter), _CHAT),
    'SnowflakeModel': ModelAdapterCase(Httpx2Transport(snowflake), _CHAT),
    'ZaiModel': ModelAdapterCase(Httpx2Transport(zai), _CHAT),
    'BedrockMantleChatModel': ModelAdapterCase(Httpx2Transport(bedrock_mantle_chat), _CHAT),
    'OpenAIResponsesModel': ModelAdapterCase(Httpx2Transport(openai_responses), _RESPONSES),
    'BedrockMantleResponsesModel': ModelAdapterCase(
        Httpx2Transport(bedrock_mantle_responses), {**_RESPONSES, **_NO_BATCHES}
    ),
    'OpenAICodexModel': ModelAdapterCase(
        Httpx2Transport(openai_codex),
        {
            **_RESPONSES,
            **_NO_BATCHES,
            'count_tokens': NotSupported(UserError, 'the Codex backend serves no input token count'),
        },
    ),
    'OpenAIDecisionsModel': ModelAdapterCase(Httpx2Transport(openai_decisions), _CHAT, DECISION_PARAMETERS),
    'AnthropicModel': ModelAdapterCase(
        Httpx2Transport(anthropic),
        {
            **_CHAT,
            **_BATCH,
            'count_tokens': CALLS_SDK,
            'compact_messages': NotSupported(UserError, '`claude-sonnet-4-5` has no on-demand compaction'),
        },
    ),
    'GroqModel': ModelAdapterCase(LegacyHttpxTransport(groq), _CHAT),
    'MistralModel': ModelAdapterCase(Httpx2Transport(mistral), _CHAT),
    'CohereModel': ModelAdapterCase(LegacyHttpxTransport(cohere), _CHAT),
    'GoogleModel': ModelAdapterCase(Httpx2Transport(google), {**_CHAT, **_BATCH, 'count_tokens': CALLS_SDK}),
    'GoogleDecisionsModel': ModelAdapterCase(Httpx2Transport(google_decisions), _CHAT, DECISION_PARAMETERS),
    'TypeSafeModel': ModelAdapterCase(Httpx2Transport(typesafe), _CHAT, DECISION_PARAMETERS),
    'SystemOneModel': ModelAdapterCase(Httpx2Transport(system_one), _CHAT, DECISION_PARAMETERS),
    'BedrockConverseModel': ModelAdapterCase(
        BotocoreTransport(bedrock_converse, _boto3_client), {**_CHAT, 'count_tokens': CALLS_SDK}
    ),
    'HuggingFaceModel': ModelAdapterCase(HuggingFaceTransport(huggingface), _CHAT),
    'XaiModel': ModelAdapterCase(XaiTransport(xai), _CHAT),
    'MCPSamplingModel': Exempt(
        'sends through the MCP client session it is given, not a provider SDK; '
        '`tests/models/test_mcp_sampling.py` covers it'
    ),
    'FallbackModel': Exempt('forwards each request to the models it falls back between'),
    'FunctionModel': Exempt('answers from the test function it is given'),
    'TestModel': Exempt('answers with generated output'),
    'WrapperModel': _WRAPPER,
    'InstrumentedModel': _WRAPPER,
    'ConcurrencyLimitedModel': _WRAPPER,
    'BatchedModel': _WRAPPER,
    'OpenAIEmbeddingModel': ModelAdapterCase(
        Httpx2Transport(_openai_embeddings),
        {
            'embed': CALLS_SDK,
            'max_input_tokens': NoIO('reads a table of known models'),
            'count_tokens': NoIO(
                'counts with `tiktoken`, which may fetch its encoding over `requests`, outside any transport',
                invoke=False,
            ),
        },
    ),
    'CohereEmbeddingModel': ModelAdapterCase(
        LegacyHttpxTransport(_cohere_embeddings),
        {'embed': CALLS_SDK, 'max_input_tokens': NoIO('reads a table of known models'), 'count_tokens': CALLS_SDK},
    ),
    'GoogleEmbeddingModel': ModelAdapterCase(
        Httpx2Transport(_google_embeddings),
        {'embed': CALLS_SDK, 'max_input_tokens': NoIO('reads a table of known models'), 'count_tokens': CALLS_SDK},
    ),
    'BedrockEmbeddingModel': ModelAdapterCase(
        BotocoreTransport(_bedrock_embeddings, _boto3_client),
        {'embed': CALLS_SDK, 'max_input_tokens': NoIO('reads a table of known models')},
    ),
    'VoyageAIEmbeddingModel': Exempt(
        'its provider takes only a `voyageai_client`, with no HTTP client to inject; '
        '`tests/test_embeddings_error_categories.py` fakes its SDK errors'
    ),
    'SentenceTransformerEmbeddingModel': Exempt('runs the model locally'),
    'TestEmbeddingModel': Exempt('answers with generated embeddings'),
    'WrapperEmbeddingModel': _WRAPPER,
    'InstrumentedEmbeddingModel': _WRAPPER,
    'OpenAIImageGenerationModel': ModelAdapterCase(Httpx2Transport(_openai_images), {'generate': CALLS_SDK}),
    'GoogleImageGenerationModel': ModelAdapterCase(Httpx2Transport(_google_images), {'generate': CALLS_SDK}),
    'XaiImageGenerationModel': ModelAdapterCase(XaiTransport(_xai_images), {'generate': CALLS_SDK}),
    'TestImageGenerationModel': Exempt('answers with generated images'),
    'WrapperImageGenerationModel': _WRAPPER,
    'InstrumentedImageGenerationModel': _WRAPPER,
}


def sdk_cells() -> list[SdkCell]:
    """Every (installed model class, entry point) that sends a request through the provider's SDK."""
    classes, _ = shipped_model_classes()
    return [
        SdkCell(model_class, entry_point, case)
        for name, model_class in sorted(classes.items())
        if isinstance(case := CASES.get(name), ModelAdapterCase)
        for entry_point, kind in case.entry_points.items()
        if kind == CALLS_SDK
    ]


def gave_up_body_reaches_model(factory: Callable[..., AdapterModel]) -> bool:
    """Whether the model `factory` builds classifies a status a retrying transport gave up on from its body.

    `factory`'s return annotation names the model class, whose SDK's HTTP client decides: a legacy `httpx` retrying
    transport closes the response it gave up on without reading its body, so only the status is left.
    """
    model_class: type[AdapterModel] = get_type_hints(factory)['return']
    case = CASES[model_class.__name__]
    return isinstance(case, ModelAdapterCase) and isinstance(case.transport, Httpx2Transport)


MESSAGES: list[ModelMessage] = [ModelRequest.user_text_prompt('hello')]


def _batch_request(parameters: ModelRequestParameters) -> BatchRequest:
    return BatchRequest(custom_id='request-1', messages=MESSAGES, model_request_parameters=parameters)


async def _batch(model: Model, parameters: ModelRequestParameters, status: BatchStatus = 'in_progress') -> Batch:
    """A handle to a batch the model's provider would have created for one request."""
    encoded = await model.batch_encode(_batch_request(parameters))
    return Batch(
        # A path like Gemini's `batches/{id}`, which the other APIs take as an opaque ID.
        id='batches/batch-1',
        model_name=model.model_name,
        provider_name=model.system,
        group=encoded.group,
        status=status,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        request_count=1,
        provider_details={
            'status': 'completed',
            'input_file_id': 'file-in',
            'output_file_id': 'file-out',
            'error_file_id': 'file-err',
        },
    )


async def _stream(model: Model, parameters: ModelRequestParameters) -> None:
    async with model_request_stream(model, MESSAGES, model_request_parameters=parameters) as stream:
        async for _ in stream:
            pass


async def _batch_create(model: Model, parameters: ModelRequestParameters) -> Batch:
    return await model.batch_create([await model.batch_encode(_batch_request(parameters))])


def _suspended_response(model: Model) -> ModelResponse:
    return ModelResponse(
        parts=[],
        model_name=model.model_name,
        provider_name=model.system,
        provider_response_id='resp-1',
        provider_details={'background': True},
        state='suspended',
    )


async def _batch_status(model: Model, parameters: ModelRequestParameters) -> Batch:
    return await model.batch_status(await _batch(model, parameters))


async def _batch_results(model: Model, parameters: ModelRequestParameters) -> object:
    return await model.batch_results(await _batch(model, parameters, 'ended'))


async def _batch_cancel(model: Model, parameters: ModelRequestParameters) -> Batch:
    return await model.batch_cancel(await _batch(model, parameters))


_MODEL_CALLS: dict[str, Callable[[Model, ModelRequestParameters], Awaitable[object]]] = {
    'request': lambda model, parameters: model_request(model, MESSAGES, model_request_parameters=parameters),
    'request_stream': _stream,
    'count_tokens': lambda model, parameters: model.count_tokens(MESSAGES, None, parameters),
    'compact_messages': lambda model, parameters: model.compact_messages(
        ModelRequestContext(model=model, messages=MESSAGES, model_settings=None, model_request_parameters=parameters)
    ),
    'batch_encode': lambda model, parameters: model.batch_encode(_batch_request(parameters)),
    'batch_create': _batch_create,
    'batch_status': _batch_status,
    'batch_results': _batch_results,
    'batch_cancel': _batch_cancel,
    'cancel_suspended_response': lambda model, parameters: model.cancel_suspended_response(_suspended_response(model)),
}


async def invoke(model: AdapterModel, entry_point: str, parameters: ModelRequestParameters | None = None) -> object:
    """Call `entry_point` on `model` with the smallest input that reaches the provider."""
    if isinstance(model, Model):
        return await _MODEL_CALLS[entry_point](model, parameters or ModelRequestParameters())
    if isinstance(model, EmbeddingModel):
        match entry_point:
            case 'embed':
                return await model.embed('hello', input_type='query')
            case 'max_input_tokens':
                return await model.max_input_tokens()
            case _:
                assert entry_point == 'count_tokens', entry_point
                return await model.count_tokens('hello')
    assert entry_point == 'generate', entry_point
    return await model.generate('a cat')
