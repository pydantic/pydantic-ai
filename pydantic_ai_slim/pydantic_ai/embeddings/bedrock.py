from __future__ import annotations

import functools
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import anyio
import anyio.to_thread

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.providers.bedrock import remove_bedrock_geo_prefix
from pydantic_ai.usage import RequestUsage

from .base import EmbeddingModel, EmbedInputType
from .result import EmbeddingResult
from .settings import EmbeddingSettings

try:
    from botocore.exceptions import ClientError
except ImportError as _import_error:
    raise ImportError(
        'Please install `boto3` to use Bedrock embedding models, '
        'you can use the `bedrock` optional group — `pip install "pydantic-ai-slim[bedrock]"`'
    ) from _import_error

if TYPE_CHECKING:
    from botocore.client import BaseClient
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime.type_defs import InvokeModelResponseTypeDef


LatestBedrockEmbeddingModelNames = Literal[
    'amazon.titan-embed-text-v1',
    'amazon.titan-embed-text-v2:0',
    'cohere.embed-english-v3',
    'cohere.embed-multilingual-v3',
    'cohere.embed-v4:0',
    'amazon.nova-2-multimodal-embeddings-v1:0',
]
"""Latest Bedrock embedding model names.

See [the Bedrock docs](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
for available embedding models.
"""

BedrockEmbeddingModelName = str | LatestBedrockEmbeddingModelNames
"""Possible Bedrock embedding model names."""


class BedrockEmbeddingSettings(EmbeddingSettings, total=False):
    """Settings used for a Bedrock embedding model request.

    All fields from [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings] are supported,
    plus Bedrock-specific settings prefixed with `bedrock_`.

    All settings are optional - if not specified, model defaults are used.

    **Note on `dimensions` parameter support:**

    - **Titan v1** (`amazon.titan-embed-text-v1`): Not supported (fixed: 1536)
    - **Titan v2** (`amazon.titan-embed-text-v2:0`): Supported (default: 1024, accepts 256/384/1024)
    - **Cohere v3** (`cohere.embed-english-v3`, `cohere.embed-multilingual-v3`): Not supported (fixed: 1024)
    - **Cohere v4** (`cohere.embed-v4:0`): Supported (default: 1536, accepts 256/512/1024/1536)
    - **Nova** (`amazon.nova-2-multimodal-embeddings-v1:0`): Supported (default: 3072, accepts 256/384/1024/3072)

    Unsupported settings are silently ignored.

    **Note on `truncate` parameter support:**

    - **Titan models** (`amazon.titan-embed-text-v1`, `amazon.titan-embed-text-v2:0`): Not supported
    - **Cohere models** (all versions): Supported (default: `False`, maps to `'END'` when `True`)
    - **Nova** (`amazon.nova-2-multimodal-embeddings-v1:0`): Supported (default: `False`, maps to `'END'` when `True`)

    For fine-grained truncation control, use model-specific settings: `bedrock_cohere_truncate` or `bedrock_nova_truncate`.

    Example:
        ```python
        from pydantic_ai.embeddings.bedrock import BedrockEmbeddingSettings

        # Use model defaults
        settings = BedrockEmbeddingSettings()

        # Customize specific settings for Titan v2:0
        settings = BedrockEmbeddingSettings(
            dimensions=512,
            bedrock_titan_normalize=True,
        )

        # Customize specific settings for Cohere v4
        settings = BedrockEmbeddingSettings(
            dimensions=512,
            bedrock_cohere_max_tokens=1000,
        )
        ```
    """

    # ALL FIELDS MUST BE `bedrock_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    # ==================== Amazon Titan Settings ====================

    bedrock_titan_normalize: bool
    """Whether to normalize embedding vectors for Titan models.

    **Supported by:** `amazon.titan-embed-text-v2:0` (default: `True`)

    **Not supported by:** `amazon.titan-embed-text-v1` (silently ignored)

    When enabled, vectors are normalized for direct cosine similarity calculations.
    """

    # ==================== Cohere Settings ====================

    bedrock_cohere_max_tokens: int
    """The maximum number of tokens to embed for Cohere models.

    **Supported by:** `cohere.embed-v4:0` (default: 128000)

    **Not supported by:** `cohere.embed-english-v3`, `cohere.embed-multilingual-v3`
    (silently ignored)
    """

    bedrock_cohere_input_type: Literal['search_document', 'search_query', 'classification', 'clustering']
    """The input type for Cohere models.

    **Supported by:** All Cohere models (`cohere.embed-english-v3`, `cohere.embed-multilingual-v3`, `cohere.embed-v4:0`)

    By default, `embed_query()` uses `'search_query'` and `embed_documents()` uses `'search_document'`.
    Also accepts `'classification'` or `'clustering'`.
    """

    bedrock_cohere_truncate: Literal['NONE', 'START', 'END']
    """The truncation strategy for Cohere models. Overrides base `truncate` setting.

    **Supported by:** All Cohere models (`cohere.embed-english-v3`, `cohere.embed-multilingual-v3`, `cohere.embed-v4:0`)

    Default: `'NONE'`

    - `'NONE'`: Raise an error if input exceeds max tokens.
    - `'START'`: Truncate the start of the input.
    - `'END'`: Truncate the end of the input.
    """

    # ==================== Amazon Nova Settings ====================

    bedrock_nova_truncate: Literal['NONE', 'START', 'END']
    """The truncation strategy for Nova models. Overrides base `truncate` setting.

    **Supported by:** `amazon.nova-2-multimodal-embeddings-v1:0`

    Default: `'NONE'`

    - `'NONE'`: Raise an error if input exceeds max tokens.
    - `'START'`: Truncate the start of the input.
    - `'END'`: Truncate the end of the input.
    """

    bedrock_nova_embedding_purpose: Literal[
        'GENERIC_INDEX',
        'GENERIC_RETRIEVAL',
        'TEXT_RETRIEVAL',
        'CLASSIFICATION',
        'CLUSTERING',
    ]
    """The embedding purpose for Nova models.

    **Supported by:** `amazon.nova-2-multimodal-embeddings-v1:0`

    By default, `embed_query()` uses `'GENERIC_RETRIEVAL'` and `embed_documents()` uses `'GENERIC_INDEX'`.
    Also accepts `'TEXT_RETRIEVAL'`, `'CLASSIFICATION'`, or `'CLUSTERING'`.

    Note: Multimodal-specific purposes (`'IMAGE_RETRIEVAL'`, `'VIDEO_RETRIEVAL'`,
    `'DOCUMENT_RETRIEVAL'`, `'AUDIO_RETRIEVAL'`) are not supported as this
    embedding client only accepts text input.
    """

    # ==================== Concurrency Settings ====================

    bedrock_max_concurrency: int
    """Maximum number of concurrent requests for models that don't support batch embedding.

    **Applies to:** `amazon.titan-embed-text-v1`, `amazon.titan-embed-text-v2:0`,
    `amazon.nova-2-multimodal-embeddings-v1:0`

    When embedding multiple texts with models that only support single-text requests,
    this controls how many requests run in parallel. Defaults to 5.
    """


# Max input tokens lookup (keys are normalized model names as returned by remove_bedrock_geo_prefix)
_MAX_INPUT_TOKENS: dict[str, int] = {
    'amazon.titan-embed-text-v1': 8192,
    'amazon.titan-embed-text-v2:0': 8192,
    'cohere.embed-english-v3': 512,
    'cohere.embed-multilingual-v3': 512,
    'cohere.embed-v4:0': 128000,
    'amazon.nova-2-multimodal-embeddings-v1:0': 8192,
}


def _extract_version(model_name: str) -> int | None:
    """Extract the version number from a model name.

    Examples:
        - 'amazon.titan-embed-text-v1' -> 1
        - 'amazon.titan-embed-text-v2:0' -> 2
        - 'cohere.embed-english-v3' -> 3
        - 'cohere.embed-v4:0' -> 4
    """
    if match := re.search(r'v(\d+)', model_name):
        return int(match.group(1))
    else:  # pragma: no cover
        return None


class _BedrockEmbeddingHandler(ABC):
    """Abstract handler for processing different Bedrock embedding model formats."""

    model_name: str

    def __init__(self, model_name: str):
        """Initialize the handler with the model name.

        Args:
            model_name: The normalized model name (e.g., 'amazon.titan-embed-text-v2:0').
        """
        self.model_name = model_name

    @property
    def supports_batch(self) -> bool:
        """Whether this handler supports batch embedding in a single request."""
        return False

    @abstractmethod
    def prepare_request(
        self,
        texts: list[str],
        input_type: EmbedInputType,
        settings: BedrockEmbeddingSettings,
    ) -> dict[str, Any]:
        """Prepare the request body for the embedding model."""
        raise NotImplementedError

    @abstractmethod
    def parse_response(
        self,
        response_body: dict[str, Any],
    ) -> tuple[list[Sequence[float]], str | None]:
        """Parse the response from the embedding model.

        Args:
            response_body: The parsed JSON response body.

        Returns:
            A tuple of (embeddings, response_id). response_id may be None.
        """
        raise NotImplementedError


class _TitanEmbeddingHandler(_BedrockEmbeddingHandler):
    """Handler for Amazon Titan embedding models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._version = _extract_version(model_name)

    def prepare_request(
        self,
        texts: list[str],
        input_type: EmbedInputType,
        settings: BedrockEmbeddingSettings,
    ) -> dict[str, Any]:
        assert len(texts) == 1, 'Titan only supports single text per request'
        body: dict[str, Any] = {'inputText': texts[0]}

        dimensions = settings.get('dimensions')
        normalize = settings.get('bedrock_titan_normalize')

        match self._version:
            case 1:
                # Titan v1 doesn't support dimensions or normalize parameters - silently ignored
                pass
            case _:
                # Titan v2+: Apply dimensions if provided
                if dimensions is not None:
                    body['dimensions'] = dimensions

                # Titan v2+: Default normalize to True if not explicitly set
                if normalize is None:
                    body['normalize'] = True
                else:
                    body['normalize'] = normalize

        return body

    def parse_response(
        self,
        response_body: dict[str, Any],
    ) -> tuple[list[Sequence[float]], str | None]:
        embedding = response_body['embedding']
        return [embedding], None


class _CohereEmbeddingHandler(_BedrockEmbeddingHandler):
    """Handler for Cohere embedding models on Bedrock."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._version = _extract_version(model_name)

    @property
    def supports_batch(self) -> bool:
        """Cohere models support batch embedding."""
        return True

    def prepare_request(
        self,
        texts: list[str],
        input_type: EmbedInputType,
        settings: BedrockEmbeddingSettings,
    ) -> dict[str, Any]:
        cohere_input_type = settings.get(
            'bedrock_cohere_input_type', 'search_document' if input_type == 'document' else 'search_query'
        )

        body: dict[str, Any] = {
            'texts': texts,
            'input_type': cohere_input_type,
        }

        max_tokens = settings.get('bedrock_cohere_max_tokens')
        dimensions = settings.get('dimensions')

        match self._version:
            case 3:
                # Cohere v3 doesn't support max_tokens or dimensions parameters - silently ignored
                pass
            case _:
                # Cohere v4+: Apply max_tokens if provided
                if max_tokens is not None:
                    body['max_tokens'] = max_tokens

                # Cohere v4+: Apply dimensions if provided
                if dimensions is not None:
                    body['output_dimension'] = dimensions

        # Model-specific truncate takes precedence, then base truncate setting, then default to NONE
        if truncate := settings.get('bedrock_cohere_truncate'):
            body['truncate'] = truncate
        elif settings.get('truncate'):
            body['truncate'] = 'END'
        else:
            body['truncate'] = 'NONE'

        return body

    def parse_response(
        self,
        response_body: dict[str, Any],
    ) -> tuple[list[Sequence[float]], str | None]:
        # Cohere returns embeddings in different formats based on embedding_types parameter.
        # We always request float embeddings (the default when embedding_types is not specified).
        embeddings: list[Sequence[float]] | None = None
        if 'embeddings' in response_body:
            raw_embeddings = response_body['embeddings']
            if isinstance(raw_embeddings, dict):
                # embeddings_by_type response format - extract float embeddings
                float_emb = cast(dict[str, list[Sequence[float]]], raw_embeddings).get('float')
                embeddings = float_emb
            elif isinstance(raw_embeddings, list):
                # Direct float embeddings response
                embeddings = cast(list[Sequence[float]], raw_embeddings)

        if embeddings is None:  # pragma: no cover
            raise UnexpectedModelBehavior(
                'The Cohere Bedrock embeddings response did not have an `embeddings` field holding a list of floats',
                str(response_body),
            )

        return embeddings, response_body.get('id')


class _NovaEmbeddingHandler(_BedrockEmbeddingHandler):
    """Handler for Amazon Nova embedding models on Bedrock."""

    def prepare_request(
        self,
        texts: list[str],
        input_type: EmbedInputType,
        settings: BedrockEmbeddingSettings,
    ) -> dict[str, Any]:
        assert len(texts) == 1, 'Nova only supports single text per request'

        text = texts[0]

        # Get truncation mode - Nova requires this field
        # Model-specific truncate takes precedence, then base truncate setting
        # Nova accepts: START, END, NONE (default: NONE)
        if truncate := settings.get('bedrock_nova_truncate'):
            pass  # Use the model-specific setting
        elif settings.get('truncate'):
            truncate = 'END'
        else:
            truncate = 'NONE'

        # Build text params
        text_params: dict[str, Any] = {
            'value': text,
            'truncationMode': truncate,
        }

        # Nova requires embeddingPurpose - default based on input_type
        # - queries default to GENERIC_RETRIEVAL (optimized for search)
        # - documents default to GENERIC_INDEX (optimized for indexing)
        default_purpose = 'GENERIC_RETRIEVAL' if input_type == 'query' else 'GENERIC_INDEX'
        embedding_purpose = settings.get('bedrock_nova_embedding_purpose', default_purpose)

        single_embedding_params: dict[str, Any] = {
            'embeddingPurpose': embedding_purpose,
            'text': text_params,
        }

        # Nova: Apply dimensions if provided
        if (dims := settings.get('dimensions')) is not None:
            single_embedding_params['embeddingDimension'] = dims

        body: dict[str, Any] = {
            'taskType': 'SINGLE_EMBEDDING',
            'singleEmbeddingParams': single_embedding_params,
        }

        return body

    def parse_response(
        self,
        response_body: dict[str, Any],
    ) -> tuple[list[Sequence[float]], str | None]:
        # Nova returns embeddings in format: {"embeddings": [{"embeddingType": "TEXT", "embedding": [...]}]}
        embeddings_list = response_body.get('embeddings', [])
        if not embeddings_list:  # pragma: no cover
            raise UnexpectedModelBehavior(
                'The Nova Bedrock embeddings response did not have an `embeddings` field',
                str(response_body),
            )

        # Extract the embedding vector from the first item
        embedding = embeddings_list[0].get('embedding')
        if embedding is None:  # pragma: no cover
            raise UnexpectedModelBehavior(
                'The Nova Bedrock embeddings response did not have an `embedding` field in the first item',
                str(response_body),
            )

        return [embedding], None


# Mapping of model name prefixes to handler classes
_HANDLER_PREFIXES: dict[str, type[_BedrockEmbeddingHandler]] = {
    'amazon.titan-embed': _TitanEmbeddingHandler,
    'cohere.embed': _CohereEmbeddingHandler,
    'amazon.nova': _NovaEmbeddingHandler,
}


def _get_handler_for_model(model_name: str) -> _BedrockEmbeddingHandler:
    """Get the appropriate handler for a Bedrock embedding model."""
    normalized_name = remove_bedrock_geo_prefix(model_name)

    for prefix, handler_class in _HANDLER_PREFIXES.items():
        if normalized_name.startswith(prefix):
            return handler_class(normalized_name)

    raise UserError(
        f'Unsupported Bedrock embedding model: {model_name}. Supported model prefixes: {list(_HANDLER_PREFIXES.keys())}'
    )


@dataclass(init=False)
class BedrockEmbeddingModel(EmbeddingModel):
    """Bedrock embedding model implementation.

    This model works with AWS Bedrock's embedding models including
    Amazon Titan Embeddings and Cohere Embed models.

    Example:
    ```python
    from pydantic_ai.embeddings.bedrock import BedrockEmbeddingModel
    from pydantic_ai.providers.bedrock import BedrockProvider

    # Using default AWS credentials
    model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0')

    # Using explicit credentials
    model = BedrockEmbeddingModel(
        'cohere.embed-english-v3',
        provider=BedrockProvider(
            region_name='us-east-1',
            aws_access_key_id='...',
            aws_secret_access_key='...',
        ),
    )
    ```
    """

    client: BedrockRuntimeClient

    _model_name: BedrockEmbeddingModelName = field(repr=False)
    _provider: Provider[BaseClient] = field(repr=False)
    _handler: _BedrockEmbeddingHandler = field(repr=False)

    def __init__(
        self,
        model_name: BedrockEmbeddingModelName,
        *,
        provider: Literal['bedrock'] | Provider[BaseClient] = 'bedrock',
        settings: EmbeddingSettings | None = None,
    ):
        """Initialize a Bedrock embedding model.

        Args:
            model_name: The name of the Bedrock embedding model to use.
                See [Bedrock embedding models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
                for available options.
            provider: The provider to use for authentication and API access. Can be:

                - `'bedrock'` (default): Uses default AWS credentials
                - A [`BedrockProvider`][pydantic_ai.providers.bedrock.BedrockProvider] instance
                  for custom configuration

            settings: Model-specific [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings]
                to use as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        self.client = cast('BedrockRuntimeClient', provider.client)
        self._handler = _get_handler_for_model(model_name)

        super().__init__(settings=settings)

    @property
    def base_url(self) -> str:
        """The base URL for the provider API."""
        return str(self.client.meta.endpoint_url)

    @property
    def model_name(self) -> BedrockEmbeddingModelName:
        """The embedding model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The embedding model provider."""
        return self._provider.name

    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        inputs_list, settings_dict = self.prepare_embed(inputs, settings)
        settings_typed = cast(BedrockEmbeddingSettings, settings_dict)

        if self._handler.supports_batch:
            # Models like Cohere support batch requests
            return await self._embed_batch(inputs_list, input_type, settings_typed)
        else:
            # Models like Titan require individual requests
            return await self._embed_concurrent(inputs_list, input_type, settings_typed)

    async def _embed_batch(
        self,
        inputs: list[str],
        input_type: EmbedInputType,
        settings: BedrockEmbeddingSettings,
    ) -> EmbeddingResult:
        """Embed all inputs in a single batch request."""
        body = self._handler.prepare_request(inputs, input_type, settings)
        response, input_tokens = await self._invoke_model(body)
        embeddings, response_id = self._handler.parse_response(response)

        return EmbeddingResult(
            embeddings=embeddings,
            inputs=inputs,
            input_type=input_type,
            usage=RequestUsage(input_tokens=input_tokens),
            model_name=self.model_name,
            provider_name=self.system,
            provider_response_id=response_id,
        )

    async def _embed_concurrent(
        self,
        inputs: list[str],
        input_type: EmbedInputType,
        settings: BedrockEmbeddingSettings,
    ) -> EmbeddingResult:
        """Embed inputs concurrently with controlled parallelism and combine results."""
        max_concurrency = settings.get('bedrock_max_concurrency', 5)
        semaphore = anyio.Semaphore(max_concurrency)

        results: list[tuple[Sequence[float], int]] = [None] * len(inputs)  # type: ignore[list-item]

        async def embed_single(index: int, text: str) -> None:
            async with semaphore:
                body = self._handler.prepare_request([text], input_type, settings)
                response, input_tokens = await self._invoke_model(body)
                embeddings, _ = self._handler.parse_response(response)
                results[index] = (embeddings[0], input_tokens)

        async with anyio.create_task_group() as tg:
            for i, text in enumerate(inputs):
                tg.start_soon(embed_single, i, text)

        all_embeddings = [embedding for embedding, _ in results]
        total_input_tokens = sum(tokens for _, tokens in results)

        return EmbeddingResult(
            embeddings=all_embeddings,
            inputs=inputs,
            input_type=input_type,
            usage=RequestUsage(input_tokens=total_input_tokens),
            model_name=self.model_name,
            provider_name=self.system,
        )

    async def _invoke_model(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Invoke the Bedrock model and return parsed response with token count.

        Returns:
            A tuple of (response_body, input_token_count).
        """
        try:
            response: InvokeModelResponseTypeDef = await anyio.to_thread.run_sync(
                functools.partial(
                    self.client.invoke_model,
                    modelId=self._model_name,
                    body=json.dumps(body),
                    contentType='application/json',
                    accept='application/json',
                )
            )
        except ClientError as e:
            status_code = e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            if isinstance(status_code, int):
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.response) from e
            raise ModelAPIError(model_name=self.model_name, message=str(e)) from e

        # Extract input token count from HTTP headers
        input_tokens = int(
            response.get('ResponseMetadata', {}).get('HTTPHeaders', {}).get('x-amzn-bedrock-input-token-count', '0')
        )

        response_body = json.loads(response['body'].read())
        return response_body, input_tokens

    async def max_input_tokens(self) -> int | None:
        """Get the maximum number of tokens that can be input to the model."""
        return _MAX_INPUT_TOKENS.get(self._handler.model_name, None)
