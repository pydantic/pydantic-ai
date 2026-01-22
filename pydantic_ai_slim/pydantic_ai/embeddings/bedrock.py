from __future__ import annotations

import functools
import json
import warnings
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

    - **Supported by:** `amazon.titan-embed-text-v2:0`, `cohere.embed-v4:0`, `amazon.nova-2-multimodal-embeddings-v1:0`
    - **Not supported by:** `amazon.titan-embed-text-v1`, `cohere.embed-english-v3`, `cohere.embed-multilingual-v3`
      (will issue a warning if provided)

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

    **Supported by:** `amazon.titan-embed-text-v2:0`

    **Not supported by:** `amazon.titan-embed-text-v1` (will issue a warning if provided)

    When enabled, vectors are normalized for direct cosine similarity calculations.
    If not specified, defaults to `True` for v2 model.
    """

    # ==================== Cohere Settings ====================

    bedrock_cohere_max_tokens: int
    """The maximum number of tokens to embed for Cohere models.

    **Supported by:** `cohere.embed-v4:0`

    **Not supported by:** `cohere.embed-english-v3`, `cohere.embed-multilingual-v3`
    (will issue a warning if provided)
    """

    bedrock_cohere_input_type: Literal['search_document', 'search_query', 'classification', 'clustering']
    """The input type for Cohere models.

    **Supported by:** All Cohere models (`cohere.embed-english-v3`, `cohere.embed-multilingual-v3`, `cohere.embed-v4:0`)

    Defaults based on `input_type`:
    - `'query'` maps to `'search_query'`
    - `'document'` maps to `'search_document'`

    Other options: `'classification'`, `'clustering'`.
    """

    bedrock_cohere_truncate: Literal['NONE', 'START', 'END']
    """The truncation strategy for Cohere models. Overrides base `truncate` setting.

    **Supported by:** All Cohere models (`cohere.embed-english-v3`, `cohere.embed-multilingual-v3`, `cohere.embed-v4:0`)

    - `'NONE'` (default): Raise an error if input exceeds max tokens.
    - `'START'`: Truncate the start of the input.
    - `'END'`: Truncate the end of the input.
    """

    # ==================== Amazon Nova Settings ====================

    bedrock_nova_truncate: Literal['NONE', 'START', 'END']
    """The truncation strategy for Nova models. Overrides base `truncate` setting.

    **Supported by:** `amazon.nova-2-multimodal-embeddings-v1:0`

    - `'NONE'` (default): Raise an error if input exceeds max tokens.
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

    Defaults based on `input_type`:
    - `'query'` maps to `'GENERIC_RETRIEVAL'` (optimized for search)
    - `'document'` maps to `'GENERIC_INDEX'` (optimized for indexing)

    Other options: `'TEXT_RETRIEVAL'`, `'CLASSIFICATION'`, `'CLUSTERING'`.

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


class BedrockEmbeddingHandler(ABC):
    """Abstract handler for processing different Bedrock embedding model formats."""

    model_name: str
    config: BedrockModelConfig

    def __init__(self, model_name: str):
        """Initialize the handler with the model name.

        Args:
            model_name: The normalized model name (e.g., 'amazon.titan-embed-text-v2:0').
        """
        self.model_name = model_name
        config = _MODEL_CONFIG.get(model_name)
        assert config is not None, f'No config found for model: {model_name}'
        self.config = config

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


class TitanEmbeddingHandler(BedrockEmbeddingHandler):
    """Handler for Amazon Titan embedding models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._is_v1 = 'v1' in model_name

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

        if self._is_v1:
            # Titan v1 doesn't support dimensions or normalize parameters
            if dimensions is not None:
                warnings.warn(
                    f'The `dimensions` setting is not supported by {self.model_name} and will be ignored. '
                    'Only Titan v2 models support custom dimensions.',
                    UserWarning,
                )
            if normalize is not None:
                warnings.warn(
                    f'The `bedrock_titan_normalize` setting is not supported by {self.model_name} and will be ignored. '
                    'Only Titan v2 models support the normalize parameter.',
                    UserWarning,
                )
        else:
            # Titan v2: Apply dimensions (default from config)
            assert self.config.default_dimensions is not None, 'Titan v2 must have default_dimensions in config'
            body['dimensions'] = dimensions if dimensions is not None else self.config.default_dimensions

            # Titan v2: Default normalize to True if not explicitly set
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


class CohereEmbeddingHandler(BedrockEmbeddingHandler):
    """Handler for Cohere embedding models on Bedrock."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._is_v3 = 'v3' in model_name

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

        if self._is_v3:
            # Cohere v3 doesn't support max_tokens or dimensions parameters
            if max_tokens is not None:
                warnings.warn(
                    f'The `bedrock_cohere_max_tokens` setting is not supported by {self.model_name} and will be ignored. '
                    'Only Cohere v4 models support the max_tokens parameter.',
                    UserWarning,
                )
            if dimensions is not None:
                warnings.warn(
                    f'The `dimensions` setting is not supported by {self.model_name} and will be ignored. '
                    'Only Cohere v4 models support custom dimensions via the output_dimension parameter.',
                    UserWarning,
                )
        else:
            # Cohere v4: Apply max_tokens (default to max_input_tokens from config)
            body['max_tokens'] = max_tokens if max_tokens is not None else self.config.max_input_tokens

            # Cohere v4: Apply dimensions (default from config)
            assert self.config.default_dimensions is not None, 'Cohere v4 must have default_dimensions in config'
            body['output_dimension'] = dimensions if dimensions is not None else self.config.default_dimensions

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

        if embeddings is None:
            raise UnexpectedModelBehavior(
                'The Cohere Bedrock embeddings response did not have an `embeddings` field holding a list of floats',
                str(response_body),
            )

        return embeddings, response_body.get('id')


class NovaEmbeddingHandler(BedrockEmbeddingHandler):
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

        # Nova: Apply dimensions (default from config)
        assert self.config.default_dimensions is not None, 'Nova must have default_dimensions in config'
        single_embedding_params['embeddingDimension'] = settings.get('dimensions') or self.config.default_dimensions

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
        if not embeddings_list:
            raise UnexpectedModelBehavior(
                'The Nova Bedrock embeddings response did not have an `embeddings` field',
                str(response_body),
            )

        # Extract the embedding vector from the first item
        embedding = embeddings_list[0].get('embedding')
        if embedding is None:
            raise UnexpectedModelBehavior(
                'The Nova Bedrock embeddings response did not have an `embedding` field in the first item',
                str(response_body),
            )

        return [embedding], None


def _get_handler_for_model(model_name: str) -> BedrockEmbeddingHandler:
    """Get the appropriate handler for a Bedrock embedding model."""
    normalized_name = remove_bedrock_geo_prefix(model_name)

    if normalized_name.startswith('amazon.titan-embed'):
        return TitanEmbeddingHandler(normalized_name)
    elif normalized_name.startswith('cohere.embed'):
        return CohereEmbeddingHandler(normalized_name)
    elif normalized_name.startswith('amazon.nova'):
        return NovaEmbeddingHandler(normalized_name)
    else:
        raise UserError(
            f'Unsupported Bedrock embedding model: {model_name}. '
            f'Supported models: Amazon Titan Embed (amazon.titan-embed-*), '
            f'Cohere Embed (cohere.embed-*), Amazon Nova (amazon.nova-*)'
        )


@dataclass
class BedrockModelConfig:
    """Configuration for a Bedrock embedding model family."""

    max_input_tokens: int
    """Maximum number of input tokens the model accepts."""

    supports_batch: bool = False
    """Whether the model supports batch embedding in a single request."""

    default_dimensions: int | None = None
    """Default embedding dimensions. None means fixed dimensions (not configurable)."""


# Model configuration lookup (keys are normalized model names as returned by remove_bedrock_geo_prefix)
_MODEL_CONFIG: dict[str, BedrockModelConfig] = {
    # Titan V1: Fixed 1536 dimensions, no batch support
    'amazon.titan-embed-text-v1': BedrockModelConfig(
        max_input_tokens=8192,
    ),
    # Titan V2: Configurable dimensions (default 1024), no batch support
    'amazon.titan-embed-text-v2:0': BedrockModelConfig(
        max_input_tokens=8192,
        default_dimensions=1024,
    ),
    # Cohere V3: Fixed 1024 dimensions, batch support
    'cohere.embed-english-v3': BedrockModelConfig(
        max_input_tokens=512,
        supports_batch=True,
    ),
    'cohere.embed-multilingual-v3': BedrockModelConfig(
        max_input_tokens=512,
        supports_batch=True,
    ),
    # Cohere V4: Configurable dimensions (default 1536), batch support
    'cohere.embed-v4:0': BedrockModelConfig(
        max_input_tokens=128000,
        supports_batch=True,
        default_dimensions=1536,
    ),
    # Nova: Configurable dimensions (default 3072), no batch support
    'amazon.nova-2-multimodal-embeddings-v1:0': BedrockModelConfig(
        max_input_tokens=8192,
        default_dimensions=3072,
    ),
}


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
    _handler: BedrockEmbeddingHandler = field(repr=False)

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

        if self._handler.config.supports_batch:
            # Models like Cohere support batch requests
            return await self._embed_batch(inputs_list, input_type, settings_typed)
        else:
            # Models like Titan require individual requests
            return await self._embed_sequential(inputs_list, input_type, settings_typed)

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

    async def _embed_sequential(
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
        return self._handler.config.max_input_tokens
