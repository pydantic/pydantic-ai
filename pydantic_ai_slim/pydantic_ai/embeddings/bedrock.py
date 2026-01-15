from __future__ import annotations

import functools
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import anyio.to_thread
from botocore.exceptions import ClientError

from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.providers.bedrock import BEDROCK_GEO_PREFIXES
from pydantic_ai.usage import RequestUsage

from .base import EmbeddingModel, EmbedInputType
from .result import EmbeddingResult
from .settings import EmbeddingSettings

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
    """

    # ALL FIELDS MUST BE `bedrock_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    # ==================== Amazon Titan Settings ====================

    bedrock_normalize: bool
    """Whether to normalize the embedding vectors (Amazon Titan only).

    Normalized vectors can be used directly for similarity calculations.
    """

    # ==================== Cohere Settings ====================

    bedrock_input_type: Literal['search_document', 'search_query', 'classification', 'clustering']
    """The Cohere-specific input type for the embedding (Cohere models only).

    Overrides the standard `input_type` argument.
    """

    bedrock_truncate: Literal['NONE', 'LEFT', 'RIGHT', 'END', 'START']
    """The truncation strategy to use (Cohere and Nova models):

    - `'NONE'`: Raise an error if input exceeds max tokens.
    - `'LEFT'` / `'START'`: Truncate the start of the input text.
    - `'RIGHT'` / `'END'`: Truncate the end of the input text.

    Defaults to `'END'` for Nova, `'NONE'` for Cohere.
    """

    bedrock_embedding_types: list[Literal['float', 'int8', 'uint8', 'binary', 'ubinary']]
    """The embedding types to return (Cohere models only).

    Defaults to `['float']`.
    """

    # ==================== Amazon Nova Settings ====================

    bedrock_embedding_purpose: Literal[
        'GENERIC_INDEX',
        'GENERIC_RETRIEVAL',
        'TEXT_RETRIEVAL',
        'IMAGE_RETRIEVAL',
        'VIDEO_RETRIEVAL',
        'DOCUMENT_RETRIEVAL',
        'AUDIO_RETRIEVAL',
        'CLASSIFICATION',
        'CLUSTERING',
    ]
    """The embedding purpose for Nova models.

    - `'GENERIC_INDEX'` (default for documents): General-purpose indexing.
    - `'GENERIC_RETRIEVAL'` (default for queries): General-purpose retrieval.
    - `'TEXT_RETRIEVAL'`: Optimized for text retrieval tasks.
    - `'IMAGE_RETRIEVAL'`: Optimized for image retrieval tasks.
    - `'VIDEO_RETRIEVAL'`: Optimized for video retrieval tasks.
    - `'DOCUMENT_RETRIEVAL'`: Optimized for document retrieval tasks.
    - `'AUDIO_RETRIEVAL'`: Optimized for audio retrieval tasks.
    - `'CLASSIFICATION'`: Optimized for classification tasks.
    - `'CLUSTERING'`: Optimized for clustering tasks.
    """


class BedrockEmbeddingHandler(ABC):
    """Abstract handler for processing different Bedrock embedding model formats."""

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
        inputs: list[str],
        input_type: EmbedInputType,
        model_name: str,
        provider_name: str,
        provider_url: str,
        input_tokens: int,
    ) -> EmbeddingResult:
        """Parse the response from the embedding model.

        Args:
            response_body: The parsed JSON response body.
            inputs: The input texts that were embedded.
            input_type: The type of input (document or query).
            model_name: The name of the model.
            provider_name: The name of the provider.
            provider_url: The URL of the provider.
            input_tokens: The input token count from HTTP headers.
        """
        raise NotImplementedError

    @property
    def supports_batch(self) -> bool:
        """Whether the model supports batch embedding in a single request."""
        return False


class TitanEmbeddingHandler(BedrockEmbeddingHandler):
    """Handler for Amazon Titan embedding models."""

    @property
    def supports_batch(self) -> bool:
        return False  # Titan only supports single text per request

    def prepare_request(
        self,
        texts: list[str],
        input_type: EmbedInputType,
        settings: BedrockEmbeddingSettings,
    ) -> dict[str, Any]:
        # Titan only supports single text, caller must handle batching
        body: dict[str, Any] = {'inputText': texts[0]}

        # Optional: Set output dimensions (Titan v2 only)
        # Titan v2 supports: 256, 384, 1024 (default)
        if dimensions := settings.get('dimensions'):
            body['dimensions'] = dimensions

        # Optional: Normalize embedding vectors for direct similarity calculations
        if (normalize := settings.get('bedrock_normalize')) is not None:
            body['normalize'] = normalize

        return body

    def parse_response(
        self,
        response_body: dict[str, Any],
        inputs: list[str],
        input_type: EmbedInputType,
        model_name: str,
        provider_name: str,
        provider_url: str,
        input_tokens: int,
    ) -> EmbeddingResult:
        embedding = response_body['embedding']

        return EmbeddingResult(
            embeddings=[embedding],
            inputs=inputs,
            input_type=input_type,
            usage=RequestUsage(input_tokens=input_tokens),
            model_name=model_name,
            provider_name=provider_name,
        )


class CohereEmbeddingHandler(BedrockEmbeddingHandler):
    """Handler for Cohere embedding models on Bedrock."""

    @property
    def supports_batch(self) -> bool:
        return True

    def prepare_request(
        self,
        texts: list[str],
        input_type: EmbedInputType,
        settings: BedrockEmbeddingSettings,
    ) -> dict[str, Any]:
        # Map generic input_type to Cohere-specific input_type
        # - 'document' maps to 'search_document' (for indexing)
        # - 'query' maps to 'search_query' (for searching)
        # Can be overridden with bedrock_input_type setting
        cohere_input_type = settings.get(
            'bedrock_input_type', 'search_document' if input_type == 'document' else 'search_query'
        )

        body: dict[str, Any] = {
            'texts': texts,
            'input_type': cohere_input_type,
        }

        # Truncation strategy (default: NONE - raise error if input exceeds max tokens)
        # Cohere accepts: NONE, LEFT, RIGHT
        truncate = settings.get('bedrock_truncate', 'NONE')
        # Normalize truncation values (START/END -> LEFT/RIGHT for Cohere)
        if truncate == 'START':
            truncate = 'LEFT'
        elif truncate == 'END':
            truncate = 'RIGHT'
        body['truncate'] = truncate

        # Optional: Specify embedding types to return (default: float)
        # Cohere supports: float, int8, uint8, binary, ubinary
        if embedding_types := settings.get('bedrock_embedding_types'):
            body['embedding_types'] = embedding_types

        return body

    def parse_response(
        self,
        response_body: dict[str, Any],
        inputs: list[str],
        input_type: EmbedInputType,
        model_name: str,
        provider_name: str,
        provider_url: str,
        input_tokens: int,
    ) -> EmbeddingResult:
        # Cohere returns embeddings in different formats based on embedding_types
        # Default is float embeddings (when embedding_types not specified)
        # When embedding_types is specified, returns a dict with keys like 'float', 'int8', etc.
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

        return EmbeddingResult(
            embeddings=embeddings,
            inputs=inputs,
            input_type=input_type,
            usage=RequestUsage(input_tokens=input_tokens),
            model_name=model_name,
            provider_name=provider_name,
            provider_response_id=response_body.get('id'),
        )


class NovaEmbeddingHandler(BedrockEmbeddingHandler):
    """Handler for Amazon Nova embedding models on Bedrock."""

    @property
    def supports_batch(self) -> bool:
        return False  # Nova only supports single text per request

    def prepare_request(
        self,
        texts: list[str],
        input_type: EmbedInputType,
        settings: BedrockEmbeddingSettings,
    ) -> dict[str, Any]:
        # Nova uses a task-based format for embeddings
        # Only supports single text per request, caller must handle batching

        # Get dimensions with meaningful default (1024 is a good balance of quality and size)
        dimensions = settings.get('dimensions', 1024)

        # Get embedding purpose with default based on input_type
        # - queries default to GENERIC_RETRIEVAL (optimized for search)
        # - documents default to GENERIC_INDEX (optimized for indexing)
        default_purpose = 'GENERIC_RETRIEVAL' if input_type == 'query' else 'GENERIC_INDEX'
        embedding_purpose = settings.get('bedrock_embedding_purpose', default_purpose)

        # Get truncation mode with default to END (truncate end of text if too long)
        truncate = settings.get('bedrock_truncate', 'END')
        # Normalize truncation values for Nova (accepts START, END, NONE)
        if truncate == 'LEFT':
            truncate = 'START'
        elif truncate == 'RIGHT':
            truncate = 'END'

        body: dict[str, Any] = {
            'taskType': 'SINGLE_EMBEDDING',
            'singleEmbeddingParams': {
                'embeddingPurpose': embedding_purpose,
                'embeddingDimension': dimensions,
                'text': {
                    'truncationMode': truncate,
                    'value': texts[0],
                },
            },
        }

        return body

    def parse_response(
        self,
        response_body: dict[str, Any],
        inputs: list[str],
        input_type: EmbedInputType,
        model_name: str,
        provider_name: str,
        provider_url: str,
        input_tokens: int,
    ) -> EmbeddingResult:
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

        return EmbeddingResult(
            embeddings=[embedding],
            inputs=inputs,
            input_type=input_type,
            usage=RequestUsage(input_tokens=input_tokens),
            model_name=model_name,
            provider_name=provider_name,
        )


def _get_handler_for_model(model_name: str) -> BedrockEmbeddingHandler:
    """Get the appropriate handler for a Bedrock embedding model."""
    # Remove regional prefix if present
    normalized_name = model_name
    for prefix in BEDROCK_GEO_PREFIXES:
        if normalized_name.startswith(f'{prefix}.'):
            normalized_name = normalized_name.removeprefix(f'{prefix}.')
            break

    if normalized_name.startswith('amazon.titan-embed'):
        return TitanEmbeddingHandler()
    elif normalized_name.startswith('cohere.embed'):
        return CohereEmbeddingHandler()
    elif normalized_name.startswith('amazon.nova'):
        return NovaEmbeddingHandler()
    else:
        raise UserError(
            f'Unsupported Bedrock embedding model: {model_name}. '
            f'Supported models: Amazon Titan Embed (amazon.titan-embed-*), '
            f'Cohere Embed (cohere.embed-*), Amazon Nova (amazon.nova-*)'
        )


# Maximum input tokens for known models (keys are normalized without version suffix)
_MAX_INPUT_TOKENS: dict[str, int] = {
    'amazon.titan-embed-text-v1': 8192,
    'amazon.titan-embed-text-v2': 8192,
    'cohere.embed-english-v3': 512,
    'cohere.embed-multilingual-v3': 512,
    'cohere.embed-v4': 128000,
    'amazon.nova-2-multimodal-embeddings-v1': 8192,  # Per AWS documentation
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

        if self._handler.supports_batch:
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

        return self._handler.parse_response(
            response,
            inputs,
            input_type,
            self.model_name,
            self.system,
            self.base_url,
            input_tokens,
        )

    async def _embed_sequential(
        self,
        inputs: list[str],
        input_type: EmbedInputType,
        settings: BedrockEmbeddingSettings,
    ) -> EmbeddingResult:
        """Embed inputs one at a time and combine results."""
        all_embeddings: list[Sequence[float]] = []
        total_input_tokens = 0

        for text in inputs:
            body = self._handler.prepare_request([text], input_type, settings)
            response, input_tokens = await self._invoke_model(body)

            result = self._handler.parse_response(
                response,
                [text],
                input_type,
                self.model_name,
                self.system,
                self.base_url,
                input_tokens,
            )
            all_embeddings.extend(result.embeddings)
            total_input_tokens += input_tokens

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
            from pydantic_ai.exceptions import ModelAPIError

            raise ModelAPIError(model_name=self.model_name, message=str(e)) from e

        # Extract input token count from HTTP headers
        input_tokens = int(
            response.get('ResponseMetadata', {}).get('HTTPHeaders', {}).get('x-amzn-bedrock-input-token-count', '0')
        )

        response_body = json.loads(response['body'].read())
        return response_body, input_tokens

    async def max_input_tokens(self) -> int | None:
        """Get the maximum number of tokens that can be input to the model."""
        # Normalize model name by removing version suffix
        normalized = self._normalize_model_name(self._model_name)
        return _MAX_INPUT_TOKENS.get(normalized)

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """Normalize model name by removing regional prefix and version suffix."""
        # Remove regional prefix
        for prefix in BEDROCK_GEO_PREFIXES:
            if model_name.startswith(f'{prefix}.'):
                model_name = model_name.removeprefix(f'{prefix}.')
                break

        # Remove version suffix like :0
        version_match = re.match(r'(.+?)(?::\d+)?$', model_name)
        if version_match:
            return version_match.group(1)
        return model_name
