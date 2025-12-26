from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.usage import RequestUsage

from .base import EmbeddingModel, EmbedInputType
from .result import EmbeddingResult
from .settings import EmbeddingSettings

try:
    from cohere import AsyncClientV2
    from cohere.core.api_error import ApiError
    from cohere.core.request_options import RequestOptions
    from cohere.types.embed_by_type_response import EmbedByTypeResponse
    from cohere.types.embed_input_type import EmbedInputType as CohereEmbedInputType
    from cohere.v2.types.v2embed_request_truncate import V2EmbedRequestTruncate

    from pydantic_ai.providers.cohere import CohereProvider
except ImportError as _import_error:
    raise ImportError(
        'Please install `cohere` to use the Cohere embeddings model, '
        'you can use the `cohere` optional group — `pip install "pydantic-ai-slim[cohere]"`'
    ) from _import_error

LatestCohereEmbeddingModelNames = Literal[
    'embed-v4.0',
    'embed-english-v3.0',
    'embed-english-light-v3.0',
    'embed-multilingual-v3.0',
    'embed-multilingual-light-v3.0',
]
"""Latest Cohere embeddings models.

See the [Cohere Embed documentation](https://docs.cohere.com/docs/cohere-embed)
for available models and their capabilities.
"""

CohereEmbeddingModelName = str | LatestCohereEmbeddingModelNames
"""Possible Cohere embeddings model names."""

# Taken from https://docs.cohere.com/docs/cohere-embed
_MAX_INPUT_TOKENS: dict[CohereEmbeddingModelName, int] = {
    'embed-v4.0': 128000,
    'embed-english-v3.0': 512,
    'embed-english-light-v3.0': 512,
    'embed-multilingual-v3.0': 512,
    'embed-multilingual-light-v3.0': 512,
}


class CohereEmbeddingSettings(EmbeddingSettings, total=False):
    """Settings used for a Cohere embedding model request.

    All fields from [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings] are supported,
    plus Cohere-specific settings prefixed with `cohere_`.
    """

    # ALL FIELDS MUST BE `cohere_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    cohere_max_tokens: int
    """The maximum number of tokens to embed."""

    cohere_input_type: CohereEmbedInputType
    """The Cohere-specific input type for the embedding.

    Overrides the standard `input_type` argument. Options include:
    `'search_query'`, `'search_document'`, `'classification'`, `'clustering'`, and `'image'`.
    """

    cohere_truncate: V2EmbedRequestTruncate
    """The truncation strategy to use:

    - `'NONE'` (default): Raise an error if input exceeds max tokens.
    - `'END'`: Truncate the end of the input text.
    - `'START'`: Truncate the start of the input text.
    """


@dataclass(init=False)
class CohereEmbeddingModel(EmbeddingModel):
    """Cohere embedding model implementation.

    This model works with Cohere's embeddings API, which offers
    multilingual support and various model sizes.

    Example:
    ```python
    from pydantic_ai.embeddings.cohere import CohereEmbeddingModel

    model = CohereEmbeddingModel('embed-v4.0')
    ```
    """

    _model_name: CohereEmbeddingModelName = field(repr=False)
    _provider: Provider[AsyncClientV2] = field(repr=False)

    def __init__(
        self,
        model_name: CohereEmbeddingModelName,
        *,
        provider: Literal['cohere'] | Provider[AsyncClientV2] = 'cohere',
        settings: EmbeddingSettings | None = None,
    ):
        """Initialize a Cohere embedding model.

        Args:
            model_name: The name of the Cohere model to use.
                See [Cohere Embed documentation](https://docs.cohere.com/docs/cohere-embed)
                for available models.
            provider: The provider to use for authentication and API access. Can be:

                - `'cohere'` (default): Uses the standard Cohere API
                - A [`CohereProvider`][pydantic_ai.providers.cohere.CohereProvider] instance
                  for custom configuration
            settings: Model-specific [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings]
                to use as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        self._client = provider.client
        self._v1_client = provider.v1_client if isinstance(provider, CohereProvider) else None

        super().__init__(settings=settings)

    @property
    def base_url(self) -> str:
        """The base URL for the provider API, if available."""
        return self._provider.base_url

    @property
    def model_name(self) -> CohereEmbeddingModelName:
        """The embedding model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The embedding model provider."""
        return self._provider.name

    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        inputs, settings = self.prepare_embed(inputs, settings)
        settings = cast(CohereEmbeddingSettings, settings)

        cohere_input_type = settings.get(
            'cohere_input_type', 'search_document' if input_type == 'document' else 'search_query'
        )

        request_options: RequestOptions = {}
        if extra_headers := settings.get('extra_headers'):  # pragma: no cover
            request_options['additional_headers'] = extra_headers
        if extra_body := settings.get('extra_body'):  # pragma: no cover
            request_options['additional_body_parameters'] = cast(dict[str, Any], extra_body)

        try:
            response = await self._client.embed(
                model=self.model_name,
                texts=inputs,
                output_dimension=settings.get('dimensions'),
                input_type=cohere_input_type,
                max_tokens=settings.get('cohere_max_tokens'),
                truncate=settings.get('cohere_truncate', 'NONE'),
                request_options=request_options,
            )
        except ApiError as e:
            if (status_code := e.status_code) and status_code >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise ModelAPIError(model_name=self.model_name, message=str(e)) from e  # pragma: no cover

        embeddings = response.embeddings.float_
        if embeddings is None:
            raise UnexpectedModelBehavior(  # pragma: no cover
                'The Cohere embeddings response did not have an `embeddings` field holding a list of floats',
                str(response),
            )

        return EmbeddingResult(
            embeddings=embeddings,
            inputs=inputs,
            input_type=input_type,
            usage=_map_usage(response, self.system, self.base_url, self.model_name),
            model_name=self.model_name,
            provider_name=self.system,
            provider_response_id=response.id,
        )

    async def max_input_tokens(self) -> int | None:
        return _MAX_INPUT_TOKENS.get(self.model_name)

    async def count_tokens(self, text: str) -> int:
        if self._v1_client is None:
            raise NotImplementedError('Counting tokens requires the Cohere v1 client')
        try:
            result = await self._v1_client.tokenize(
                model=self.model_name,
                text=text,  # Has a max length of 65536 characters
                offline=False,
            )
        except ApiError as e:  # pragma: no cover
            if (status_code := e.status_code) and status_code >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise ModelAPIError(model_name=self.model_name, message=str(e)) from e

        return len(result.tokens)


def _map_usage(response: EmbedByTypeResponse, provider: str, provider_url: str, model: str) -> RequestUsage:
    u = response.meta
    if u is None or u.billed_units is None:
        return RequestUsage()  # pragma: no cover
    usage_data = {
        k: int(v)
        for k, v in u.billed_units.model_dump(exclude_none=True).items()
        if isinstance(v, int | float) and v > 0
    }
    details = {k: int(v) for k, v in usage_data.items() if k != 'input_tokens' and isinstance(v, int | float) and v > 0}
    response_data = dict(model=model, meta=dict(billed_units=usage_data))

    return RequestUsage.extract(
        response_data,
        provider=provider,
        provider_url=provider_url,
        provider_fallback='cohere',
        api_flavor='embeddings',
        details=details,
    )
