from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from pydantic_ai import _utils
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.usage import RequestUsage

from . import OpenAIEmbeddingsCompatibleProvider
from .base import EmbeddingModel, EmbedInputType
from .result import EmbeddingResult
from .settings import EmbeddingSettings

try:
    import tiktoken
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI
    from openai.types import EmbeddingModel as LatestOpenAIEmbeddingModelNames
    from openai.types.create_embedding_response import Usage

    from pydantic_ai.models.openai import OMIT
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI embeddings model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

OpenAIEmbeddingModelName = str | LatestOpenAIEmbeddingModelNames
"""Possible OpenAI embeddings model names.

See the [OpenAI embeddings documentation](https://platform.openai.com/docs/guides/embeddings)
for available models.
"""


class OpenAIEmbeddingSettings(EmbeddingSettings, total=False):
    """Settings used for an OpenAI embedding model request.

    All fields from [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings] are supported.
    """

    # ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.


@dataclass(init=False)
class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model implementation.

    This model works with OpenAI's embeddings API and any
    [OpenAI-compatible providers](../models/openai.md#openai-compatible-models).

    Example:
    ```python
    from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
    from pydantic_ai.providers.openai import OpenAIProvider

    # Using OpenAI directly
    model = OpenAIEmbeddingModel('text-embedding-3-small')

    # Using an OpenAI-compatible provider
    model = OpenAIEmbeddingModel(
        'text-embedding-3-small',
        provider=OpenAIProvider(base_url='https://my-provider.com/v1'),
    )
    ```
    """

    _model_name: OpenAIEmbeddingModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    def __init__(
        self,
        model_name: OpenAIEmbeddingModelName,
        *,
        provider: OpenAIEmbeddingsCompatibleProvider | Literal['openai'] | Provider[AsyncOpenAI] = 'openai',
        settings: EmbeddingSettings | None = None,
    ):
        """Initialize an OpenAI embedding model.

        Args:
            model_name: The name of the OpenAI model to use.
                See [OpenAI's embedding models](https://platform.openai.com/docs/guides/embeddings)
                for available options.
            provider: The provider to use for authentication and API access. Can be:

                - `'openai'` (default): Uses the standard OpenAI API
                - A provider name string (e.g., `'azure'`, `'deepseek'`)
                - A [`Provider`][pydantic_ai.providers.Provider] instance for custom configuration

                See [OpenAI-compatible providers](../models/openai.md#openai-compatible-models)
                for a list of supported providers.
            settings: Model-specific [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings]
                to use as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        self._client = provider.client

        super().__init__(settings=settings)

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    @property
    def model_name(self) -> OpenAIEmbeddingModelName:
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
        settings = cast(OpenAIEmbeddingSettings, settings)

        try:
            response = await self._client.embeddings.create(
                input=inputs,
                model=self.model_name,
                dimensions=settings.get('dimensions') or OMIT,
                extra_headers=settings.get('extra_headers'),
                extra_body=settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover
        except APIConnectionError as e:  # pragma: no cover
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

        embeddings = [item.embedding for item in response.data]

        return EmbeddingResult(
            embeddings=embeddings,
            inputs=inputs,
            input_type=input_type,
            usage=_map_usage(response.usage, self.system, self.base_url, response.model),
            model_name=response.model,
            provider_name=self.system,
        )

    async def max_input_tokens(self) -> int | None:
        if self.system != 'openai':
            return None

        # https://platform.openai.com/docs/guides/embeddings#embedding-models
        return 8192

    async def count_tokens(self, text: str) -> int:
        if self.system != 'openai':
            raise UserError(
                'Counting tokens is not supported for non-OpenAI embedding models',
            )
        try:
            encoding = await _utils.run_in_executor(tiktoken.encoding_for_model, self.model_name)
        except KeyError as e:  # pragma: no cover
            raise ValueError(
                f'The embedding model {self.model_name!r} is not supported by tiktoken',
            ) from e
        return len(encoding.encode(text))


def _map_usage(
    usage: Usage,
    provider: str,
    provider_url: str,
    model: str,
) -> RequestUsage:
    usage_data = usage.model_dump(exclude_none=True)
    details = {k: v for k, v in usage_data.items() if k not in {'prompt_tokens', 'total_tokens'} if isinstance(v, int)}
    response_data = dict(model=model, usage=usage_data)

    return RequestUsage.extract(
        response_data,
        provider=provider,
        provider_url=provider_url,
        provider_fallback='openai',
        api_flavor='embeddings',
        details=details,
    )
