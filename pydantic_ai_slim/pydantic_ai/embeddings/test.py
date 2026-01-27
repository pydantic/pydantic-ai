import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai.usage import RequestUsage

from .base import EmbeddingModel, EmbedInputType
from .result import EmbeddingResult
from .settings import EmbeddingSettings

# Regex for splitting text into approximate tokens (matches FunctionModel approach)
_TOKEN_SPLIT_RE = re.compile(r'[\s",.:]+')


def _estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    This is a rough approximation that splits on whitespace and punctuation,
    matching the approach used by FunctionModel.
    """
    if not text:
        return 0  # pragma: no cover
    return len(_TOKEN_SPLIT_RE.split(text.strip()))


@dataclass(init=False)
class TestEmbeddingModel(EmbeddingModel):
    """A mock embedding model for testing.

    This model returns deterministic embeddings (all 1.0 values) and tracks
    the settings used in the last call via the `last_settings` attribute.

    Example:
    ```python
    from pydantic_ai import Embedder
    from pydantic_ai.embeddings import TestEmbeddingModel

    test_model = TestEmbeddingModel()
    embedder = Embedder('openai:text-embedding-3-small')


    async def main():
        with embedder.override(model=test_model):
            await embedder.embed_query('test')
            assert test_model.last_settings is not None
    ```
    """

    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    _model_name: str
    """The model name to report in results."""

    _provider_name: str
    """The provider name to report in results."""

    _dimensions: int
    """The number of dimensions for generated embeddings."""

    last_settings: EmbeddingSettings | None = None
    """The settings used in the most recent embed call."""

    def __init__(
        self,
        model_name: str = 'test',
        *,
        provider_name: str = 'test',
        dimensions: int = 8,
        settings: EmbeddingSettings | None = None,
    ):
        """Initialize the test embedding model.

        Args:
            model_name: The model name to report in results.
            provider_name: The provider name to report in results.
            dimensions: The number of dimensions for the generated embeddings.
            settings: Optional default settings for the model.
        """
        self._model_name = model_name
        self._provider_name = provider_name
        self._dimensions = dimensions
        self.last_settings = None
        super().__init__(settings=settings)

    @property
    def model_name(self) -> str:
        """The embedding model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The embedding model provider."""
        return self._provider_name

    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        inputs, settings = self.prepare_embed(inputs, settings)
        self.last_settings = settings

        dimensions = settings.get('dimensions') or self._dimensions

        return EmbeddingResult(
            embeddings=[[1.0] * dimensions] * len(inputs),
            inputs=inputs,
            input_type=input_type,
            usage=RequestUsage(input_tokens=sum(_estimate_tokens(text) for text in inputs)),
            model_name=self.model_name,
            provider_name=self.system,
            provider_response_id=str(uuid.uuid4()),
        )

    async def max_input_tokens(self) -> int | None:
        return 1024

    async def count_tokens(self, text: str) -> int:
        return _estimate_tokens(text)
