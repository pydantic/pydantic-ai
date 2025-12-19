import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai.usage import RequestUsage

from .base import EmbeddingModel, EmbedInputType
from .result import EmbeddingResult
from .settings import EmbeddingSettings


@dataclass(init=False)
class TestEmbeddingModel(EmbeddingModel):
    """Test embedding model."""

    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    last_settings: EmbeddingSettings | None = None

    def __init__(self, *, settings: EmbeddingSettings | None = None):
        self.last_settings = None
        super().__init__(settings=settings)

    @property
    def model_name(self) -> str:
        """The embedding model name."""
        return 'test'

    @property
    def system(self) -> str:
        """The embedding model provider."""
        return 'test'

    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        inputs, settings = self.prepare_embed(inputs, settings)
        self.last_settings = settings

        return EmbeddingResult(
            embeddings=[[1.0] * 8] * len(inputs),
            inputs=inputs,
            input_type=input_type,
            usage=RequestUsage(input_tokens=sum(len(input) for input in inputs)),
            model_name=self.model_name,
            provider_name=self.system,
            provider_response_id=str(uuid.uuid4()),
        )

    async def max_input_tokens(self) -> int | None:
        return 1024

    async def count_tokens(self, text: str) -> int:
        return len(text)
