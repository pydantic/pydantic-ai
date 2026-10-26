from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime
from typing import Any, Literal

from genai_prices import calc_price, types as genai_types

from pydantic_ai._utils import now_utc as _now_utc
from pydantic_ai.usage import RequestUsage

EmbedInputType = Literal['query', 'document']
"""The type of input to the embedding model.

- `'query'`: Text that will be used as a search query
- `'document'`: Text that will be stored and searched against

Some embedding models optimize differently for queries vs documents.
"""


@dataclass
class EmbeddingResult:
    """The result of an embedding operation.

    This class contains the generated embeddings along with metadata about
    the operation, including the original inputs, model information, usage
    statistics, and timing.

    Example:
    ```python
    from pydantic_ai import Embedder

    embedder = Embedder('openai:text-embedding-3-small')


    async def main():
        result = await embedder.embed_query('What is AI?')

        # Access embeddings by index
        print(len(result.embeddings[0]))
        #> 1536

        # Access embeddings by original input text
        print(result['What is AI?'] == result.embeddings[0])
        #> True

        # Check usage
        print(f'Tokens used: {result.usage.input_tokens}')
        #> Tokens used: 3
    ```
    """

    embeddings: Sequence[Sequence[float]]
    """The computed embedding vectors, one per input text.

    Each embedding is a sequence of floats representing the text in vector space.
    """

    _: KW_ONLY

    inputs: Sequence[str]
    """The original input texts that were embedded."""

    input_type: EmbedInputType
    """Whether the inputs were embedded as queries or documents."""

    model_name: str
    """The name of the model that generated these embeddings."""

    provider_name: str
    """The name of the provider (e.g., 'openai', 'cohere')."""

    timestamp: datetime = field(default_factory=_now_utc)
    """When the embedding request was made."""

    usage: RequestUsage = field(default_factory=RequestUsage)
    """Token usage statistics for this request."""

    provider_details: dict[str, Any] | None = None
    """Provider-specific details from the response."""

    provider_response_id: str | None = None
    """Unique identifier for this response from the provider, if available."""

    def __getitem__(self, item: int | str) -> Sequence[float]:
        """Get the embedding for an input by index or by the original input text.

        Args:
            item: Either an integer index or the original input string.

        Returns:
            The embedding vector for the specified input.

        Raises:
            IndexError: If the index is out of range.
            ValueError: If the string is not found in the inputs.
        """
        if isinstance(item, str):
            item = self.inputs.index(item)

        return self.embeddings[item]

    def cost(self) -> genai_types.PriceCalculation:
        """Calculate the cost of the embedding request.

        Uses [`genai-prices`](https://github.com/pydantic/genai-prices) for pricing data.

        Returns:
            A price calculation object with `total_price`, `input_price`, and other cost details.

        Raises:
            LookupError: If pricing data is not available for this model/provider.
        """
        assert self.model_name, 'Model name is required to calculate price'
        return calc_price(
            self.usage,
            self.model_name,
            provider_id=self.provider_name,
            genai_request_timestamp=self.timestamp,
        )
