from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime
from typing import Any, Literal

from genai_prices import calc_price, types as genai_types

from pydantic_ai._utils import now_utc as _now_utc
from pydantic_ai.usage import RequestUsage

EmbedInputType = Literal['query', 'document']
"""The type of input to the embedding model."""


@dataclass
class EmbeddingResult:
    """The result of an embedding operation."""

    embeddings: Sequence[Sequence[float]]

    _: KW_ONLY

    inputs: Sequence[str] | None = None

    input_type: EmbedInputType | None = None

    usage: RequestUsage = field(default_factory=RequestUsage)

    model_name: str | None = None

    timestamp: datetime = field(default_factory=_now_utc)

    provider_name: str | None = None

    provider_details: dict[str, Any] | None = None

    provider_response_id: str | None = None

    # TODO (DouweM): Support `result[idx: int]` and `result[document: str]`

    def cost(self) -> genai_types.PriceCalculation:
        """Calculate the cost of the usage.

        Uses [`genai-prices`](https://github.com/pydantic/genai-prices).
        """
        assert self.model_name, 'Model name is required to calculate price'
        return calc_price(
            self.usage,
            self.model_name,
            provider_id=self.provider_name,
            genai_request_timestamp=self.timestamp,
        )
