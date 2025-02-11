from dataclasses import dataclass
from typing import Any

from ..messages import ModelResponse
from ..usage import Usage
from . import Model


@dataclass
class WrapperModel(Model):
    """Model which wraps another model."""

    wrapped: Model

    async def request(self, *args: Any, **kwargs: Any) -> tuple[ModelResponse, Usage]:
        return await self.wrapped.request(*args, **kwargs)

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)
