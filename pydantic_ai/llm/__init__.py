"""Logic related to making requests to an LLM.

The aim here is to make a common interface
"""

from abc import ABC, abstractmethod
from typing import Literal

from ..result import LLMMessage, Message


class Model(ABC):
    """Abstract class for a model."""

    @abstractmethod
    async def request(self, messages: list[Message]) -> LLMMessage:
        """Request a response from the model."""
        raise NotImplementedError()

    # TODO streamed response


ModelName = Literal['openai-gpt-4o', 'openai-gpt-4-turbo', 'openai-gpt-4', 'openai-gpt-3.5-turbo']


def infer_model(model: ModelName | Model) -> Model:
    """Infer the model from the name."""
    if isinstance(model, Model):
        return model
    elif model.startswith('openai-'):
        open_ai_model = model.removeprefix('openai-')

        from .openai import OpenAIModel

        return OpenAIModel(open_ai_model)  # type: ignore[reportArgumentType]
    else:
        raise TypeError(f'Invalid model: {model}')
