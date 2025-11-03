"""OpenAI protocol integration for Pydantic AI agents."""

from ._adapter import ChatCompletionsAdapter
from ._event_stream import ChatCompletionsEventStream

__all__ = [
    'ChatCompletionsAdapter',
    'ChatCompletionsEventStream',
]

