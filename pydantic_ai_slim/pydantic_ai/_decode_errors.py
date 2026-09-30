"""Map response bodies an SDK cannot decode as JSON to `ModelAPIError`.

Extracted from the OpenAI adapters (originally introduced there) so every provider-facing
adapter can reuse the same mapping. See https://github.com/pydantic/pydantic-ai/issues/9340.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, Generator
from contextlib import contextmanager
from typing import Generic, TypeVar

from typing_extensions import Self

from .exceptions import ModelAPIError

_ChunkT = TypeVar('_ChunkT')


@contextmanager
def _map_decode_errors(model_name: str, *more_decode_errors: type[Exception]) -> Generator[None]:
    """Map a response body the SDK could not decode as JSON to `ModelAPIError`.

    Wrap only the SDK's own work: our processing of a response parses JSON too, and those errors stay unmapped.
    `more_decode_errors` adds SDK-specific decode failures, e.g. google-genai's `UnknownApiResponseError`.
    """
    try:
        yield
    except (json.JSONDecodeError, UnicodeDecodeError, *more_decode_errors) as e:
        raise ModelAPIError(model_name=model_name, message=f'Failed to decode response as JSON: {e}') from e


class _MapStreamDecodeErrors(Generic[_ChunkT]):
    """Apply `_map_decode_errors` to the SDK decoding each chunk, but not to the code consuming it.

    A plain iterator rather than an async generator, so it adds no generator for the event loop to finalize when a stream
    is abandoned.
    """

    def __init__(self, stream: AsyncIterable[_ChunkT], model_name: str, *more_decode_errors: type[Exception]):
        self._iterator = aiter(stream)
        self._model_name = model_name
        self._more_decode_errors = more_decode_errors

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> _ChunkT:
        with _map_decode_errors(self._model_name, *self._more_decode_errors):
            return await anext(self._iterator)
