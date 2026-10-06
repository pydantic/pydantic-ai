"""Map a provider response body the SDK could not decode to `ModelAPIError`.

Some gateways answer 200 with a body that isn't JSON (e.g. keep-alive whitespace before an upstream failure). Most
provider SDKs then raise their JSON decoder's error, which isn't a `ModelAPIError`, so `FallbackModel` wouldn't
fall back.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterable, Generator
from contextlib import contextmanager
from typing import Generic, Self, TypeVar

from ..exceptions import ModelAPIError

_DECODE_ERRORS: tuple[type[Exception], ...] = (json.JSONDecodeError, UnicodeDecodeError)


@contextmanager
def map_decode_errors(model_name: str, *error_types: type[Exception]) -> Generator[None]:
    """Map a response body the SDK could not decode as JSON to `ModelAPIError`.

    Wrap only the SDK's own work: our processing of a response parses JSON too, and those errors stay unmapped.

    Args:
        model_name: The name of the model the request was made to.
        error_types: SDK-specific exceptions for an undecodable body, in addition to the JSON decoder's own.
    """
    try:
        yield
    except (*_DECODE_ERRORS, *error_types) as e:
        raise ModelAPIError(model_name=model_name, message=f'Failed to decode response as JSON: {e}') from e


_ChunkT = TypeVar('_ChunkT')


class MapStreamDecodeErrors(Generic[_ChunkT]):
    """Apply `map_decode_errors` to the SDK decoding each chunk, but not to the code consuming it.

    A plain iterator rather than an async generator, so it adds no generator for the event loop to finalize when a stream
    is abandoned.
    """

    def __init__(self, stream: AsyncIterable[_ChunkT], model_name: str, *error_types: type[Exception]):
        self._iterator = aiter(stream)
        self._model_name = model_name
        self._error_types = error_types

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> _ChunkT:
        with map_decode_errors(self._model_name, *self._error_types):
            return await anext(self._iterator)
