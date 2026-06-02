"""Card 49: `StreamedResponse.usage()` method-to-property migration.

In 1.x, accessing `streamed_response.usage` as a method (with parentheses) emits a
`PydanticAIDeprecationWarning`. Attribute-style access (no parentheses) is the new
contract and does not warn. `StreamedResponse` is the model-level streaming response
base class in `pydantic_ai.models`, missed by card 01's result-class audit.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from pydantic_ai import ModelRequest, UserPromptPart
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.direct import model_request_stream
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

pytestmark = pytest.mark.anyio


def _assert_no_deprecation(getter: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticAIDeprecationWarning)
        return getter()


async def test_streamed_response_usage_property_then_call():
    """`StreamedResponse.usage` — property access silent, method call warns; both yield a `RequestUsage`."""
    messages = [ModelRequest(parts=[UserPromptPart(content='hello')])]
    async with model_request_stream(TestModel(), messages) as stream:
        async for _ in stream:
            pass

        usage_attr = _assert_no_deprecation(lambda: stream.usage)
        assert isinstance(usage_attr, RequestUsage)

        with pytest.warns(PydanticAIDeprecationWarning, match=r'`StreamedResponse\.usage` is no longer a method'):
            usage_call = stream.usage()
        assert isinstance(usage_call, RequestUsage)
        assert usage_attr == usage_call

        # `__eq__`/`__repr__` make the wrapper indistinguishable from the underlying type.
        assert repr(usage_call).startswith('RequestUsage(')
        assert (usage_call == 'not-a-request-usage') is False
