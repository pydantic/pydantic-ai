from __future__ import annotations

from typing import Any, cast

import pytest

from pydantic_ai._agent_graph import ModelRequestNode
from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.messages import ModelRequest

pytestmark = pytest.mark.anyio


async def test_model_request_node_run_raises_if_streaming_not_finished() -> None:
    node = cast(Any, ModelRequestNode(request=ModelRequest(parts=[])))
    setattr(node, '_did_stream', True)
    with pytest.raises(AgentRunError, match='finish streaming before calling run'):
        await node.run(cast(Any, None))


async def test_model_request_node_make_request_returns_cached_result() -> None:
    node = cast(Any, ModelRequestNode(request=ModelRequest(parts=[])))
    cached_result = cast(Any, object())
    setattr(node, '_result', cached_result)
    make_request = getattr(node, '_make_request')
    assert await make_request(cast(Any, None)) is cached_result
