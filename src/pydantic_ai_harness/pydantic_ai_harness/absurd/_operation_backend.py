from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from typing import TypeGuard

from pydantic_ai._utils import is_str_dict
from pydantic_ai.durable_exec import (
    CallableOperationBackend,
    DurableOperationId,
    JournalOperationNamer,
    RoleBasedOperationConfig,
    ToolsetCallToolId,
    ToolsetGetInstructionsId,
    ToolsetGetToolsId,
    ToolsetValidateToolArgumentsId,
)

from ._context import current_async_task_context

_ToolsetOperationId = ToolsetGetToolsId | ToolsetGetInstructionsId | ToolsetValidateToolArgumentsId | ToolsetCallToolId

# A `pydantic-ai-absurd` tool-call checkpoint holds the tool's raw return value, and a control-flow
# exception (`ModelRetry`, `CallDeferred`, ...) propagates out of the step without a checkpoint.
# This backend writes the same shape, so a run started under `pydantic-ai-absurd` resumes here.
# The one result that has no raw form is a `ToolReturn` object, which goes under this reserved key.
_ENVELOPE_KEY = '__pydantic_ai_harness_absurd_tool_result__'
_RAW_RESULT_KINDS = frozenset({'tool_return', 'tool_content_result'})

# Absurd steps take no per-operation options.
_NO_CONFIG = RoleBasedOperationConfig[None](model=None, event=None, capability=None, tool=None)


class AbsurdOperationNamer(JournalOperationNamer):
    """Journal naming, plus `pydantic-ai-absurd`'s names for toolsets without an `id`.

    `pydantic-ai-absurd` named an id-less toolset's steps without the `__<id>` segment
    (`agent__mcp_server.call_tool`), so those names are kept for checkpoints to line up.
    """

    def operation_name(self, operation_id: DurableOperationId) -> str:
        # The base passes the toolset's `id` through unchanged, so an id-less toolset arrives without one.
        if isinstance(operation_id, _ToolsetOperationId) and not operation_id.toolset_id:
            placeholder = dataclasses.replace(operation_id, toolset_id='\0')
            return super().operation_name(placeholder).replace('__\0', '', 1)
        return super().operation_name(operation_id)


class _UncheckpointedResult(Exception):
    def __init__(self, payload: object) -> None:
        self.payload = payload


def _is_tool_return_object(value: object) -> bool:
    return is_str_dict(value) and value.get('kind') == 'tool-return'


def _is_envelope(stored: object) -> TypeGuard[dict[str, object]]:
    if not (is_str_dict(stored) and stored.keys() == {_ENVELOPE_KEY}):
        return False
    payload = stored[_ENVELOPE_KEY]
    return (
        is_str_dict(payload) and payload.get('kind') == 'tool_return' and _is_tool_return_object(payload.get('result'))
    )


def _to_checkpoint(payload: object) -> object:
    """Reduce an encoded `CallToolResult` to what `pydantic-ai-absurd` would have stored."""
    assert is_str_dict(payload)
    kind = payload.get('kind')
    if kind not in _RAW_RESULT_KINDS:
        # Control flow: raising keeps it out of the checkpoint table, as in `pydantic-ai-absurd`.
        raise _UncheckpointedResult(payload)
    result = payload['result']
    if kind == 'tool_return' and _is_tool_return_object(result):
        return {_ENVELOPE_KEY: payload}
    return result


def _from_checkpoint(stored: object) -> object:
    """Rebuild the encoded `CallToolResult` for a stored checkpoint, raw or enveloped."""
    if _is_envelope(stored):
        return stored[_ENVELOPE_KEY]
    if _is_tool_return_object(stored):
        # A raw dict whose `kind` would otherwise be decoded as a `ToolReturn`.
        return {'kind': 'tool_content_result', 'result': stored}
    return {'kind': 'tool_return', 'result': stored}


class AbsurdOperationBackend(CallableOperationBackend[None]):
    def __init__(self, *, agent_name: str, default_model_id: str | None) -> None:
        super().__init__(
            namer=AbsurdOperationNamer(agent_name, default_model_id=default_model_id or 'default'), config=_NO_CONFIG
        )

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: None,
    ) -> object:
        del cache_key, config
        task_ctx = current_async_task_context()
        assert task_ctx is not None
        if not isinstance(operation_id, ToolsetCallToolId):
            return await task_ctx.step(name, body)

        async def checkpointed_body() -> object:
            return _to_checkpoint(await body())

        try:
            stored = await task_ctx.step(name, checkpointed_body)
        except _UncheckpointedResult as uncheckpointed:
            return uncheckpointed.payload
        return _from_checkpoint(stored)
