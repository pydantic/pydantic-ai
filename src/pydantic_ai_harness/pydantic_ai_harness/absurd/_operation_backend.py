from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeGuard

from pydantic_ai._utils import is_str_dict
from pydantic_ai.durable_exec import (
    DurableOperationId,
    JournalCallableOperationBackend,
    RoleBasedOperationConfig,
    ToolsetCallToolId,
)

from ._context import current_async_task_context

# A tool-call checkpoint holds the raw return value; a `ToolReturn` has no raw form, so it goes under this key.
_ENVELOPE_KEY = '__pydantic_ai_harness_absurd_tool_result__'
_RAW_RESULT_KINDS = frozenset({'tool_return', 'tool_content_result'})
# Control flow (`ModelRetry`, `CallDeferred`, ...) is not checkpointed, so the call runs again on replay.
_CONTROL_FLOW_KINDS = frozenset(
    {'approval_required', 'call_deferred', 'model_retry', 'validation_error', 'tool_failed'}
)

# Absurd steps take no per-operation options.
_NO_CONFIG = RoleBasedOperationConfig[None](model=None, event=None, capability=None, tool=None)


def _is_tool_return_object(value: object) -> bool:
    return is_str_dict(value) and value.get('kind') == 'tool-return'


def _is_envelope(stored: object) -> TypeGuard[dict[str, object]]:
    if not (is_str_dict(stored) and stored.keys() == {_ENVELOPE_KEY}):
        return False
    payload = stored[_ENVELOPE_KEY]
    return is_str_dict(payload) and payload.get('kind') in _RAW_RESULT_KINDS and 'result' in payload


def _to_checkpoint(payload: dict[str, object]) -> object:
    """Reduce an encoded `CallToolResult` to the stored checkpoint: the raw return value."""
    result = payload['result']
    # A raw value shaped like the envelope is enveloped too, so it reads back as itself.
    if _is_envelope(result) or (payload['kind'] == 'tool_return' and _is_tool_return_object(result)):
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


class AbsurdOperationBackend(JournalCallableOperationBackend[None]):
    def __init__(self, *, agent_name: str, default_model_id: str | None) -> None:
        super().__init__(agent_name=agent_name, default_model_id=default_model_id, config=_NO_CONFIG)

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

        handle = await task_ctx.begin_step(name)
        if handle.done:
            return _from_checkpoint(handle.state)
        payload = await body()
        assert is_str_dict(payload)
        kind = payload.get('kind')
        if kind in _CONTROL_FLOW_KINDS:
            return payload
        # Fail on a result kind this backend doesn't know, rather than silently not checkpointing it.
        assert kind in _RAW_RESULT_KINDS, kind
        return _from_checkpoint(await task_ctx.complete_step(handle, _to_checkpoint(payload)))
