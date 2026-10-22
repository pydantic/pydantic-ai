from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeGuard

import anyio
from absurd_sdk import AsyncTaskContext

from pydantic_ai._utils import is_str_dict
from pydantic_ai.durable_exec import (
    DurableOperationId,
    JournalCallableOperationBackend,
    RoleBasedOperationConfig,
    ToolsetCallToolId,
)
from pydantic_ai.exceptions import UserError

from ._context import current_async_task_context

# A tool-call checkpoint holds the raw return value; a `ToolReturn` has no raw form, so it goes under this key.
_ENVELOPE_KEY = '__pydantic_ai_harness_absurd_tool_result__'
_RAW_RESULT_KINDS = frozenset({'tool_return', 'tool_content_result'})
# Control flow (`ModelRetry`, `CallDeferred`, ...) is not checkpointed, so the call runs again on replay.
_CONTROL_FLOW_KINDS = frozenset(
    {'approval_required', 'call_deferred', 'model_retry', 'validation_error', 'tool_failed'}
)
# A tool result that can't be stored is checkpointed as this kind, so a retry fails without re-running the tool.
_UNSERIALIZABLE_KIND = 'unserializable'

# Absurd steps take no per-operation options.
_NO_CONFIG = RoleBasedOperationConfig[None](model=None, event=None, capability=None, tool=None)


class UnserializableStepResultError(UserError):
    """A step's result can't be serialized to JSON for its Absurd checkpoint."""


def serialization_failure(error: Exception) -> UnserializableStepResultError:
    return UnserializableStepResultError(
        f'Absurd stores each step result as JSON, and this one could not be serialized: {error}. '
        'If it is a tool result, return a JSON-serializable value from the tool.'
    )


def _is_tool_return_object(value: object) -> bool:
    return is_str_dict(value) and value.get('kind') == 'tool-return'


def _is_envelope(stored: object) -> TypeGuard[dict[str, object]]:
    if not (is_str_dict(stored) and stored.keys() == {_ENVELOPE_KEY}):
        return False
    payload = stored[_ENVELOPE_KEY]
    if not is_str_dict(payload):
        return False
    kind = payload.get('kind')
    if kind == _UNSERIALIZABLE_KIND:
        return isinstance(payload.get('error'), str)
    return kind in _RAW_RESULT_KINDS and 'result' in payload


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
        payload = stored[_ENVELOPE_KEY]
        assert is_str_dict(payload)
        if payload['kind'] == _UNSERIALIZABLE_KIND:
            raise UnserializableStepResultError(payload['error'])
        return payload
    if _is_tool_return_object(stored):
        # A raw dict whose `kind` would otherwise be decoded as a `ToolReturn`.
        return {'kind': 'tool_content_result', 'result': stored}
    return {'kind': 'tool_return', 'result': stored}


def _claim_timeout(task_ctx: AsyncTaskContext) -> int:
    # The SDK has no public accessor for the worker's claim timeout. Read it directly: a fallback
    # would skip heartbeats for a shorter `claim_timeout` without any error, while a rename in the SDK
    # fails every step with `AttributeError`.
    return task_ctx._claim_timeout  # pyright: ignore[reportPrivateUsage]


def _heartbeat_interval(claim_timeout: int) -> float:
    # Half the claim timeout is the interval the SDK's own long waits use.
    return max(0.5, claim_timeout / 2)


async def _run_holding_claim(task_ctx: AsyncTaskContext, body: Callable[[], Awaitable[object]]) -> object:
    """Run a step's body while extending the run's claim, so it can outlast the worker's `claim_timeout`.

    Without this, a model request or tool call longer than `claim_timeout` lets another worker claim
    the task and run the same step again while this one is still in it. If extending the claim
    fails, the run no longer owns the task (it was cancelled, or failed and retried elsewhere), so
    the body is cancelled and the SDK's error propagates.
    """
    claim_timeout = _claim_timeout(task_ctx)
    if claim_timeout <= 0:
        # A worker with `claim_timeout=0` holds its claim with no expiry, which Absurd refuses to extend.
        return await body()
    interval = _heartbeat_interval(claim_timeout)

    async def heartbeat() -> None:
        # Cancelling a query mid-flight leaves the task's shared connection busy.
        with anyio.CancelScope(shield=True):
            await task_ctx.heartbeat()

    async def extend_claim() -> None:
        while True:
            await anyio.sleep(interval)
            await heartbeat()

    # `begin_step` leaves the claim as it was, so a step starting late in the claim window would lose
    # it before the first heartbeat. Extending it first also stops a lost task before the body runs.
    await heartbeat()

    error: Exception | None = None
    result: object = None
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(extend_claim)
            try:
                result = await body()
            except Exception as exc:
                # Raised below, outside the task group, so it isn't wrapped in an `ExceptionGroup`.
                error = exc
            tg.cancel_scope.cancel()
    except BaseExceptionGroup as group:
        # Usually just `extend_claim`'s failure, or a `BaseException` the body raised.
        if len(group.exceptions) == 1:
            raise group.exceptions[0] from None
        # Both failed at once: raise them together rather than lose either.
        raise
    if error is not None:
        raise error
    return result


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
        handle = await task_ctx.begin_step(name)
        if not isinstance(operation_id, ToolsetCallToolId):
            if handle.done:
                return handle.state
            return await task_ctx.complete_step(handle, await _run_holding_claim(task_ctx, body))

        if handle.done:
            return _from_checkpoint(handle.state)
        try:
            payload = await _run_holding_claim(task_ctx, body)
        except UnserializableStepResultError as exc:
            # The tool already ran: checkpoint the failure so a retry raises it instead of running the tool again.
            await task_ctx.complete_step(handle, {_ENVELOPE_KEY: {'kind': _UNSERIALIZABLE_KIND, 'error': str(exc)}})
            raise
        assert is_str_dict(payload)
        kind = payload.get('kind')
        if kind in _CONTROL_FLOW_KINDS:
            return payload
        # Fail on a result kind this backend doesn't know, rather than silently not checkpointing it.
        assert kind in _RAW_RESULT_KINDS, kind
        return _from_checkpoint(await task_ctx.complete_step(handle, _to_checkpoint(payload)))
