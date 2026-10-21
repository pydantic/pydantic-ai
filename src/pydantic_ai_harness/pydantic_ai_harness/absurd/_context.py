from absurd_sdk import AsyncTaskContext, get_current_context

from pydantic_ai.exceptions import UserError

ENGINE_NAME = 'Absurd'


def current_async_task_context() -> AsyncTaskContext | None:
    """Return the active Absurd async task context, or `None` outside a task."""
    ctx = get_current_context()
    if ctx is None:
        return None
    if isinstance(ctx, AsyncTaskContext):
        return ctx
    raise UserError(
        f'{ENGINE_NAME} durability requires an async Absurd task context, but a synchronous '
        '`TaskContext` is active. Agent runs are async, so run the agent from an async task '
        'handler (`AsyncTaskContext`), not a synchronous one.'
    )
