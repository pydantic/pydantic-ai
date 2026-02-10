from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic.errors import PydanticUserError

from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._restate_types import Context, RunOptions, TerminalError
from ._serde import PydanticTypeAdapter


@dataclass
class RestateContextRunResult:
    """Serializable wrapper for tool outcomes used with Restate's `run_typed()`.

    `output` is intentionally `Any`: values are serialized as JSON and replayed as
    JSON-compatible Python types.
    """

    kind: Literal['output', 'call_deferred', 'approval_required', 'model_retry']
    output: Any
    error: str | None = None
    metadata: dict[str, Any] | None = None


CONTEXT_RUN_SERDE = PydanticTypeAdapter(RestateContextRunResult)


def unwrap_context_run_result(res: RestateContextRunResult) -> Any:
    """Convert a durable step result back into tool-control-flow exceptions."""
    if res.kind == 'call_deferred':
        raise CallDeferred(metadata=res.metadata)
    elif res.kind == 'approval_required':
        raise ApprovalRequired(metadata=res.metadata)
    elif res.kind == 'model_retry':
        assert res.error is not None
        raise ModelRetry(res.error)
    else:
        assert res.kind == 'output'
        return res.output


async def wrap_tool_call_result(action: Callable[[], Awaitable[Any]]) -> RestateContextRunResult:
    try:
        output = await action()
        return RestateContextRunResult(kind='output', output=output, error=None)
    except ModelRetry as e:
        return RestateContextRunResult(kind='model_retry', output=None, error=e.message)
    except CallDeferred as e:
        return RestateContextRunResult(kind='call_deferred', output=None, error=None, metadata=e.metadata)
    except ApprovalRequired as e:
        return RestateContextRunResult(kind='approval_required', output=None, error=None, metadata=e.metadata)
    except (UserError, PydanticUserError) as e:
        raise TerminalError(str(e)) from e


class RestateContextRunToolset(WrapperToolset[AgentDepsT]):
    """A toolset that automatically wraps tool calls with restate's `ctx.run_typed()`."""

    def __init__(self, wrapped: AbstractToolset[AgentDepsT], context: Context):
        super().__init__(wrapped)
        self._context = context
        self._options = RunOptions[RestateContextRunResult](serde=CONTEXT_RUN_SERDE)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        async def action() -> Any:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)

        async def action_in_context() -> RestateContextRunResult:
            return await wrap_tool_call_result(action)

        res = await self._context.run_typed(f'Calling {name}', action_in_context, self._options)
        return unwrap_context_run_result(res)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return self
