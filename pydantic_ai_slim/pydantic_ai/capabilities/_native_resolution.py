from __future__ import annotations

from typing import TypeVar

from pydantic_ai._utils import await_maybe
from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.tools import AgentDepsT, NativeToolFunc, RunContext

_NativeToolT = TypeVar('_NativeToolT', bound=AbstractNativeTool)


async def resolve_native_tool(
    tool_cls: type[_NativeToolT],
    native_tool: AbstractNativeTool | NativeToolFunc[AgentDepsT] | None,
    ctx: RunContext[AgentDepsT],
) -> _NativeToolT:
    """Resolve a native tool instance or factory, raising if it doesn't yield a `tool_cls`.

    A factory returning `None` means "omit this tool" on the native path
    ([`NativeToolFunc`][pydantic_ai.tools.NativeToolFunc]). The fallback subagent cannot omit
    once it has already been invoked, so it raises [`UserError`][pydantic_ai.exceptions.UserError]
    rather than enabling the default tool.

    `tool_cls` comes first so the return type is solved from it rather than from `native_tool`,
    which is what lets callers use the result as a `tool_cls` without re-asserting.

    KEEP IN SYNC with the graph's resolution in `_prepare_request_parameters`, with two deliberate
    differences: `None`, which the native path omits for that step while the already-invoked
    fallback subagent cannot, so it raises; and the result type, which the graph accepts as any
    `AbstractNativeTool` while the fallback subagent needs the `tool_cls` its subagent is built
    around, so a sibling native tool raises here too.
    """
    if native_tool is not None and not isinstance(native_tool, AbstractNativeTool):
        native_tool = await await_maybe(native_tool(ctx))
    if native_tool is None:
        raise UserError(
            f'`{tool_cls.__name__}` native tool factory returned `None`, but the already-invoked '
            '`fallback_model` subagent cannot omit the tool. '
            f'Return a `{tool_cls.__name__}` instance, or drop `fallback_model` so that `None` omits the tool.'
        )
    if not isinstance(native_tool, tool_cls):
        raise UserError(f'Native tool must resolve to an instance of `{tool_cls.__name__}`, got {native_tool!r}')
    return native_tool
