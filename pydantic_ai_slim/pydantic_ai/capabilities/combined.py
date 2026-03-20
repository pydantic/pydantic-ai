from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai import _instructions, _system_prompt
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai.tools import AgentDepsT, BuiltinToolFunc, RunContext
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, ToolsetFunc
from pydantic_ai.toolsets._dynamic import DynamicToolset

from .abstract import AbstractCapability, BeforeModelRequestContext

if TYPE_CHECKING:
    from pydantic_ai.run import AgentRunResult


@dataclass
class CombinedCapability(AbstractCapability[AgentDepsT]):
    """A capability that combines multiple capabilities."""

    capabilities: Sequence[AbstractCapability[AgentDepsT]]

    def get_instructions(self) -> _instructions.Instructions[AgentDepsT] | None:
        instructions: list[str | _system_prompt.SystemPromptFunc[AgentDepsT]] = []
        for capability in self.capabilities:
            instructions.extend(_instructions.normalize_instructions(capability.get_instructions()))
        return instructions or None

    def get_model_settings(self) -> ModelSettings | None:
        model_settings: ModelSettings | None = None
        for capability in self.capabilities:
            cap_settings = capability.get_model_settings()
            if cap_settings is not None:
                model_settings = merge_model_settings(model_settings, cap_settings)
        return model_settings

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT] | None:
        toolsets: list[AbstractToolset[AgentDepsT]] = []
        for capability in self.capabilities:
            toolset = capability.get_toolset()
            if toolset is None:
                pass
            elif isinstance(toolset, AbstractToolset):
                toolsets.append(toolset)  # pyright: ignore[reportUnknownArgumentType]
            else:
                toolsets.append(DynamicToolset[AgentDepsT](toolset_func=toolset))
        return CombinedToolset(toolsets) if toolsets else None

    def get_builtin_tools(self) -> Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]]:
        builtin_tools: list[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] = []
        for capability in self.capabilities:
            builtin_tools.extend(capability.get_builtin_tools() or [])
        return builtin_tools

    # --- Run lifecycle hooks ---

    async def before_run(
        self,
        ctx: RunContext[AgentDepsT],
    ) -> None:
        for capability in self.capabilities:
            await capability.before_run(ctx)

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        for capability in reversed(self.capabilities):
            result = await capability.after_run(ctx, result=result)
        return result

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: Callable[[], Awaitable[AgentRunResult[Any]]],
    ) -> AgentRunResult[Any]:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_run_wrap(cap, ctx, chain)
        return await chain()

    # --- Event stream hook ---

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        for cap in reversed(self.capabilities):
            stream = await cap.wrap_run_event_stream(ctx, stream=stream)
        return stream

    # --- Model request lifecycle hooks ---

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: BeforeModelRequestContext,
    ) -> BeforeModelRequestContext:
        for capability in self.capabilities:
            request_context = await capability.before_model_request(ctx, request_context)
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        response: ModelResponse,
    ) -> ModelResponse:
        for capability in reversed(self.capabilities):
            response = await capability.after_model_request(ctx, response=response)
        return response

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        messages: list[ModelMessage],
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
        handler: Callable[
            [list[ModelMessage], ModelSettings, ModelRequestParameters],
            Awaitable[ModelResponse],
        ],
    ) -> ModelResponse:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_model_request_wrap(cap, ctx, chain)
        return await chain(messages, model_settings, model_request_parameters)

    # --- Tool validate lifecycle hooks ---

    async def before_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: str | dict[str, Any],
    ) -> str | dict[str, Any]:
        for capability in self.capabilities:
            args = await capability.before_tool_validate(ctx, call=call, args=args)
        return args

    async def after_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        for capability in reversed(self.capabilities):
            args = await capability.after_tool_validate(ctx, call=call, args=args)
        return args

    async def wrap_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: str | dict[str, Any],
        handler: Callable[[str | dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_tool_validate_wrap(cap, ctx, call, chain)
        return await chain(args)

    # --- Tool execute lifecycle hooks ---

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        for capability in self.capabilities:
            args = await capability.before_tool_execute(ctx, call=call, args=args)
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        for capability in reversed(self.capabilities):
            result = await capability.after_tool_execute(ctx, call=call, args=args, result=result)
        return result

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_tool_execute_wrap(cap, ctx, call, chain)
        return await chain(args)


# --- Composition helpers ---
# These create closures that bind the current capability and inner handler,
# building a middleware chain from outermost (first cap) to innermost (last cap).


def _make_run_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    inner: Callable[[], Awaitable[AgentRunResult[Any]]],
) -> Callable[[], Awaitable[AgentRunResult[Any]]]:
    async def wrapped() -> AgentRunResult[Any]:
        return await cap.wrap_run(ctx, handler=inner)

    return wrapped


def _make_model_request_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    inner: Callable[
        [list[ModelMessage], ModelSettings, ModelRequestParameters],
        Awaitable[ModelResponse],
    ],
) -> Callable[
    [list[ModelMessage], ModelSettings, ModelRequestParameters],
    Awaitable[ModelResponse],
]:
    async def wrapped(
        messages: list[ModelMessage],
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await cap.wrap_model_request(
            ctx,
            messages=messages,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
            handler=inner,
        )

    return wrapped


def _make_tool_validate_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    call: ToolCallPart,
    inner: Callable[[str | dict[str, Any]], Awaitable[dict[str, Any]]],
) -> Callable[[str | dict[str, Any]], Awaitable[dict[str, Any]]]:
    async def wrapped(args: str | dict[str, Any]) -> dict[str, Any]:
        return await cap.wrap_tool_validate(ctx, call=call, args=args, handler=inner)

    return wrapped


def _make_tool_execute_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    call: ToolCallPart,
    inner: Callable[[dict[str, Any]], Awaitable[Any]],
) -> Callable[[dict[str, Any]], Awaitable[Any]]:
    async def wrapped(args: dict[str, Any]) -> Any:
        return await cap.wrap_tool_execute(ctx, call=call, args=args, handler=inner)

    return wrapped
