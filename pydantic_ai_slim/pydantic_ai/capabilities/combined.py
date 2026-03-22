from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pydantic_ai import _system_prompt
from pydantic_ai._instructions import AgentInstructions, normalize_instructions
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ToolCallPart
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai.tools import AgentBuiltinTool, AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, AgentToolset, CombinedToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

from .abstract import AbstractCapability, ModelRequestContext

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.result import FinalResult
    from pydantic_ai.run import AgentRunResult
    from pydantic_graph import End


@dataclass
class CombinedCapability(AbstractCapability[AgentDepsT]):
    """A capability that combines multiple capabilities."""

    capabilities: Sequence[AbstractCapability[AgentDepsT]]

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        new_caps = [await c.for_run(ctx) for c in self.capabilities]
        if all(new is old for new, old in zip(new_caps, self.capabilities)):
            return self
        return replace(self, capabilities=new_caps)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        instructions: list[str | _system_prompt.SystemPromptFunc[AgentDepsT]] = []
        for capability in self.capabilities:
            instructions.extend(normalize_instructions(capability.get_instructions()))
        return instructions or None

    def get_model_settings(self) -> ModelSettings | Callable[[RunContext[AgentDepsT]], ModelSettings] | None:
        static_settings: ModelSettings | None = None
        dynamic_settings: list[Callable[[RunContext[AgentDepsT]], ModelSettings]] = []
        for capability in self.capabilities:
            cap_settings = capability.get_model_settings()
            if cap_settings is None:
                pass
            elif callable(cap_settings):
                dynamic_settings.append(cap_settings)
            else:
                static_settings = merge_model_settings(static_settings, cap_settings)
        if not dynamic_settings:
            return static_settings

        def resolve(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            merged = static_settings
            for func in dynamic_settings:
                # Update ctx.model_settings so each callable sees prior capabilities' contributions
                ctx.model_settings = merge_model_settings(ctx.model_settings, merged)
                resolved = func(ctx)
                merged = merge_model_settings(merged, resolved)
            return merged if merged is not None else ModelSettings()

        return resolve

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        toolsets: list[AbstractToolset[AgentDepsT]] = []
        for capability in self.capabilities:
            toolset = capability.get_toolset()
            if toolset is None:
                pass
            elif isinstance(toolset, AbstractToolset):
                # Pyright can't narrow Callable type aliases out of unions after isinstance check
                toolsets.append(toolset)  # pyright: ignore[reportUnknownArgumentType]
            else:
                toolsets.append(DynamicToolset[AgentDepsT](toolset_func=toolset))
        return CombinedToolset(toolsets) if toolsets else None

    def get_builtin_tools(self) -> Sequence[AgentBuiltinTool[AgentDepsT]]:
        builtin_tools: list[AgentBuiltinTool[AgentDepsT]] = []
        for capability in self.capabilities:
            builtin_tools.extend(capability.get_builtin_tools() or [])
        return builtin_tools

    # --- Tool preparation hook ---

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        for capability in self.capabilities:
            tool_defs = await capability.prepare_tools(ctx, tool_defs)
        return tool_defs

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

    async def wrap_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        handler: Callable[
            [_agent_graph.AgentNode[AgentDepsT, Any]],
            Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]],
        ],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_node_run_wrap(cap, ctx, chain)
        return await chain(node)

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
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        for capability in self.capabilities:
            request_context = await capability.before_model_request(ctx, request_context)
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        for capability in reversed(self.capabilities):
            response = await capability.after_model_request(ctx, request_context=request_context, response=response)
        return response

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_model_request_wrap(cap, ctx, chain)
        return await chain(request_context)

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
    inner: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
) -> Callable[[ModelRequestContext], Awaitable[ModelResponse]]:
    async def wrapped(request_context: ModelRequestContext) -> ModelResponse:
        return await cap.wrap_model_request(
            ctx,
            request_context=request_context,
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


def _make_node_run_wrap(
    cap: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    inner: Callable[
        [_agent_graph.AgentNode[AgentDepsT, Any]],
        Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]],
    ],
) -> Callable[
    [_agent_graph.AgentNode[AgentDepsT, Any]],
    Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]],
]:
    async def wrapped(
        node: _agent_graph.AgentNode[AgentDepsT, Any],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        return await cap.wrap_node_run(ctx, node=node, handler=inner)

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
