from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from pydantic_ai import _instructions, _system_prompt
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.messages import ModelResponse
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai.tools import AgentDepsT, BuiltinToolFunc, RunContext
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, ToolsetFunc
from pydantic_ai.toolsets._dynamic import DynamicToolset

from .abstract import AbstractCapability, BeforeModelRequestContext


@dataclass
class CombinedCapability(AbstractCapability[AgentDepsT]):
    """A capability that combines multiple capabilities."""

    capabilities: Sequence[AbstractCapability[AgentDepsT]]

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        new_caps = [await c.for_run(ctx) for c in self.capabilities]
        if all(new is old for new, old in zip(new_caps, self.capabilities)):
            return self
        return replace(self, capabilities=new_caps)

    def get_instructions(self) -> _instructions.Instructions[AgentDepsT] | None:
        instructions: list[str | _system_prompt.SystemPromptFunc[AgentDepsT]] = []
        for capability in self.capabilities:
            instructions.extend(_instructions.normalize_instructions(capability.get_instructions()))
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
                merged = merge_model_settings(merged, func(ctx))
            return merged or ModelSettings()

        return resolve

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT] | None:
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

    def get_builtin_tools(self) -> Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]]:
        builtin_tools: list[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] = []
        for capability in self.capabilities:
            builtin_tools.extend(capability.get_builtin_tools() or [])
        return builtin_tools

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
