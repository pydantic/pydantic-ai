from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic

from pydantic_ai import _instructions
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, BuiltinToolFunc, RunContext
from pydantic_ai.toolsets import AbstractToolset


@dataclass
class AbstractCapability(ABC, Generic[AgentDepsT]):
    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the name used for spec serialization (CamelCase class name by default).

        Return None to opt out of spec-based construction.
        """
        return cls.__name__

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> AbstractCapability[Any]:
        """Create from spec arguments. Default: ``cls(*args, **kwargs)``.

        Override when ``__init__`` takes non-serializable types.
        """
        return cls(*args, **kwargs)

    def get_instructions(self) -> _instructions.Instructions[AgentDepsT] | None:
        # TODO: Use only the pre-request-hook based route instead of ctx.deps.get_instructions. How does override work? Just replace field instead of append?
        return None

    def get_model_settings(self) -> ModelSettings | None:
        return None

    # should take existing toolset(s)?
    # is capability stateful? can this cache a toolset in an ivar? or should it be a constant?
    # toolset state: https://github.com/pydantic/pydantic-ai/issues/4347

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        return None

    # builtin tools
    # should take existing builtin tools?
    # fallback how? https://github.com/pydantic/pydantic-ai/issues/3212 -> ToolDefinition.prefer_builtin='unique_id
    # can this check self.model.profile.supported_builtin_tools? how does interact with fallbackmodel?
    def get_builtin_tools(self) -> Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]]:
        return []

    # hooks: before, after, around

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        messages: list[ModelMessage],
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[ModelMessage], ModelSettings, ModelRequestParameters]:
        return messages, model_settings, model_request_parameters

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        response: ModelResponse,
    ) -> ModelResponse:
        return response

    # aenter, aexit
    # if this handles the toolset lifecycle, should this always call super()?
    # or does a toolset become a cap?

    # from_file

    # async def __aenter__(self) -> Self:
    #     """Enter the capability context.

    #     This is where you can set up network connections in a concrete implementation.
    #     """
    #     return self

    # async def __aexit__(self, *args: Any) -> bool | None:
    #     """Exit the toolset context.

    #     This is where you can tear down network connections in a concrete implementation.
    #     """
    #     return None
