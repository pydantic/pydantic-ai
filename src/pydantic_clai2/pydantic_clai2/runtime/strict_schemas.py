"""Strict mode off for every tool and output schema CLAI sends to a model."""

from dataclasses import dataclass, replace

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition


@dataclass
class NonStrictSchemas(AbstractCapability[AgentDepsT]):
    """Sets `strict=False` on every tool, output tool, and native output schema.

    OpenAI, Anthropic, Gemini 2.5+, and Bedrock otherwise turn strict mode on for any
    strict-compatible schema that leaves `strict` unset.
    """

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        return _non_strict(tool_defs)

    async def prepare_output_tools(
        self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return _non_strict(tool_defs)

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        parameters = request_context.model_request_parameters
        if parameters.output_object is not None:
            request_context.model_request_parameters = replace(
                parameters, output_object=replace(parameters.output_object, strict=False)
            )
        return request_context


def _non_strict(tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    return [replace(tool_def, strict=False) for tool_def in tool_defs]


NON_STRICT_SCHEMAS = NonStrictSchemas()
"""The one instance CLAI binds, so a session's plugin snapshot keeps its identity between runs."""
