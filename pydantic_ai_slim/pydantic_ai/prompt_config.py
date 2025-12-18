from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import pydantic_core
from typing_extensions import assert_never

from pydantic_ai.usage import RunUsage

from ._run_context import RunContext
from .messages import ModelMessage, ModelRequest, ModelRequestPart, RetryPromptPart, ToolReturnPart

if TYPE_CHECKING:
    from pydantic_ai.agent import Agent
    from pydantic_ai.models import Model

# Default template strings - used when template field is None
DEFAULT_FINAL_RESULT_PROCESSED = 'Final result processed.'
"""Default confirmation message when a final result is successfully processed."""

DEFAULT_OUTPUT_TOOL_NOT_EXECUTED = 'Output tool not used - a final result was already processed.'
"""Default message when an output tool call is skipped because a result was already found."""

DEFAULT_OUTPUT_VALIDATION_FAILED = 'Output tool not used - output failed validation.'
"""Default message when an output tool fails validation."""

DEFAULT_FUNCTION_TOOL_NOT_EXECUTED = 'Tool not executed - a final result was already processed.'
"""Default message when a function tool call is skipped because a result was already found."""

DEFAULT_TOOL_CALL_DENIED = 'The tool call was denied.'
"""Default message when a tool call is denied by an approval handler."""

DEFAULT_MODEL_RETRY = 'Fix the errors and try again.'
"""Default message appended to retry prompts."""

DEFAULT_PROMPTED_OUTPUT_TEMPLATE = dedent(
    """
    Always respond with a JSON object that's compatible with this schema:

    {schema}

    Don't include any text or Markdown fencing before or after.
    """
)
"""Default template for prompted output schema instructions."""


def DEFAULT_VALIDATION_FEEDBACK(content: str | list[pydantic_core.ErrorDetails]) -> str:
    """Generate a default validation feedback message."""
    assert isinstance(content, str)
    return f'Validation feedback:\n{content}'


def DEFAULT_VALIDATION_ERROR(content: str | list[pydantic_core.ErrorDetails]) -> str:
    """Generate a default validation error message from a list of Pydantic `ErrorDetails`."""
    from .messages import error_details_ta

    assert isinstance(content, list)

    json_errors = error_details_ta.dump_json(content, exclude={'__all__': {'ctx'}}, indent=2)
    plural = len(content) != 1
    return f'{len(content)} validation error{"s" if plural else ""}:\n```json\n{json_errors.decode()}\n```'


return_kind_to_default_prompt_template: dict[str, str] = {
    'final-result-processed': DEFAULT_FINAL_RESULT_PROCESSED,
    'output-tool-not-executed': DEFAULT_OUTPUT_TOOL_NOT_EXECUTED,
    'output-validation-failed': DEFAULT_OUTPUT_VALIDATION_FAILED,
    'function-tool-not-executed': DEFAULT_FUNCTION_TOOL_NOT_EXECUTED,
    'tool-denied': DEFAULT_TOOL_CALL_DENIED,
}


@dataclass
class PromptTemplates:
    """Templates for customizing system-generated messages that Pydantic AI sends to models.

    Each template can be either:
    - `None` to use the default message (or preserve existing content for `tool_call_denied`)
    - A static string that replaces the default message
    - A callable that receives the message part and [`RunContext`][pydantic_ai.RunContext]
      and returns a dynamically generated string

    These templates are used within [`PromptConfig`][pydantic_ai.PromptConfig] to customize
    retry prompts, tool return confirmations, validation error messages, and more.

    Example:
        ```python
        from pydantic_ai import Agent, PromptConfig, PromptTemplates

        # Using static strings
        templates = PromptTemplates(
            validation_errors_retry='Please fix the validation errors.',
            final_result_processed='Done!',
        )

        # Using callable for dynamic messages
        templates = PromptTemplates(
            validation_errors_retry=lambda part, ctx: f'Retry #{ctx.retry}: Fix the errors.',
        )

        agent = Agent('openai:gpt-4o', prompt_config=PromptConfig(templates=templates))
        ```
    """

    final_result_processed: str | Callable[[ToolReturnPart, RunContext[Any]], str] | None = None
    """Confirmation message sent when a final result is successfully processed.

    If `None`, uses the default: 'Final result processed.'
    """

    output_tool_not_executed: str | Callable[[ToolReturnPart, RunContext[Any]], str] | None = None
    """Message sent when an output tool call is skipped because a result was already found.

    If `None`, uses the default: 'Output tool not used - a final result was already processed.'
    """

    output_validation_failed: str | Callable[[ToolReturnPart, RunContext[Any]], str] | None = None
    """Message sent when an output tool fails validation."""

    function_tool_not_executed: str | Callable[[ToolReturnPart, RunContext[Any]], str] | None = None
    """Message sent when a function tool call is skipped because a result was already found.

    If `None`, uses the default: 'Tool not executed - a final result was already processed.'
    """

    tool_call_denied: str | Callable[[ToolReturnPart, RunContext[Any]], str] | None = None
    """Message sent when a tool call is denied by an approval handler.

    If `None`, preserves the custom message from `ToolDenied` (or uses the default if none was set).
    Set explicitly to override all denied tool messages.
    """

    validation_errors_retry: str | Callable[[RetryPromptPart, RunContext[Any]], str] | None = None
    """Message appended to validation errors when asking the model to retry.

    If `None`, uses the default: 'Fix the errors and try again.'
    """

    model_retry_string_tool: str | Callable[[RetryPromptPart, RunContext[Any]], str] | None = None
    """Message sent when a `ModelRetry` exception is raised from a tool.

    If `None`, uses the default: 'Fix the errors and try again.'
    """

    model_retry_string_no_tool: str | Callable[[RetryPromptPart, RunContext[Any]], str] | None = None
    """Message sent when a `ModelRetry` exception is raised outside of a tool context.

    If `None`, uses the default: 'Fix the errors and try again.'
    """

    prompted_output_template: str | None = None
    """Template for prompted output schema instructions.

    If `None`, uses the template from `PromptedOutput` if set, otherwise the model's
    profile-specific default template is used.
    Set explicitly to override the template for all prompted outputs.
    """

    description_template: Callable[[str | list[pydantic_core.ErrorDetails]], str] | None = None
    """Generate a description message prefix while asking the model to retry."""

    def apply_template(self, message_part: ModelRequestPart, ctx: RunContext[Any]) -> ModelRequestPart:
        if isinstance(message_part, ToolReturnPart):
            template: str | Callable[[ToolReturnPart, RunContext[Any]], str] | None = None
            if message_part.return_kind == 'final-result-processed':
                template = (
                    self.final_result_processed or return_kind_to_default_prompt_template[message_part.return_kind]
                )
            elif message_part.return_kind == 'output-tool-not-executed':
                template = (
                    self.output_tool_not_executed or return_kind_to_default_prompt_template[message_part.return_kind]
                )
            elif message_part.return_kind == 'output-validation-failed':
                template = (
                    self.output_validation_failed or return_kind_to_default_prompt_template[message_part.return_kind]
                )
            elif message_part.return_kind == 'function-tool-not-executed':
                template = (
                    self.function_tool_not_executed or return_kind_to_default_prompt_template[message_part.return_kind]
                )
            elif message_part.return_kind == 'tool-denied':
                if self.tool_call_denied is not None:
                    message_part = self._apply_tool_template(message_part, ctx, self.tool_call_denied)
            elif message_part.return_kind == 'tool-executed' or message_part.return_kind is None:
                pass  # No template applied for normal tool execution or when return_kind is None
            else:
                assert_never(message_part.return_kind)

            if template is not None:
                message_part = self._apply_tool_template(message_part, ctx, template)
        elif isinstance(message_part, RetryPromptPart):
            message_part = self._apply_retry_template(message_part, ctx, *self._get_template_for_retry(message_part))
        # Other part types (UserPromptPart, SystemPromptPart, etc.) are returned unchanged
        # as they don't have system-generated content that needs templating.
        return message_part

    def apply_template_message_history(self, _messages: list[ModelMessage], ctx: RunContext[Any]) -> list[ModelMessage]:
        return [
            replace(
                message,
                parts=[self.apply_template(part, ctx) for part in message.parts],
            )
            if isinstance(message, ModelRequest)
            else message
            for message in _messages
        ]

    def _get_template_for_retry(
        self, message_part: RetryPromptPart
    ) -> tuple[
        None | Callable[[str | list[pydantic_core.ErrorDetails]], str],
        str | Callable[[RetryPromptPart, RunContext[Any]], str],
    ]:
        # This is based on RetryPromptPart.model_response() implementation
        # We follow the same structure here to populate the correct template
        description_template: None | Callable[[str | list[pydantic_core.ErrorDetails]], str] = None
        if isinstance(message_part.content, str):
            if message_part.tool_name is None:
                template = self.model_retry_string_no_tool
                description_template = self.description_template or DEFAULT_VALIDATION_FEEDBACK
            else:
                template = self.model_retry_string_tool
        else:
            template = self.validation_errors_retry
            description_template = self.description_template or DEFAULT_VALIDATION_ERROR

        return (description_template, template or DEFAULT_MODEL_RETRY)

    def _apply_retry_template(
        self,
        message_part: RetryPromptPart,
        ctx: RunContext[Any],
        description_template: None | Callable[[str | list[pydantic_core.ErrorDetails]], str],
        template: str | Callable[[RetryPromptPart, RunContext[Any]], str],
    ) -> RetryPromptPart:
        if not isinstance(template, str):
            template = template(message_part, ctx)
        return replace(message_part, retry_message=template, description_template=description_template)

    def _apply_tool_template(
        self,
        message_part: ToolReturnPart,
        ctx: RunContext[Any],
        template: str | Callable[[ToolReturnPart, RunContext[Any]], str],
    ) -> ToolReturnPart:
        if isinstance(template, str):
            message_part = replace(message_part, content=template)
        else:
            message_part = replace(message_part, content=template(message_part, ctx))
        return message_part


@dataclass
class ToolConfig:
    """Configuration for customizing tool descriptions and argument descriptions at runtime.

    This allows you to override tool metadata without modifying the original tool definitions.
    """

    name: str | None = None
    tool_description: str | None = None
    strict: bool | None = None
    tool_args_descriptions: dict[str, str] | None = None


@dataclass
class PromptConfig:
    """Configuration for customizing all strings and prompts sent to the model by Pydantic AI.

    `PromptConfig` provides a clean, extensible interface for overriding any text that
    Pydantic AI sends to the model. This includes:

    - **Prompt Templates**: Messages for retry prompts, tool return confirmations,
      validation errors, and other system-generated text via [`PromptTemplates`][pydantic_ai.PromptTemplates].
    - **Tool Configuration**: Tool descriptions, parameter descriptions, and other
      tool metadata - allowing you to override descriptions and args for tools at the agent level.

    This allows you to fully customize how your agent communicates with the model
    without modifying the underlying tool or agent code.

    Example:
        ```python
        from pydantic_ai import Agent, PromptConfig, PromptTemplates

        agent = Agent(
            'openai:gpt-4o',
            prompt_config=PromptConfig(
                templates=PromptTemplates(
                    validation_errors_retry='Please correct the errors and try again.',
                    final_result_processed='Result received successfully.',
                ),
            ),
        )
        ```

    Attributes:
        templates: Templates for customizing system-generated messages like retry prompts,
            tool return confirmations, and validation error messages.
        tool_config: Configuration for customizing tool descriptions and metadata.
    """

    templates: PromptTemplates | None = None
    """Templates for customizing system-generated messages sent to the model.

    See [`PromptTemplates`][pydantic_ai.PromptTemplates] for available template options.
    """

    tool_config: dict[str, ToolConfig] | None = None
    """Configuration for customizing tool descriptions and metadata, keyed by tool name.
    See [`ToolConfig`][pydantic_ai.ToolConfig] for available configuration options.
    """

    @staticmethod
    async def generate_prompt_config_from_agent(agent: Agent, model: Model) -> PromptConfig:
        """Generate a PromptConfig instance based on an Agent instance.

        The information we can find we will fill in, everything else can be None or defaults if any.

        """
        tool_config: dict[str, ToolConfig] = {}

        prompt_templates: PromptTemplates = PromptTemplates(
            final_result_processed=DEFAULT_FINAL_RESULT_PROCESSED,
            output_tool_not_executed=DEFAULT_OUTPUT_TOOL_NOT_EXECUTED,
            output_validation_failed=DEFAULT_OUTPUT_VALIDATION_FAILED,
            function_tool_not_executed=DEFAULT_FUNCTION_TOOL_NOT_EXECUTED,
            tool_call_denied=DEFAULT_TOOL_CALL_DENIED,
            validation_errors_retry=DEFAULT_MODEL_RETRY,
            model_retry_string_tool=DEFAULT_MODEL_RETRY,
            model_retry_string_no_tool=DEFAULT_MODEL_RETRY,
            prompted_output_template=DEFAULT_PROMPTED_OUTPUT_TEMPLATE,
            description_template=DEFAULT_VALIDATION_FEEDBACK,
        )

        run_ctx = RunContext(deps=None, model=model, usage=RunUsage())

        toolsets = agent.toolsets
        for toolset in toolsets:
            tools = await toolset.get_tools(run_ctx)
            for tool_name, toolset_tool in tools.items():
                tool_def = toolset_tool.tool_def
                config = ToolConfig(
                    name=tool_name,
                    tool_description=tool_def.description,
                    strict=tool_def.strict,
                    tool_args_descriptions=_extract_descriptions_from_json_schema(tool_def.parameters_json_schema),
                )
                tool_config[tool_name] = config

        return PromptConfig(
            tool_config=tool_config,
            templates=prompt_templates,
        )


# JSON Schema keys
_PROPERTIES = 'properties'
_DEFS = '$defs'
_REF = '$ref'
_REF_PREFIX = '#/$defs/'
_DESCRIPTION = 'description'


def _extract_descriptions_from_json_schema(parameters_json_schema: dict[str, Any]) -> dict[str, str]:
    """Extract field descriptions from a JSON schema into dot notation format.

    Recursively traverses the schema's properties to build a flat dictionary mapping
    dot-notation paths to their descriptions. This is useful for prompt optimizers
    that need to modify tool argument descriptions.

    Handles two types of nested structures:
    - Inline nested properties (e.g., `{'type': 'object', 'properties': {...}}`)
    - `$ref` references to `$defs` (e.g., `{'$ref': '#/$defs/NestedModel'}`)

    Example:
        Given a schema for a tool with nested BaseModel arguments::

            schema = {
                '$defs': {'Config': {'properties': {'level': {'description': 'Log level'}}}},
                'properties': {
                    'name': {'description': 'User name'},
                    'config': {'$ref': '#/$defs/Config'}
                }
            }
            # Result: {'name': 'User name', 'config.level': 'Log level'}

    Args:
        parameters_json_schema: The JSON schema containing properties and optional $defs.

    Returns:
        A dict mapping dot-notation paths (e.g., 'user.profile.address') to descriptions.
    """
    properties = parameters_json_schema.get(_PROPERTIES, {})
    if not properties:
        return {}

    result: dict[str, str] = {}
    defs = parameters_json_schema.get(_DEFS, {})
    visited: set[str] = set()

    def extract_from_properties(path: str, props: dict[str, Any]) -> None:
        """Recursively extract descriptions from properties."""
        for key, value in props.items():
            full_path = f'{path}.{key}' if path else key

            if _DESCRIPTION in value:
                result[full_path] = value[_DESCRIPTION]

            if _PROPERTIES in value:
                extract_from_properties(full_path, value[_PROPERTIES])

            elif _REF in value and value[_REF].startswith(_REF_PREFIX):
                def_name = value[_REF][len(_REF_PREFIX) :]
                # Guard against circular references to prevent infinite recursion
                if def_name in visited:
                    continue
                visited.add(def_name)
                referenced_def = defs.get(def_name, {})
                if _PROPERTIES in referenced_def:
                    extract_from_properties(full_path, referenced_def[_PROPERTIES])

    extract_from_properties('', properties)
    return result
