from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import pydantic_core
from typing_extensions import assert_never

from pydantic_ai.usage import RunUsage

from ._run_context import RunContext
from .messages import ModelMessage, ModelRequest, ModelRequestPart, RetryPromptPart, ToolReturnKind, ToolReturnPart

if TYPE_CHECKING:
    from pydantic_ai.agent import Agent
    from pydantic_ai.models import Model

__all__ = (
    # templates & configuration
    'PromptConfig',
    'PromptTemplates',
    'ToolConfig',
    # defaults
    'DEFAULT_FINAL_RESULT_PROCESSED',
    'DEFAULT_OUTPUT_TOOL_NOT_EXECUTED',
    'DEFAULT_OUTPUT_VALIDATION_FAILED',
    'DEFAULT_FUNCTION_TOOL_NOT_EXECUTED',
    'DEFAULT_TOOL_CALL_DENIED',
    'DEFAULT_MODEL_RETRY',
    'DEFAULT_PROMPTED_OUTPUT_TEMPLATE',
)

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


def default_validation_feedback(content: str | list[pydantic_core.ErrorDetails]) -> str:
    """Generate a default validation feedback message."""
    assert isinstance(content, str)
    return f'Validation feedback:\n{content}'


def default_validation_error(content: str | list[pydantic_core.ErrorDetails]) -> str:
    """Generate a default validation error message from a list of Pydantic `ErrorDetails`."""
    from .messages import error_details_ta

    assert isinstance(content, list)

    json_errors = error_details_ta.dump_json(content, exclude={'__all__': {'ctx'}}, indent=2)
    plural = len(content) != 1
    return f'{len(content)} validation error{"s" if plural else ""}:\n```json\n{json_errors.decode()}\n```'


return_kind_to_default_prompt_template: dict[ToolReturnKind, str] = {
    'final-result-processed': DEFAULT_FINAL_RESULT_PROCESSED,
    'output-tool-not-executed': DEFAULT_OUTPUT_TOOL_NOT_EXECUTED,
    'output-validation-failed': DEFAULT_OUTPUT_VALIDATION_FAILED,
    'function-tool-not-executed': DEFAULT_FUNCTION_TOOL_NOT_EXECUTED,
    'tool-denied': DEFAULT_TOOL_CALL_DENIED,
    # tool-executed does not have a default prompt template or a prompt template callable
}


@dataclass
class PromptTemplates:
    """Templates for customizing system-generated messages that Pydantic AI sends to models.

    Each template can be either:
    - `None` to use the default message (or preserve existing content for `tool_denied`)
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
            validation_errors_retry_prompt='Please fix the validation errors.',
            final_result_processed='Done!',
        )

        # Using callable for dynamic messages
        templates = PromptTemplates(
            validation_errors_retry_prompt=lambda part, ctx: f'Retry #{ctx.retry}: Fix the errors.',
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

    tool_denied: str | Callable[[ToolReturnPart, RunContext[Any]], str] | None = None
    """Message sent when a tool call is denied by an approval handler.

    If `None`, preserves the custom message from `ToolDenied` (or uses the default if none was set).
    Set explicitly to override all denied tool messages.
    """

    validation_errors_retry_prompt: str | Callable[[RetryPromptPart, RunContext[Any]], str] | None = None
    """Message appended to validation errors when asking the model to retry.

    If `None`, uses the default: 'Fix the errors and try again.'
    """

    tool_retry_prompt: str | Callable[[RetryPromptPart, RunContext[Any]], str] | None = None
    """Message appended when a `ModelRetry` exception is raised from a tool.

    If `None`, uses the default: 'Fix the errors and try again.'
    """

    no_tool_retry_prompt: str | Callable[[RetryPromptPart, RunContext[Any]], str] | None = None
    """Message appended when a `ModelRetry` exception is raised outside of a tool context.

    If `None`, uses the default: 'Fix the errors and try again.'
    """

    prompted_output_template: str | None = None
    """Template for prompted output schema instructions.

    If set, this overrides the template used for [`PromptedOutput`][pydantic_ai.output.PromptedOutput].

    If `None`, the template from `PromptedOutput` is used if set, otherwise the model profile default
    ([`ModelProfile.prompted_output_template`][pydantic_ai.profiles.ModelProfile.prompted_output_template])
    is used.
    """

    description_template: Callable[[str | list[pydantic_core.ErrorDetails]], str] | None = None
    """Format a description message while asking the model to retry."""

    def apply_template(self, message_part: ModelRequestPart, ctx: RunContext[Any]) -> ModelRequestPart:
        if isinstance(message_part, ToolReturnPart):
            if message_part.return_kind in (None, 'tool-executed'):
                return message_part

            if message_part.return_kind == 'tool-denied':
                template = self.tool_denied
            elif message_part.return_kind == 'final-result-processed':
                template = self.final_result_processed
            elif message_part.return_kind == 'output-tool-not-executed':
                template = self.output_tool_not_executed
            elif message_part.return_kind == 'output-validation-failed':
                template = self.output_validation_failed
            elif message_part.return_kind == 'function-tool-not-executed':
                template = self.function_tool_not_executed
            else:
                assert_never(message_part.return_kind)

            # ToolDenied cannot fallback to a default prompt template
            # ToolDenied may have a template set via ToolDenied('')
            # If we set a default template then this message will get overridden by the default template, so we only set it if the template is explicitly set
            if message_part.return_kind == 'tool-denied':
                return self._apply_tool_template(message_part, ctx, template) if template else message_part

            if template := template or return_kind_to_default_prompt_template.get(message_part.return_kind):
                return self._apply_tool_template(message_part, ctx, template)

        elif isinstance(message_part, RetryPromptPart):
            return self._apply_retry_template(message_part, ctx)

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

    def _apply_retry_template(
        self,
        message_part: RetryPromptPart,
        ctx: RunContext[Any],
    ) -> RetryPromptPart:
        """Render the full retry response based on content type.

        Selects the appropriate templates and applies them in a single pass,
        pre-rendering everything so model_response() can just return the result.
        """
        content = message_part.content

        if isinstance(content, str):
            if message_part.tool_name is None:
                # String without tool context (e.g., output validator raising ModelRetry)
                description_template = self.description_template or default_validation_feedback
                description = description_template(content)
                retry_template = self.no_tool_retry_prompt or DEFAULT_MODEL_RETRY
            else:
                # String from a tool - use content directly
                description = content
                retry_template = self.tool_retry_prompt or DEFAULT_MODEL_RETRY
        else:
            # List of ErrorDetails (validation errors)
            description_template = self.description_template or default_validation_error
            description = description_template(content)
            retry_template = self.validation_errors_retry_prompt or DEFAULT_MODEL_RETRY

        # Resolve callable if needed
        if callable(retry_template):
            retry_template = retry_template(message_part, ctx)

        return replace(message_part, retry_message=f'{description}\n\n{retry_template}')

    def _apply_tool_template(
        self,
        message_part: ToolReturnPart,
        ctx: RunContext[Any],
        template: str | Callable[[ToolReturnPart, RunContext[Any]], str],
    ) -> ToolReturnPart:
        content = template(message_part, ctx) if callable(template) else template
        return replace(message_part, content=content)


@dataclass
class ToolConfig:
    """Configuration for customizing tool descriptions and argument descriptions at runtime.

    This allows you to override tool metadata without modifying the original tool definitions.

    This is applied via [`PromptConfig`][pydantic_ai.PromptConfig] and can be used to rename tools,
    override descriptions, enable strict schema behavior for supported models, and override argument
    descriptions.
    """

    name: str | None = None
    """The new name for the tool. If set, this overrides the original tool name."""

    description: str | None = None
    """The new description for the tool. If set, this overrides the original tool description."""

    strict: bool | None = None
    """Whether to enable strict schema behavior for the tool.

    If set, this overrides the strict setting on the tool definition.
    Currently only supported by OpenAI models.
    """

    parameters_descriptions: dict[str, str] | None = None
    """A dictionary mapping parameter names to their descriptions.

    This allows you to override or set descriptions for specific tool arguments.
    Keys are dot-separated paths to the parameter, and values are the descriptions.

    This supports nested fields and Pydantic models with references (e.g. recursive models).
    The path should follow the structure as if the reference was expanded inline.

    Example:
        Given a tool with arguments modeled like:
        ```python
        class Address(BaseModel):
            city: str = Field(description='City name')

        class User(BaseModel):
            name: str
            address: Address
            best_friend: User

        ToolConfig(
            parameters_descriptions={
                'name': 'The user\\'s full name',
                'address.city': 'The city where the user lives',
                # For recursive/referenced models, use dot notation:
                'best_friend.name': 'The name of the user\\'s best friend',
            }
        )
        ```

        This results in a JSON schema where the `best_friend` reference is inlined and modified,
        without affecting the original `User` definition:

        ```json
        {
          "properties": {
            "name": {"description": "The user's full name", "type": "string"},
            "best_friend": {
              "type": "object",
              "properties": {
                 # The description here is updated:
                 "name": {"description": "The name of the user's best friend", "type": "string"},
                 # Other fields remain as references or unchanged:
                 "address": {"$ref": "#/$defs/Address"},
                 "best_friend": {"$ref": "#/$defs/User"}
              }
            },
            ...
          },
          "$defs": {
             "User": {
               # The original definition remains unchanged:
               "properties": {
                 "name": {"title": "Name", "type": "string"},
                 ...
               }
             }
          }
        }
        ```
    """


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
                            validation_errors_retry_prompt='Please correct the errors and try again.',
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
    async def generate_prompt_config_from_agent(agent: Agent[Any, Any], model: Model) -> PromptConfig:
        """Generate a PromptConfig instance based on an Agent instance.

        This fills in per-tool metadata and default templates, producing a `PromptConfig` that can be
        used as a starting point for prompt optimizers.

        If the agent already has a `prompt_config`, it is used to override the defaults.
        """
        tool_config: dict[str, ToolConfig] = {}

        prompt_templates: PromptTemplates = DEFAULT_PROMPT_TEMPLATES

        run_ctx = RunContext(deps=None, model=model, usage=RunUsage())

        # Include both regular and output tools
        from .toolsets import CombinedToolset

        all_toolsets = [*agent.toolsets]
        if output_toolset := getattr(agent, '_output_toolset', None):
            all_toolsets.append(output_toolset)

        toolset = CombinedToolset(all_toolsets)
        tools = await toolset.get_tools(run_ctx)

        for tool_name, toolset_tool in tools.items():
            tool_def = toolset_tool.tool_def
            tool_config[tool_name] = ToolConfig(
                name=tool_name,
                description=tool_def.description,
                strict=tool_def.strict,
                parameters_descriptions=_extract_descriptions_from_json_schema(tool_def.parameters_json_schema),
            )

        if agent.prompt_config:
            if agent.prompt_config.templates:
                current_templates = asdict(agent.prompt_config.templates)
                updated_templates = {k: v for k, v in current_templates.items() if v is not None}
                prompt_templates = replace(prompt_templates, **updated_templates)

            if agent.prompt_config.tool_config:
                for tool_name, config in agent.prompt_config.tool_config.items():
                    if tool_name in tool_config:
                        current_config = asdict(config)
                        updates = {k: v for k, v in current_config.items() if v is not None}
                        tool_config[tool_name] = replace(tool_config[tool_name], **updates)
                    else:
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

            if description := value.get(_DESCRIPTION):
                result[full_path] = description

            if nested_props := value.get(_PROPERTIES):
                # Handle nested properties directly (e.g. nested models inline)
                extract_from_properties(full_path, nested_props)
            elif (ref := value.get(_REF)) and ref.startswith(_REF_PREFIX):
                # Handle $ref (shared definitions / recursive models)
                def_name = ref[len(_REF_PREFIX) :]
                # Avoid infinite recursion for recursive models (e.g. User -> best_friend: User)
                if def_name not in visited:
                    visited.add(def_name)
                    if nested_props := defs.get(def_name, {}).get(_PROPERTIES):
                        extract_from_properties(full_path, nested_props)
                    visited.remove(def_name)

    extract_from_properties('', properties)
    return result


DEFAULT_PROMPT_TEMPLATES = PromptTemplates(
    final_result_processed=DEFAULT_FINAL_RESULT_PROCESSED,
    output_tool_not_executed=DEFAULT_OUTPUT_TOOL_NOT_EXECUTED,
    output_validation_failed=DEFAULT_OUTPUT_VALIDATION_FAILED,
    function_tool_not_executed=DEFAULT_FUNCTION_TOOL_NOT_EXECUTED,
    tool_denied=DEFAULT_TOOL_CALL_DENIED,
    validation_errors_retry_prompt=DEFAULT_MODEL_RETRY,
    tool_retry_prompt=DEFAULT_MODEL_RETRY,
    no_tool_retry_prompt=DEFAULT_MODEL_RETRY,
    prompted_output_template=DEFAULT_PROMPTED_OUTPUT_TEMPLATE,
    description_template=None,
)
