from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from textwrap import dedent
from typing import Any

from ._run_context import RunContext
from .messages import ModelRequestPart, RetryPromptPart, ToolReturnPart

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

    def apply_template(self, message_part: ModelRequestPart, ctx: RunContext[Any]) -> ModelRequestPart:
        if isinstance(message_part, ToolReturnPart):
            if message_part.return_kind == 'final-result-processed':
                template = (
                    self.final_result_processed
                    if self.final_result_processed is not None
                    else DEFAULT_FINAL_RESULT_PROCESSED
                )
                message_part = self._apply_tool_template(message_part, ctx, template)
            elif message_part.return_kind == 'output-tool-not-executed':
                template = (
                    self.output_tool_not_executed
                    if self.output_tool_not_executed is not None
                    else DEFAULT_OUTPUT_TOOL_NOT_EXECUTED
                )
                message_part = self._apply_tool_template(message_part, ctx, template)
            elif message_part.return_kind == 'output-validation-failed':
                template = (
                    self.output_validation_failed
                    if self.output_validation_failed is not None
                    else DEFAULT_OUTPUT_VALIDATION_FAILED
                )
                message_part = self._apply_tool_template(message_part, ctx, template)
            elif message_part.return_kind == 'function-tool-not-executed':
                template = (
                    self.function_tool_not_executed
                    if self.function_tool_not_executed is not None
                    else DEFAULT_FUNCTION_TOOL_NOT_EXECUTED
                )
                message_part = self._apply_tool_template(message_part, ctx, template)
            elif message_part.return_kind == 'tool-denied':
                if self.tool_call_denied is not None:
                    message_part = self._apply_tool_template(message_part, ctx, self.tool_call_denied)
        elif isinstance(message_part, RetryPromptPart):
            template = self._get_template_for_retry(message_part)
            message_part = self._apply_retry_template(message_part, ctx, template)
        return message_part

    def _get_template_for_retry(
        self, message_part: RetryPromptPart
    ) -> str | Callable[[RetryPromptPart, RunContext[Any]], str]:
        # This is based on RetryPromptPart.model_response() implementation
        # We follow the same structure here to populate the correct template
        if isinstance(message_part.content, str):
            if message_part.tool_name is None:
                template = self.model_retry_string_no_tool
            else:
                template = self.model_retry_string_tool
        else:
            template = self.validation_errors_retry

        if template is None:
            template = DEFAULT_MODEL_RETRY

        return template

    def _apply_retry_template(
        self,
        message_part: RetryPromptPart,
        ctx: RunContext[Any],
        template: str | Callable[[RetryPromptPart, RunContext[Any]], str],
    ) -> RetryPromptPart:
        if isinstance(template, str):
            message_part = replace(message_part, retry_message=template)
        else:
            message_part = replace(message_part, retry_message=template(message_part, ctx))

        return message_part

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

    Example:
        ```python {test="skip"}
        from pydantic_ai import Agent, PromptConfig, ToolConfig

        agent = Agent('openai:gpt-4o')

        @agent.tool_plain(description='Search for items.')
        def search(query: str, limit: int) -> list[str]:
            return []

        # Override the description and arg descriptions at runtime
        result = agent.run_sync(
            'Find products',
            prompt_config=PromptConfig(
                tool_config=ToolConfig(
                    tool_descriptions={'search': 'Search product catalog by name or SKU.'},
                    tool_args_descriptions={
                        'search': {
                            'query': 'Product name or SKU code.',
                            'limit': 'Maximum results to return (1-100).',
                        }
                    },
                )
            ),
        )
        ```
    """

    tool_descriptions: dict[str, str] = field(default_factory=lambda: {})
    """Custom descriptions for tools, keyed by tool name."""

    tool_args_descriptions: dict[str, dict[str, str]] = field(default_factory=lambda: {})
    """Custom descriptions for tool arguments: `{'tool_name': {'arg_name': 'description'}}`."""

    def get_tool_args_for_tool(self, tool_name: str) -> dict[str, str] | None:
        """Get the tool argument descriptions for the given tool name."""
        return self.tool_args_descriptions.get(tool_name)


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

    Note:
        At least one of `templates` or `tool_config` must be provided. Creating a
        `PromptConfig()` with no arguments will raise a `ValueError`.

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

    tool_config: ToolConfig | None = None
    """Configuration for customizing tool descriptions and metadata.
    See [`ToolConfig`][pydantic_ai.ToolConfig] for available configuration options.
    """

    def __post_init__(self):  # pragma: no cover
        if self.templates is None and self.tool_config is None:
            raise ValueError(
                "PromptConfig requires at least 'templates' or 'tool_config' to be provided. "
                'Use PromptConfig(templates=PromptTemplates()) for default template behavior, '
                'or PromptConfig(tool_config=ToolConfig(...)) for tool customization.'
            )
