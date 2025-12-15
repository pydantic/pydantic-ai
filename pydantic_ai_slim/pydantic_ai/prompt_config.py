from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from textwrap import dedent
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._output import OutputSchema
    from ._run_context import RunContext as _RunContext


from .messages import ModelRequestPart, RetryPromptPart, ToolReturnPart


@dataclass
class PromptTemplates:
    """Templates for customizing system-generated messages that Pydantic AI sends to models.

    Each template can be either:
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
            validation_errors_retry=lambda part, ctx: f'Retry #{ctx.retries}: Fix the errors.',
        )

        agent = Agent('openai:gpt-4o', prompt_config=PromptConfig(templates=templates))
        ```
    """

    final_result_processed: str | Callable[[ToolReturnPart, _RunContext[Any]], str] = 'Final result processed.'
    """Confirmation message sent when a final result is successfully processed."""

    output_tool_not_executed: str | Callable[[ToolReturnPart, _RunContext[Any]], str] = (
        'Output tool not used - a final result was already processed.'
    )
    """Message sent when an output tool call is skipped because a result was already found."""

    function_tool_not_executed: str | Callable[[ToolReturnPart, _RunContext[Any]], str] = (
        'Tool not executed - a final result was already processed.'
    )
    """Message sent when a function tool call is skipped because a result was already found."""

    tool_call_denied: str | Callable[[ToolReturnPart, _RunContext[Any]], str] = 'The tool call was denied.'
    """Message sent when a tool call is denied by an approval handler.

    Note: Custom messages set via `ToolDenied` are preserved unless this template is explicitly overridden.
    """

    default_model_retry: str | Callable[[RetryPromptPart, _RunContext[Any]], str] = 'Fix the errors and try again.'
    """Default message sent when a `ModelRetry` exception is raised."""

    validation_errors_retry: str | Callable[[RetryPromptPart, _RunContext[Any]], str] = 'Fix the errors and try again.'
    """Message appended to validation errors when asking the model to retry."""

    model_retry_string_tool: str | Callable[[RetryPromptPart, _RunContext[Any]], str] = 'Fix the errors and try again.'
    """Message sent when a `ModelRetry` exception is raised from a tool."""

    model_retry_string_no_tool: str | Callable[[RetryPromptPart, _RunContext[Any]], str] = (
        'Fix the errors and try again.'
    )
    """Message sent when a `ModelRetry` exception is raised outside of a tool context."""

    prompted_output_template: str = dedent(
        """
        Always respond with a JSON object that's compatible with this schema:

        {schema}

        Don't include any text or Markdown fencing before or after.
        """
    )

    def apply_template(self, message_part: ModelRequestPart, ctx: _RunContext[Any]) -> ModelRequestPart:
        if isinstance(message_part, ToolReturnPart):
            if message_part.return_kind == 'final-result-processed':
                return self._apply_tool_template(message_part, ctx, self.final_result_processed)
            elif message_part.return_kind == 'output-tool-not-executed':
                return self._apply_tool_template(message_part, ctx, self.output_tool_not_executed)
            elif message_part.return_kind == 'function-tool-not-executed':
                return self._apply_tool_template(message_part, ctx, self.function_tool_not_executed)
            elif message_part.return_kind == 'tool-denied':
                # The content may already have a custom message from ToolDenied in which case we should not override it
                if self.tool_call_denied != DEFAULT_PROMPT_CONFIG.templates.tool_call_denied:
                    return self._apply_tool_template(message_part, ctx, self.tool_call_denied)
                return message_part
        elif isinstance(message_part, RetryPromptPart):
            template = self._get_template_for_retry(message_part)
            return self._apply_retry_tempelate(message_part, ctx, template)
        return message_part  # Returns the original message if no template is applied

    def _get_template_for_retry(
        self, message_part: RetryPromptPart
    ) -> str | Callable[[RetryPromptPart, _RunContext[Any]], str]:
        template: str | Callable[[RetryPromptPart, _RunContext[Any]], str] = self.default_model_retry
        # This is based no RetryPromptPart.model_response() implementation
        # We follow the same structure here to populate the correct template
        if isinstance(message_part.content, str):
            if message_part.tool_name is None:
                template = self.model_retry_string_no_tool
            else:
                template = self.model_retry_string_tool
        else:
            template = self.validation_errors_retry

        return template

    def _apply_retry_tempelate(
        self,
        message_part: RetryPromptPart,
        ctx: _RunContext[Any],
        template: str | Callable[[RetryPromptPart, _RunContext[Any]], str],
    ) -> RetryPromptPart:
        if isinstance(template, str):
            message_part = replace(message_part, retry_message=template)
        else:
            message_part = replace(message_part, retry_message=template(message_part, ctx))

        return message_part

    def _apply_tool_template(
        self,
        message_part: ToolReturnPart,
        ctx: _RunContext[Any],
        template: str | Callable[[ToolReturnPart, _RunContext[Any]], str],
    ) -> ToolReturnPart:
        if isinstance(template, str):
            message_part = replace(message_part, content=template)

        else:
            message_part = replace(message_part, content=template(message_part, ctx))
        return message_part

    def get_prompted_output_template(self, output_schema: OutputSchema[Any]) -> str | None:
        """Get the prompted output template for the given output schema."""
        from ._output import PromptedOutputSchema

        if not isinstance(output_schema, PromptedOutputSchema):
            return None

        return self.prompted_output_template


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

    def get_description_for_tool(self, tool_name: str) -> str | None:
        """Get the tool description for the given tool name."""
        return self.tool_descriptions.get(tool_name)

    def get_tool_arg_description(self, tool_name: str, arg_name: str) -> str | None:
        """Get the tool argument description for the given tool name and argument name."""
        tool_args = self.get_tool_args_for_tool(tool_name)
        if tool_args is None:
            return None
        return tool_args.get(arg_name)


@dataclass
class PromptConfig:
    """Configuration for customizing all strings and prompts sent to the model by Pydantic AI.

    `PromptConfig` provides a clean, extensible interface for overriding any text that
    Pydantic AI sends to the model. This includes:

    - **Prompt Templates**: Messages for retry prompts, tool return confirmations,
      validation errors, and other system-generated text via [`PromptTemplates`][pydantic_ai.PromptTemplates].
    - **Tool Configuration** (planned): Tool descriptions, parameter descriptions, and other
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
    """

    templates: PromptTemplates = field(default_factory=PromptTemplates)
    """Templates for customizing system-generated messages sent to the model.

    See [`PromptTemplates`][pydantic_ai.PromptTemplates] for available template options.
    """

    tool_config: ToolConfig | None = None
    """Configuration for customizing tool descriptions and metadata.
    See [`ToolConfig`][pydantic_ai.ToolConfig] for available configuration options.
    """


DEFAULT_PROMPT_CONFIG = PromptConfig()
"""The default prompt configuration used when no custom configuration is provided.

This uses the default [`PromptTemplates`][pydantic_ai.PromptTemplates] with sensible
defaults for all system-generated messages.
"""
