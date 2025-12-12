from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._run_context import RunContext as _RunContext

from .messages import ModelRequestPart, RetryPromptPart, ToolReturnPart


@dataclass
class PromptTemplates:
    """Templates for customizing messages that Pydantic AI sends to models.

    Each template can be a static string or a callable that receives context and returns a string.
    """

    retry_prompt: str | Callable[[RetryPromptPart, _RunContext[Any]], str] | None = None
    """Message sent to the model after validation failures or invalid responses.

    Default: "Validation feedback: {errors}\\n\\nFix the errors and try again."
    """

    final_result_processed: str | Callable[[ToolReturnPart, _RunContext[Any]], str] = 'Final result processed.'
    """Confirmation message sent when a final result is successfully processed.

    """

    output_tool_not_executed: str | Callable[[ToolReturnPart, _RunContext[Any]], str] = (
        'Output tool not used - a final result was already processed.'
    )
    """Message sent when an output tool call is skipped because a result was already found.

    """

    function_tool_not_executed: str | Callable[[ToolReturnPart, _RunContext[Any]], str] = (
        'Tool not executed - a final result was already processed.'
    )
    """Message sent when a function tool call is skipped because a result was already found.

    """

    tool_call_denied: str | Callable[[ToolReturnPart, _RunContext[Any]], str] = 'Tool call was denied.'
    """Message sent when a tool call is denied."""

    def apply_template(self, message_part: ModelRequestPart, ctx: _RunContext[Any]) -> ModelRequestPart:
        if isinstance(message_part, ToolReturnPart):
            if message_part.return_kind == 'final-result-processed':
                return self._apply_tool_template(message_part, ctx, self.final_result_processed)
            elif message_part.return_kind == 'output-tool-not-executed':
                return self._apply_tool_template(message_part, ctx, self.output_tool_not_executed)
            elif message_part.return_kind == 'function-tool-not-executed':
                return self._apply_tool_template(message_part, ctx, self.function_tool_not_executed)
            elif message_part.return_kind == 'tool-denied':
                # For tool-denied, only apply template if user configured a custom one
                # The content may already have a custom message from ToolDenied
                if self.tool_call_denied != DEFAULT_PROMPT_TEMPLATES.tool_call_denied:
                    return self._apply_tool_template(message_part, ctx, self.tool_call_denied)
                return message_part
        elif isinstance(message_part, RetryPromptPart) and self.retry_prompt:
            if isinstance(self.retry_prompt, str):
                return replace(message_part, retry_message=self.retry_prompt)
            else:
                return replace(message_part, retry_message=self.retry_prompt(message_part, ctx))
        return message_part  # Returns the original message if no template is applied

    def _apply_tool_template(
        self,
        message: ToolReturnPart,
        ctx: _RunContext[Any],
        template: str | Callable[[ToolReturnPart, _RunContext[Any]], str],
    ):
        message_part: ToolReturnPart = message

        if isinstance(template, str):
            message_part = replace(message_part, content=template)

        else:
            message_part = replace(message_part, content=template(message, ctx))
        return message_part


DEFAULT_PROMPT_TEMPLATES = PromptTemplates()
