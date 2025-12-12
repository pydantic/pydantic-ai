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

    tool_call_denied: str | Callable[[ToolReturnPart, _RunContext[Any]], str] = 'The tool call was denied.'
    """Message sent when a tool call is denied."""

    validation_errors_retry: str | Callable[[RetryPromptPart, _RunContext[Any]], str] = 'Fix these validation errors and try again.'
    """Message sent to the model after validation errors."""

    model_retry_string_tool: str | Callable[[RetryPromptPart, _RunContext[Any]], str] = 'The previous response was invalid. Please try again.'
    """Message sent to the model when a ModelRetry exception is raised and tool is present."""

    model_retry_string_no_tool: str | Callable[[RetryPromptPart, _RunContext[Any]], str] = 'The previous response was invalid. Please try again without using any tools.'
    """Message sent to the model when a ModelRetry exception is raised and no tool is present."""

    
    

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
                if self.tool_call_denied != DEFAULT_PROMPT_TEMPLATES.tool_call_denied:
                    return self._apply_tool_template(message_part, ctx, self.tool_call_denied)
                return message_part
        elif isinstance(message_part, RetryPromptPart):
            template = self._get_template_for_retry(message_part)
            return self._apply_retry_tempelate(message_part, ctx, template)
        return message_part  # Returns the original message if no template is applied
    
    def _get_template_for_retry(self, message: RetryPromptPart) -> str | Callable[[RetryPromptPart, _RunContext[Any]], str]:
        if isinstance(message.content, str):
            if message.tool_name is None:
                return self.model_retry_string_no_tool
            else:
                return self.model_retry_string_tool
        else:
            return self.validation_errors_retry

    def _apply_retry_tempelate(self, message: RetryPromptPart, ctx: _RunContext[Any], template: str | Callable[[RetryPromptPart, _RunContext[Any]], str]):
        if isinstance(template, str):
            return replace(message, retry_message=template)
        else:
            return replace(message, retry_message=template(message, ctx))

        

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
