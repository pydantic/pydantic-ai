from __future__ import annotations

from pydantic_core import ValidationError

from .exceptions import ModelRetry
from .messages import RetryPromptPart


def retry_prompt_from_error(
    error: ValidationError | ModelRetry,
    *,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
) -> RetryPromptPart:
    """Build the retry prompt for a failed tool call or output validation.

    This is the exact message the model receives when the error is handled by the agent loop,
    so anything else presenting the failure (e.g. instrumentation spans) must build it the same way.
    """
    content = (
        error.errors(include_url=False, include_context=False) if isinstance(error, ValidationError) else error.message
    )
    part = RetryPromptPart(content=content, tool_name=tool_name)
    if tool_call_id:
        part.tool_call_id = tool_call_id
    return part
