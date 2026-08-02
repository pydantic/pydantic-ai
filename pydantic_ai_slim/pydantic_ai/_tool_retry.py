from pydantic_core import ValidationError

from .exceptions import ModelRetry
from .messages import RetryPromptPart


def build_tool_retry_prompt(tool_name: str, tool_call_id: str, error: ValidationError | ModelRetry) -> RetryPromptPart:
    """Build the retry prompt used for a failed tool-call validation."""
    content = (
        error.errors(include_url=False, include_context=False) if isinstance(error, ValidationError) else error.message
    )
    return RetryPromptPart(tool_name=tool_name, content=content, tool_call_id=tool_call_id)
