"""Status messages Pydantic AI writes as the `content` of a `ToolReturnPart` when a tool isn't run or its result isn't used."""

from __future__ import annotations as _annotations

from typing import Final

FINAL_RESULT_PROCESSED: Final = 'Final result processed.'
RETRY_WINS: Final = 'Output not used as the final result - addressing tool retries from this round first.'
OUTPUT_SKIPPED_FINAL_ALREADY_PROCESSED: Final = 'Output tool not used - a final result was already processed.'
OUTPUT_NOT_FINAL_RESULT: Final = 'Output tool processed, but its value will not be the final result of the agent run.'
OUTPUT_EXECUTION_FAILED: Final = 'Output tool not used - output function execution failed.'
OUTPUT_VALIDATION_FAILED: Final = 'Output tool not used - output failed validation.'
TOOL_SKIPPED_FINAL_ALREADY_PROCESSED: Final = 'Tool not executed - a final result was already processed.'

TOOL_STATUS_TEXTS: Final = frozenset(
    {
        FINAL_RESULT_PROCESSED,
        RETRY_WINS,
        OUTPUT_SKIPPED_FINAL_ALREADY_PROCESSED,
        OUTPUT_NOT_FINAL_RESULT,
        OUTPUT_EXECUTION_FAILED,
        OUTPUT_VALIDATION_FAILED,
        TOOL_SKIPPED_FINAL_ALREADY_PROCESSED,
    }
)
"""Every status message above, to tell them from text a tool returned, in history from any run."""
