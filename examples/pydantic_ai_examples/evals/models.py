from __future__ import annotations as _annotations

from pydantic import AwareDatetime, BaseModel
from typing_extensions import TypedDict


class TimeRangeBuilderSuccess(BaseModel, use_attribute_docstrings=True):
    """Response when a time range could be successfully generated."""

    min_timestamp_with_offset: AwareDatetime
    """A datetime in ISO format with timezone offset."""

    max_timestamp_with_offset: AwareDatetime
    """A datetime in ISO format with timezone offset."""

    explanation: str | None
    """
    A brief explanation of the time range that was selected.

    For example, if a user only mentions a specific point in time, you might explain that you selected a 10 minute
    window around that time.
    """

    def __str__(self):
        readable_min_timestamp = self.min_timestamp_with_offset.strftime(
            '%A, %B %d, %Y %H:%M:%S %Z'
        )
        readable_max_timestamp = self.max_timestamp_with_offset.strftime(
            '%A, %B %d, %Y %H:%M:%S %Z'
        )
        lines = [
            'TimeRangeBuilderSuccess:',
            f'* min_timestamp_with_offset: {readable_min_timestamp}',
            f'* max_timestamp_with_offset: {readable_max_timestamp}',
        ]
        if self.explanation is not None:
            lines.append(f'* explanation: {self.explanation}')
        return '\n'.join(lines)


class TimeRangeBuilderError(BaseModel):
    """Response when a time range cannot not be generated."""

    error_message: str

    def __str__(self):
        return f'TimeRangeBuilderError:\n* {self.error_message}'


TimeRangeResponse = TimeRangeBuilderSuccess | TimeRangeBuilderError


class TimeRangeInputs(TypedDict):
    """The inputs for the time range inference agent."""

    prompt: str
    now: AwareDatetime
