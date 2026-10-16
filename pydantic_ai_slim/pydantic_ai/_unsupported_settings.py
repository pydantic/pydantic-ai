"""When a model warns that it ignores a setting: see `Model.unsupported_settings` and `UnsupportedSettingWarning`."""

from __future__ import annotations as _annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

GUARANTEE_SETTINGS: dict[str, str] = {
    'timeout': ' Use `request_timeout` instead, which Pydantic AI enforces for every model.',
    'max_retries': '',
    'stop_sequences': '',
    'parallel_tool_calls': '',
    'extra_headers': '',
    'extra_body': '',
}
"""The settings a model warns about ignoring, with a hint to append to the warning.

These change what a request guarantees, so ignoring one silently would surprise the user. The other settings are
tuning knobs that a model ignores silently, so the same settings work across models.
"""

_warnings_suppressed: ContextVar[bool] = ContextVar('_unsupported_setting_warnings_suppressed', default=False)


def warnings_suppressed() -> bool:
    return _warnings_suppressed.get()


@contextmanager
def suppress_warnings() -> Generator[None]:
    """Don't warn about ignored settings from `Model.prepare_request` calls made inside this block.

    For framework code that prepares a request without sending it (to resolve the output mode, or describe the
    request on a span, possibly in a Temporal workflow), so a request warns once, when the model that sends it
    prepares it; and for settings the user didn't choose, like the ones an MCP server asks a sampling model for.
    """
    token = _warnings_suppressed.set(True)
    try:
        yield
    finally:
        _warnings_suppressed.reset(token)
