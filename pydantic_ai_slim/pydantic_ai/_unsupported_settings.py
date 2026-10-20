"""When a model warns that it ignores a setting: see `Model.unsupported_settings` and `UnsupportedSettingWarning`."""

from __future__ import annotations as _annotations

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar

from .settings import ModelSettingName

GUARANTEE_SETTINGS: dict[ModelSettingName, str] = {
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

SUPPRESSED_WARNINGS: ContextVar[frozenset[ModelSettingName]] = ContextVar(
    'unsupported_setting_warnings_suppressed', default=frozenset()
)
"""The settings `Model.prepare_request` doesn't warn about ignoring, set by `suppress_warnings`."""


@contextmanager
def suppress_warnings(settings: Iterable[ModelSettingName] = GUARANTEE_SETTINGS) -> Generator[None]:
    """Don't warn about ignoring `settings` (by default, all of them) in `Model.prepare_request` calls inside this block.

    For framework code that prepares a request without sending it (to resolve the output mode, count its tokens, or
    describe it on a span, possibly in a Temporal workflow), so a request warns once, when the model that sends it
    prepares it; and for settings the user didn't choose, like the `stop_sequences` an MCP server's sampling request
    sends.
    """
    token = SUPPRESSED_WARNINGS.set(SUPPRESSED_WARNINGS.get() | frozenset(settings))
    try:
        yield
    finally:
        SUPPRESSED_WARNINGS.reset(token)
