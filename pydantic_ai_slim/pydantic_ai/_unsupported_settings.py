"""When a model warns that it ignores a setting: see `Model.unsupported_settings` and `UnsupportedSettingWarning`."""

from __future__ import annotations as _annotations

import warnings
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from .exceptions import UnsupportedSettingWarning
from .settings import ModelSettingName

if TYPE_CHECKING:
    from .realtime.settings import RealtimeModelSettingName

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

_TOOL_CALL_GUARANTEES = frozenset({'parallel_tool_calls', 'async_tool_calls'})
"""Settings that are a guarantee only when `False` and the request offers function tools."""

SUPPRESSED_WARNINGS: ContextVar[frozenset[ModelSettingName]] = ContextVar(
    'unsupported_setting_warnings_suppressed', default=frozenset()
)
"""The settings `Model.prepare_request` doesn't warn about, set by `suppress_warnings`.

For `thinking`, the suppressed warning is that a provider-specific setting from the model's `settings` takes precedence
over it; for the others, that the model ignores them.
"""


@contextmanager
def suppress_warnings(
    settings: Iterable[ModelSettingName] = (*GUARANTEE_SETTINGS, 'thinking'),
) -> Generator[None]:
    """Don't warn about `settings` in `Model.prepare_request` calls inside this block.

    By default, these are all the `GUARANTEE_SETTINGS` a model may ignore, plus `thinking`, which a provider-specific
    setting from the model's `settings` may take precedence over.

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


def warn_unsupported_settings(
    model_id: str,
    settings: Mapping[str, object],
    unsupported: Iterable[str],
    *,
    has_function_tools: bool,
    guarantees: Mapping[ModelSettingName, str] | Mapping[RealtimeModelSettingName, str] = GUARANTEE_SETTINGS,
    stacklevel: int = 3,
) -> None:
    """Warn about each set setting in `guarantees` that the model `model_id` ignores (`unsupported`).

    Shared by `Model.prepare_request` and the realtime session, which passes its own `guarantees`, so both
    follow the same rule.
    """
    unsupported = frozenset(unsupported)
    suppressed = SUPPRESSED_WARNINGS.get()
    for name, hint in guarantees.items():
        if name not in settings or name not in unsupported or name in suppressed:
            continue
        # For these, only `False` is a guarantee, and only one about how function tools are called.
        if name in _TOOL_CALL_GUARANTEES and (settings.get(name) is not False or not has_function_tools):
            continue
        warnings.warn(
            f'The `{name}` setting is not supported by model {model_id!r} and will be ignored.{hint}',
            UnsupportedSettingWarning,
            stacklevel=stacklevel,
        )
