"""Deprecated import location for `pydantic_ai_harness.announce_current_time`.

This capability was renamed to follow the harness naming convention: a capability that acts on the
run is named with a verb phrase. Importing from here still works but emits a
`HarnessDeprecationWarning`. Import from `pydantic_ai_harness.announce_current_time` instead.
"""

from pydantic_ai_harness._warn import warn_module_renamed
from pydantic_ai_harness.announce_current_time._deprecated import CurrentTime  # pyright: ignore[reportDeprecated]

warn_module_renamed(
    'current_time',
    'announce_current_time',
    reason='`CurrentTime` is now `AnnounceCurrentTime`: capabilities that act on the run are named with a verb phrase.',
)

__all__ = [
    'CurrentTime',
]
