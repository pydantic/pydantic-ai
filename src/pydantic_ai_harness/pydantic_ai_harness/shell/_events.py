"""Progress events emitted by the persistent `shell` tool.

Events carry command text and command output, so a UI that renders them must
treat the content as untrusted. They add no telemetry spans; core already
traces the tool call.
"""

from dataclasses import dataclass
from typing import Annotated

from pydantic import AliasChoices, Field
from typing_extensions import deprecated

from pydantic_ai import CapabilityEvent
from pydantic_ai_harness._warn import HarnessDeprecationWarning

SHELL_EVENTS = 'shell'


@dataclass(kw_only=True)
class CommandStartedEvent(CapabilityEvent, namespace=SHELL_EVENTS):
    """A command supervisor has started; its output log combines stdout and stderr."""

    command: str
    pid: int


@dataclass(kw_only=True)
class CommandOutputEvent(CapabilityEvent, namespace=SHELL_EVENTS):
    """A bounded chunk of the command's combined output log."""

    text: str


@dataclass(kw_only=True)
class CommandFinishedEvent(CapabilityEvent, namespace=SHELL_EVENTS):
    """The tool stopped waiting for the command, which may still be running.

    `exit_code` is `None` while no completed status has been published.
    """

    pid: int
    output_path: str
    status_path: str
    exit_code: int | None
    truncated: bool
    """Whether output beyond the per-call event budget was left in the log, or `max_log_bytes` emptied it."""
    line_count: Annotated[int | None, Field(validation_alias=AliasChoices('line_count', 'total_lines'))] = 0
    """Logical lines in the log up to 1 MiB; `None` for larger logs, which are not scanned, and emptied ones."""

    @property
    @deprecated('`total_lines` is deprecated, use `line_count` instead.', category=HarnessDeprecationWarning)
    def total_lines(self) -> int | None:
        """Deprecated alias for `line_count`."""
        return self.line_count
