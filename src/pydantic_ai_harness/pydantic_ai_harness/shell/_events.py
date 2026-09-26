"""Progress events emitted by the persistent `shell` tool.

Events carry command text and command output, so a UI that renders them must
treat the content as untrusted. They add no telemetry spans; core already
traces the tool call.
"""

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent

SHELL_EVENTS = 'shell'


@dataclass(kw_only=True)
class CommandStartedEvent(CapabilityEvent, namespace=SHELL_EVENTS, name='command_started'):
    """A command supervisor has started; its output log combines stdout and stderr."""

    command: str
    pid: int


@dataclass(kw_only=True)
class CommandOutputEvent(CapabilityEvent, namespace=SHELL_EVENTS, name='command_output'):
    """A bounded chunk of the command's combined output log."""

    text: str


@dataclass(kw_only=True)
class CommandFinishedEvent(CapabilityEvent, namespace=SHELL_EVENTS, name='command_finished'):
    """The tool stopped waiting for the command, which may still be running.

    `exit_code` is `None` while no completed status has been published.
    """

    pid: int
    output_path: str
    status_path: str
    exit_code: int | None
    truncated: bool
    """Whether output beyond the per-call event budget was left in the log."""
    total_lines: int | None = 0
    """Logical lines in the log up to 1 MiB; `None` for larger logs, which are not scanned."""
