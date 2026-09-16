"""Bounded shell progress events for attached Coder calls."""

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent


@dataclass(kw_only=True)
class ShellStartedEvent(CapabilityEvent, namespace='coder', name='shell_started'):
    """A shell supervisor has started; output combines stdout and stderr."""

    command: str
    pid: int


@dataclass(kw_only=True)
class ShellOutputEvent(CapabilityEvent, namespace='coder', name='shell_output'):
    """A bounded chunk of the command's combined output log."""

    text: str


@dataclass(kw_only=True)
class ShellFinishedEvent(CapabilityEvent, namespace='coder', name='shell_finished'):
    """The tool stopped waiting, not necessarily the underlying process."""

    pid: int
    output_path: str
    status_path: str
    exit_code: int | None
    truncated: bool
    total_lines: int | None = 0
    """Logical lines in logs up to 1 MiB; `None` for larger logs to bound scan work."""
