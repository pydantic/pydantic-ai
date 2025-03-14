from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .persistence import SnapshotStatus


class GraphSetupError(TypeError):
    """Error caused by an incorrectly configured graph."""

    message: str
    """Description of the mistake."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class GraphRuntimeError(RuntimeError):
    """Error caused by an issue during graph execution."""

    message: str
    """The error message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class GraphNodeStatusError(GraphRuntimeError):
    """Error caused by trying to run a node that already has status `'running'`, `'success'`, or `'error'`."""

    def __init__(self, actual_status: 'SnapshotStatus'):
        self.actual_status = actual_status
        super().__init__(f"Incorrect snapshot status {actual_status!r}, must be 'created' or 'pending'.")

    @classmethod
    def check(cls, status: 'SnapshotStatus') -> None:
        """Check if the status is valid."""
        if status not in {'created', 'pending'}:
            raise cls(status)
