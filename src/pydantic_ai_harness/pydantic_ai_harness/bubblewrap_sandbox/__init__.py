"""Bubblewrapped workspace capability: runs another workspace's commands in a Linux bubblewrap wrapper on its host.

`BubblewrappedWorkspace` is the supported entry point: wrap a workspace capability such as
`SSHWorkspace` or `LocalWorkspace` in it. `BubblewrapWorkspace` is the workspace wrapper itself,
public for applications that build their own workspace and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.bubblewrap_sandbox._capability import BubblewrappedWorkspace
from pydantic_ai_harness.bubblewrap_sandbox._workspace import BubblewrapWorkspace

__all__ = [
    'BubblewrappedWorkspace',
    'BubblewrapWorkspace',
]
