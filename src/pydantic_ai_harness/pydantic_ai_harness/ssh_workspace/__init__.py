"""SSH workspace capability: runs the agent's commands and file edits on a remote host over SSH.

`SSHWorkspace` is the supported entry point; build an agent with it and add tools that use the
workspace, such as `Coder`. `SSHWorkspaceBackend` is the workspace backend itself, public for
applications that want to build one and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.ssh_workspace._backend import SSHWorkspaceBackend
from pydantic_ai_harness.ssh_workspace._capability import SSHWorkspace

__all__ = [
    'SSHWorkspace',
    'SSHWorkspaceBackend',
]
