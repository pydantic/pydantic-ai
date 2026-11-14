"""AgentCore sandbox capability: gives agents an isolated AWS-hosted computer to work in.

`AgentCoreSandbox` is the supported entry point; build an agent with it and add tools that use the
workspace, such as `Coder`. `AgentCoreSandboxBackend` is the Amazon Bedrock AgentCore Code
Interpreter implementation of Pydantic AI's workspace backend protocol, public for applications
that want to start or attach to a session themselves and pass it to a run as `workspace=`.
"""

from pydantic_ai_harness.agentcore_sandbox._backend import AgentCoreSandboxBackend
from pydantic_ai_harness.agentcore_sandbox._capability import AgentCoreSandbox

__all__ = [
    'AgentCoreSandbox',
    'AgentCoreSandboxBackend',
]
