"""An Amazon Bedrock AgentCore Code Interpreter session as a Pydantic AI workspace backend."""

from ._backend import AgentCoreSandboxBackend, destroy_session

__all__ = ('AgentCoreSandboxBackend', 'destroy_session')
