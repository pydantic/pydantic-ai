from importlib.metadata import version

from .agent import Agent
from .call_typing import CallContext
from .exceptions import AgentError, ModelRetry, UnexpectedModelBehaviour, UserError

__all__ = 'Agent', 'AgentError', 'CallContext', 'ModelRetry', 'UnexpectedModelBehaviour', 'UserError', '__version__'
__version__ = version('pydantic_ai')
