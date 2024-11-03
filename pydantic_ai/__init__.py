from importlib.metadata import version

from .agent import Agent
from .shared import AgentError, CallContext, ModelRetry, UnexpectedModelBehaviour, UserError

__all__ = 'Agent', 'AgentError', 'CallContext', 'ModelRetry', 'UnexpectedModelBehaviour', 'UserError', '__version__'
__version__ = version('pydantic_ai')
