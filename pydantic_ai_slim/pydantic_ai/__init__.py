from importlib.metadata import version

from .agent import Agent
from .exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from .tools import RunContext, Tool

__all__ = 'Agent', 'Tool', 'RunContext', 'ModelRetry', 'UnexpectedModelBehavior', 'UserError', '__version__'
__version__ = version('pydantic_ai_slim')
