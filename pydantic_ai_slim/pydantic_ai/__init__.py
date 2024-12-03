from importlib.metadata import version

from .agent import Agent
from .dependencies import RunContext
from .exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from .tool import Tool

__all__ = 'Agent', 'Tool', 'RunContext', 'ModelRetry', 'UnexpectedModelBehavior', 'UserError', '__version__'
__version__ = version('pydantic_ai_slim')
