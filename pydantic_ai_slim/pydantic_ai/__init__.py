from importlib.metadata import version

from .agent import Agent
from .exceptions import AgentRunError, ModelRetry, UnexpectedModelBehavior, UsageLimitExceeded, UserError
from .format_examples import format_examples
from .tools import RunContext, Tool

__all__ = (
    'Agent',
    'RunContext',
    'Tool',
    'AgentRunError',
    'ModelRetry',
    'UnexpectedModelBehavior',
    'UsageLimitExceeded',
    'UserError',
    '__version__',
    'format_examples',
)
__version__ = version('pydantic_ai_slim')
