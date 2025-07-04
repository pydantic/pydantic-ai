from importlib.metadata import version as _metadata_version

from .agent import Agent, CallToolsNode, EndStrategy, ModelRequestNode, UserPromptNode, capture_run_messages
from .exceptions import (
    AgentRunError,
    FallbackExceptionGroup,
    ModelHTTPError,
    ModelRetry,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from .format_prompt import format_as_xml
from .messages import AudioUrl, BinaryContent, DocumentUrl, ImageUrl, VideoUrl
from .output import StructuredOutput, NativeOutput, PromptedOutput, TextOutput, ToolOutput, structured
from .tools import RunContext, Tool

__all__ = (
    '__version__',
    # agent
    'Agent',
    'EndStrategy',
    'CallToolsNode',
    'ModelRequestNode',
    'UserPromptNode',
    'capture_run_messages',
    # exceptions
    'AgentRunError',
    'ModelRetry',
    'ModelHTTPError',
    'FallbackExceptionGroup',
    'UnexpectedModelBehavior',
    'UsageLimitExceeded',
    'UserError',
    # messages
    'ImageUrl',
    'AudioUrl',
    'VideoUrl',
    'DocumentUrl',
    'BinaryContent',
    # tools
    'Tool',
    'RunContext',
    # output
    'ToolOutput',
    'StructuredOutput',
    'NativeOutput',
    'PromptedOutput',
    'TextOutput',
    # format_prompt
    'format_as_xml',
    # type aliases
    'structured'
)
__version__ = _metadata_version('pydantic_ai_slim')
