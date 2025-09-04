from importlib.metadata import version as _metadata_version

from .agent import (
    Agent,
    CallToolsNode,
    EndStrategy,
    InstrumentationSettings,
    ModelRequestNode,
    UserPromptNode,
    capture_run_messages,
)
from .builtin_tools import CodeExecutionTool, UrlContextTool, WebSearchTool, WebSearchUserLocation
from .exceptions import (
    AgentRunError,
    ApprovalRequired,
    CallDeferred,
    FallbackExceptionGroup,
    ModelHTTPError,
    ModelRetry,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from .format_prompt import format_as_xml
from .messages import AudioUrl, BinaryContent, DocumentUrl, ImageUrl, VideoUrl
from .output import NativeOutput, PromptedOutput, StructuredDict, TextOutput, ToolOutput
from .settings import ModelSettings
from .tools import DeferredToolRequests, DeferredToolResults, RunContext, Tool, ToolApproved, ToolDefinition, ToolDenied
from .usage import RequestUsage, RunUsage, UsageLimits

__all__ = (
    '__version__',
    # agent
    'Agent',
    'EndStrategy',
    'CallToolsNode',
    'ModelRequestNode',
    'UserPromptNode',
    'capture_run_messages',
    'InstrumentationSettings',
    # exceptions
    'AgentRunError',
    'CallDeferred',
    'ApprovalRequired',
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
    'ToolDefinition',
    'RunContext',
    'DeferredToolRequests',
    'DeferredToolResults',
    'ToolApproved',
    'ToolDenied',
    # builtin_tools
    'WebSearchTool',
    'WebSearchUserLocation',
    'UrlContextTool',
    'CodeExecutionTool',
    # output
    'ToolOutput',
    'NativeOutput',
    'PromptedOutput',
    'TextOutput',
    'StructuredDict',
    # format_prompt
    'format_as_xml',
    # settings
    'ModelSettings',
    # usage
    'RunUsage',
    'RequestUsage',
    'UsageLimits',
)
__version__ = _metadata_version('pydantic_ai_slim')
