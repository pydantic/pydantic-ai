from importlib.metadata import version as _metadata_version
from typing import Any

from ._template import TemplateStr
from .agent import (
    Agent,
    AgentModelSettings,
    CallToolsNode,
    EndStrategy,
    InstrumentationSettings,
    ModelRequestNode,
    UserPromptNode,
    capture_run_messages,
)
from .agent.spec import AgentSpec
from .capabilities import AgentCapability, CapabilityFunc
from .concurrency import (
    AbstractConcurrencyLimiter,
    AnyConcurrencyLimit,
    ConcurrencyLimit,
    ConcurrencyLimiter,
)
from .embeddings import (
    Embedder,
    EmbeddingModel,
    EmbeddingResult,
    EmbeddingSettings,
)
from .exceptions import (
    AgentRunError,
    ApprovalRequired,
    CallDeferred,
    ConcurrencyLimitExceeded,
    FallbackExceptionGroup,
    IncompleteToolCall,
    ModelAPIError,
    ModelHTTPError,
    ModelRetry,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from .format_prompt import format_as_xml
from .messages import (
    AgentStreamEvent,
    AudioFormat,
    AudioMediaType,
    AudioUrl,
    BaseToolCallPart,
    BaseToolReturnPart,
    BinaryContent,
    BinaryImage,
    CachePoint,
    CompactionPart,
    DocumentFormat,
    DocumentMediaType,
    DocumentUrl,
    FilePart,
    FileUrl,
    FinalResultEvent,
    FinishReason,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    HandleResponseEvent,
    ImageFormat,
    ImageMediaType,
    ImageUrl,
    InstructionPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelResponsePartDelta,
    ModelResponseState,
    ModelResponseStreamEvent,
    MultiModalContent,
    NativeToolCallPart,
    NativeToolReturnPart,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallEvent,
    ToolCallPart,
    ToolCallPartDelta,
    ToolResultEvent,
    ToolReturn,
    ToolReturnPart,
    UploadedFile,
    UserContent,
    UserPromptPart,
    VideoFormat,
    VideoMediaType,
    VideoUrl,
)
from .models import ModelRequestContext
from .models.concurrency import ConcurrencyLimitedModel, limit_model_concurrency
from .native_tools import (
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    MCPServerTool,
    MemoryTool,
    UrlContextTool,  # pyright: ignore[reportDeprecated]
    WebFetchTool,
    WebSearchTool,
    WebSearchUserLocation,
    XSearchTool,
)
from .output import NativeOutput, PromptedOutput, StructuredDict, TextOutput, ToolOutput
from .profiles import (
    DEFAULT_PROFILE,
    InlineDefsJsonSchemaTransformer,
    JsonSchemaTransformer,
    ModelProfile,
    ModelProfileSpec,
)
from .result import AgentEventStream
from .run import AgentRun, AgentRunResult, AgentRunResultEvent
from .settings import ModelSettings, ToolChoice, ToolOrOutput
from .tools import (
    AgentNativeTool,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    Tool,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
)
from .toolsets import (
    AbstractToolset,
    AgentToolset,
    ApprovalRequiredToolset,
    CombinedToolset,
    DeferredLoadingToolset,
    ExternalToolset,
    FilteredToolset,
    FunctionToolset,
    IncludeReturnSchemasToolset,
    PrefixedToolset,
    PreparedToolset,
    RenamedToolset,
    SetMetadataToolset,
    ToolsetFunc,
    ToolsetTool,
    WrapperToolset,
)
from .usage import RequestUsage, RunUsage, UsageLimits

__all__ = (
    '__version__',
    # agent
    'Agent',
    'AgentModelSettings',
    'AgentSpec',
    'EndStrategy',
    'CallToolsNode',
    'ModelRequestNode',
    'UserPromptNode',
    'capture_run_messages',
    'InstrumentationSettings',
    # embeddings
    'Embedder',
    'EmbeddingModel',
    'EmbeddingSettings',
    'EmbeddingResult',
    # concurrency
    'AbstractConcurrencyLimiter',
    'AnyConcurrencyLimit',
    'ConcurrencyLimit',
    'ConcurrencyLimitedModel',
    'ConcurrencyLimiter',
    'limit_model_concurrency',
    # exceptions
    'AgentRunError',
    'CallDeferred',
    'ApprovalRequired',
    'ConcurrencyLimitExceeded',
    'ModelRetry',
    'ModelAPIError',
    'ModelHTTPError',
    'FallbackExceptionGroup',
    'IncompleteToolCall',
    'SkipModelRequest',
    'SkipToolExecution',
    'SkipToolValidation',
    'UnexpectedModelBehavior',
    'UsageLimitExceeded',
    'UserError',
    # messages
    'AgentStreamEvent',
    'AudioFormat',
    'AudioMediaType',
    'AudioUrl',
    'BaseToolCallPart',
    'BaseToolReturnPart',
    'BinaryContent',
    'NativeToolCallPart',
    'NativeToolReturnPart',
    'CachePoint',
    'CompactionPart',
    'DocumentFormat',
    'DocumentMediaType',
    'DocumentUrl',
    'FileUrl',
    'FilePart',
    'FinalResultEvent',
    'FinishReason',
    'FunctionToolCallEvent',
    'FunctionToolResultEvent',
    'HandleResponseEvent',
    'ImageFormat',
    'ImageMediaType',
    'ImageUrl',
    'BinaryImage',
    'InstructionPart',
    'ModelMessage',
    'ModelMessagesTypeAdapter',
    'ModelRequest',
    'ModelRequestPart',
    'ModelResponse',
    'ModelResponsePart',
    'ModelResponsePartDelta',
    'ModelResponseState',
    'ModelResponseStreamEvent',
    'MultiModalContent',
    'OutputToolCallEvent',
    'OutputToolResultEvent',
    'PartDeltaEvent',
    'PartEndEvent',
    'PartStartEvent',
    'RetryPromptPart',
    'SystemPromptPart',
    'TextContent',
    'TextPart',
    'TextPartDelta',
    'ThinkingPart',
    'ThinkingPartDelta',
    'ToolCallEvent',
    'ToolCallPart',
    'ToolCallPartDelta',
    'ToolResultEvent',
    'ToolReturn',
    'ToolReturnPart',
    'UploadedFile',
    'UserContent',
    'UserPromptPart',
    'VideoFormat',
    'VideoMediaType',
    'VideoUrl',
    # profiles
    'ModelProfile',
    'ModelProfileSpec',
    'DEFAULT_PROFILE',
    'InlineDefsJsonSchemaTransformer',
    'JsonSchemaTransformer',
    # tools
    'AgentNativeTool',
    'Tool',
    'ToolDefinition',
    'RunContext',
    'DeferredToolRequests',
    'DeferredToolResults',
    'ToolApproved',
    'ToolDenied',
    # toolsets
    'AbstractToolset',
    'AgentToolset',
    'ApprovalRequiredToolset',
    'CombinedToolset',
    'DeferredLoadingToolset',
    'ExternalToolset',
    'FilteredToolset',
    'FunctionToolset',
    'IncludeReturnSchemasToolset',
    'PrefixedToolset',
    'PreparedToolset',
    'RenamedToolset',
    'SetMetadataToolset',
    'ToolsetFunc',
    'ToolsetTool',
    'WrapperToolset',
    # builtin_tools
    'CodeExecutionTool',
    'FileSearchTool',
    'ImageGenerationTool',
    'MCPServerTool',
    'MemoryTool',
    'UrlContextTool',
    'WebFetchTool',
    'WebSearchTool',
    'WebSearchUserLocation',
    'XSearchTool',
    # capabilities
    'AgentCapability',
    'CapabilityFunc',
    # output
    'ToolOutput',
    'NativeOutput',
    'PromptedOutput',
    'TextOutput',
    'StructuredDict',
    # template
    'TemplateStr',
    # format_prompt
    'format_as_xml',
    # models
    'ModelRequestContext',
    # settings
    'ModelSettings',
    'ToolChoice',
    'ToolOrOutput',
    # usage
    'RunUsage',
    'RequestUsage',
    'UsageLimits',
    # run
    'AgentRun',
    'AgentRunResult',
    'AgentRunResultEvent',
    # result
    'AgentEventStream',
)
__version__ = _metadata_version('pydantic_ai_slim')


# Deprecated top-level aliases for names renamed in the built-in → native tools rename.
# Importing these from `pydantic_ai` continues to work in 1.x with a deprecation
# warning that points at the new name.
_BUILTIN_TO_NATIVE_TOP_LEVEL: dict[str, str] = {
    'BuiltinToolCallPart': 'NativeToolCallPart',
    'BuiltinToolReturnPart': 'NativeToolReturnPart',
    'AgentBuiltinTool': 'AgentNativeTool',
}


def __getattr__(name: str) -> Any:
    if name in _BUILTIN_TO_NATIVE_TOP_LEVEL:
        import warnings

        from ._warnings import PydanticAIDeprecationWarning

        new_name = _BUILTIN_TO_NATIVE_TOP_LEVEL[name]
        warnings.warn(
            f'`pydantic_ai.{name}` is deprecated, use `pydantic_ai.{new_name}` instead.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        import pydantic_ai as _self

        return getattr(_self, new_name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
