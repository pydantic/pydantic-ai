# API Reference — Public Exports

Source: `pydantic_ai_slim/pydantic_ai/__init__.py`

All items below are importable from `pydantic_ai` unless noted otherwise.

## Agent & Core

| Name | Type | Description |
|------|------|-------------|
| `Agent` | class | Main agent class `Agent[AgentDepsT, OutputDataT]` |
| `EndStrategy` | type alias | `Literal['early', 'exhaustive']` |
| `CallToolsNode` | class | Agent graph node for tool execution |
| `ModelRequestNode` | class | Agent graph node for model requests |
| `UserPromptNode` | class | Agent graph node for user prompts |
| `capture_run_messages` | function | Context manager to capture messages |
| `InstrumentationSettings` | class | OpenTelemetry instrumentation config |

## Embeddings

| Name | Type | Description |
|------|------|-------------|
| `Embedder` | class | Embedding wrapper |
| `EmbeddingModel` | class | Abstract embedding model |
| `EmbeddingSettings` | class | Embedding configuration |
| `EmbeddingResult` | class | Embedding result |

## Exceptions

| Name | Type | Description |
|------|------|-------------|
| `AgentRunError` | exception | Base run error |
| `CallDeferred` | exception | Defer tool execution |
| `ApprovalRequired` | exception | Request human approval |
| `ModelRetry` | exception | Retry tool with feedback |
| `ModelAPIError` | exception | API request failure |
| `ModelHTTPError` | exception | HTTP error response |
| `FallbackExceptionGroup` | exception | All fallbacks failed |
| `IncompleteToolCall` | exception | Truncated tool call |
| `UnexpectedModelBehavior` | exception | Unexpected model output |
| `UsageLimitExceeded` | exception | Token limit exceeded |
| `UserError` | exception | Developer usage error |

## Messages

| Name | Type | Description |
|------|------|-------------|
| `ModelMessage` | type alias | Union of request/response |
| `ModelRequest` | class | Request message container |
| `ModelResponse` | class | Response message container |
| `SystemPromptPart` | class | System prompt in request |
| `UserPromptPart` | class | User prompt in request |
| `ToolReturnPart` | class | Tool return in request |
| `RetryPromptPart` | class | Retry prompt in request |
| `TextPart` | class | Text in response |
| `ToolCallPart` | class | Tool call in response |
| `ThinkingPart` | class | Thinking/reasoning in response |
| `ModelMessagesTypeAdapter` | adapter | JSON serialization for messages |

## Multimedia

| Name | Type | Description |
|------|------|-------------|
| `ImageUrl` | class | Image from URL |
| `BinaryImage` | class | Image from bytes |
| `AudioUrl` | class | Audio from URL |
| `VideoUrl` | class | Video from URL |
| `DocumentUrl` | class | Document from URL |
| `FileUrl` | class | Generic file URL |
| `FilePart` | class | File in response |
| `BinaryContent` | class | Binary content |
| `MultiModalContent` | type alias | Union of all media types |
| `UserContent` | type alias | Valid user prompt content |

## Media Format Types

| Name | Type |
|------|------|
| `ImageFormat` | Literal type for image formats |
| `ImageMediaType` | Literal type for image MIME types |
| `AudioFormat` | Literal type for audio formats |
| `AudioMediaType` | Literal type for audio MIME types |
| `VideoFormat` | Literal type for video formats |
| `VideoMediaType` | Literal type for video MIME types |
| `DocumentFormat` | Literal type for document formats |
| `DocumentMediaType` | Literal type for document MIME types |

## Stream Events

| Name | Type | Description |
|------|------|-------------|
| `AgentStreamEvent` | type alias | Union of all stream events |
| `PartStartEvent` | class | New response part started |
| `PartDeltaEvent` | class | Incremental update |
| `PartEndEvent` | class | Response part complete |
| `FinalResultEvent` | class | Final result available |
| `FunctionToolCallEvent` | class | Tool being called |
| `FunctionToolResultEvent` | class | Tool returned result |
| `HandleResponseEvent` | class | Response being handled |
| `ModelResponseStreamEvent` | class | Raw model stream event |

## Delta Types

| Name | Type | Description |
|------|------|-------------|
| `TextPartDelta` | class | Text chunk |
| `ThinkingPartDelta` | class | Thinking chunk |
| `ToolCallPartDelta` | class | Tool call args chunk |
| `ModelResponsePartDelta` | type alias | Union of deltas |

## Output

| Name | Type | Description |
|------|------|-------------|
| `ToolOutput` | class | Tool-based structured output |
| `NativeOutput` | class | Provider-native structured output |
| `PromptedOutput` | class | Prompt-injected schema extraction |
| `TextOutput` | class | Custom text processing |
| `StructuredDict` | class | Dict-based structured output |

## Tools & Toolsets

| Name | Type | Description |
|------|------|-------------|
| `Tool` | class | Tool wrapper |
| `ToolDefinition` | class | Tool schema + metadata |
| `RunContext` | class | Dependency injection context |
| `DeferredToolRequests` | class | Deferred tool call data |
| `DeferredToolResults` | class | Results for deferred tools |
| `ToolApproved` | class | Approval response |
| `ToolDenied` | class | Denial response |
| `AbstractToolset` | class | Base toolset class |
| `FunctionToolset` | class | Function-based toolset |
| `CombinedToolset` | class | Merge multiple toolsets |
| `FilteredToolset` | class | Filter tools by predicate |
| `PrefixedToolset` | class | Add prefix to tool names |
| `RenamedToolset` | class | Rename specific tools |
| `PreparedToolset` | class | Toolset with prepare function |
| `WrapperToolset` | class | Wrap toolset behavior |
| `ExternalToolset` | class | External tool definitions |
| `ApprovalRequiredToolset` | class | Require approval for tools |
| `ToolsetFunc` | type alias | Function returning toolset |
| `ToolsetTool` | class | Single tool in a toolset |

## Builtin Tools

| Name | Type | Description |
|------|------|-------------|
| `CodeExecutionTool` | class | Execute code |
| `FileSearchTool` | class | Search files |
| `ImageGenerationTool` | class | Generate images |
| `MCPServerTool` | class | MCP server tool |
| `MemoryTool` | class | Memory/knowledge tool |
| `UrlContextTool` | class | URL context (deprecated) |
| `WebFetchTool` | class | Fetch web content |
| `WebSearchTool` | class | Web search |
| `WebSearchUserLocation` | class | User location for web search |

## Profiles & Settings

| Name | Type | Description |
|------|------|-------------|
| `ModelProfile` | class | Model capability profile |
| `ModelProfileSpec` | type alias | Profile specification |
| `DEFAULT_PROFILE` | instance | Default model profile |
| `JsonSchemaTransformer` | class | JSON schema transformer |
| `InlineDefsJsonSchemaTransformer` | class | Inline $defs transformer |
| `ModelSettings` | TypedDict | Model configuration |

## Usage & Run

| Name | Type | Description |
|------|------|-------------|
| `RunUsage` | class | Token usage for a run |
| `RequestUsage` | class | Token usage per request |
| `UsageLimits` | class | Token/request limits |
| `AgentRun` | class | Stateful run from `iter()` |
| `AgentRunResult` | class | Result of `run()`/`run_sync()` |
| `AgentRunResultEvent` | class | Result event |

## Other

| Name | Type | Description |
|------|------|-------------|
| `format_as_xml` | function | Format data as XML string |
| `CachePoint` | class | Cache point marker in messages |
| `FinishReason` | type alias | Model finish reason |
| `BaseToolCallPart` | class | Base for tool call parts |
| `BaseToolReturnPart` | class | Base for tool return parts |
| `BuiltinToolCallPart` | class | Builtin tool call |
| `BuiltinToolReturnPart` | class | Builtin tool return |
| `ToolReturn` | class | Tool return value wrapper |

## Models (from `pydantic_ai.models`)

| Name | Import | Description |
|------|--------|-------------|
| `Model` | `pydantic_ai.models.Model` | Abstract base model class |
| `KnownModelName` | `pydantic_ai.models.KnownModelName` | Literal of all known models |
| `TestModel` | `pydantic_ai.models.test.TestModel` | Deterministic test model |
| `FunctionModel` | `pydantic_ai.models.function.FunctionModel` | Custom function model |
| `FallbackModel` | `pydantic_ai.models.fallback.FallbackModel` | Multi-model fallback |

## MCP (from `pydantic_ai.mcp`)

| Name | Import | Description |
|------|--------|-------------|
| `MCPServerStreamableHTTP` | `pydantic_ai.mcp` | HTTP MCP server |
| `MCPServerSSE` | `pydantic_ai.mcp` | SSE MCP server |
| `MCPServerStdio` | `pydantic_ai.mcp` | Stdio MCP server |
| `load_mcp_servers` | `pydantic_ai.mcp` | Load from config |

## Graph (from `pydantic_graph`)

| Name | Import | Description |
|------|--------|-------------|
| `Graph` | `pydantic_graph.Graph` | Graph definition |
| `BaseNode` | `pydantic_graph.BaseNode` | Abstract node |
| `End` | `pydantic_graph.End` | End signal |
| `GraphRunContext` | `pydantic_graph.GraphRunContext` | Node context |
