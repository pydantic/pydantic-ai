# Pydantic AI ModelMessage Schema Guide

## Overview

Pydantic AI uses a sophisticated message schema system to handle communication between the application and language models. The core of this system is the `ModelMessage` type hierarchy, which represents all messages that can be sent to or received from a model.

## Core Schema Architecture

### ModelMessage (Top Level)
Location: `pydantic_ai_slim/pydantic_ai/messages.py:861`

```python
ModelMessage = Annotated[Union[ModelRequest, ModelResponse], pydantic.Discriminator('kind')]
```

The `ModelMessage` type is a discriminated union that represents any message in the system. It has two main subtypes:
- `ModelRequest` (kind='request'): Messages sent TO the model
- `ModelResponse` (kind='response'): Messages received FROM the model

### ModelRequest Schema
Location: `pydantic_ai_slim/pydantic_ai/messages.py:661-679`

```python
@dataclass(repr=False)
class ModelRequest:
    parts: list[ModelRequestPart]
    instructions: str | None = None
    kind: Literal['request'] = 'request'
```

A `ModelRequest` contains:
- **parts**: A list of message parts that can include system prompts, user prompts, tool returns, and retry prompts
- **instructions**: Optional instructions for the model
- **kind**: Always 'request' (discriminator)

### ModelRequestPart Types
Location: `pydantic_ai_slim/pydantic_ai/messages.py:655-657`

```python
ModelRequestPart = Annotated[
    Union[SystemPromptPart, UserPromptPart, ToolReturnPart, RetryPromptPart],
    pydantic.Discriminator('part_kind')
]
```

Request parts include:

1. **SystemPromptPart** (`pydantic_ai_slim/pydantic_ai/messages.py:57-85`)
   - `content`: str
   - `timestamp`: datetime
   - `dynamic_ref`: str | None (for dynamic prompts)
   - `part_kind`: 'system-prompt'

2. **UserPromptPart** (`pydantic_ai_slim/pydantic_ai/messages.py:489-526`)
   - `content`: str | Sequence[UserContent]
   - `timestamp`: datetime
   - `part_kind`: 'user-prompt'
   - Can contain multimodal content (images, audio, documents, video)

3. **ToolReturnPart** (`pydantic_ai_slim/pydantic_ai/messages.py:534-582`)
   - `tool_name`: str
   - `content`: Any
   - `tool_call_id`: str
   - `metadata`: Any (optional)
   - `timestamp`: datetime
   - `part_kind`: 'tool-return'

4. **RetryPromptPart** (`pydantic_ai_slim/pydantic_ai/messages.py:588-652`)
   - `content`: list[pydantic_core.ErrorDetails] | str
   - `tool_name`: str | None
   - `tool_call_id`: str
   - `timestamp`: datetime
   - `part_kind`: 'retry-prompt'

### ModelResponse Schema
Location: `pydantic_ai_slim/pydantic_ai/messages.py:787-858`

```python
@dataclass(repr=False)
class ModelResponse:
    parts: list[ModelResponsePart]
    usage: Usage = field(default_factory=Usage)
    model_name: str | None = None
    timestamp: datetime = field(default_factory=_now_utc)
    kind: Literal['response'] = 'response'
    vendor_details: dict[str, Any] | None = field(default=None)
    vendor_id: str | None = None
```

### ModelResponsePart Types
Location: `pydantic_ai_slim/pydantic_ai/messages.py:783`

```python
ModelResponsePart = Annotated[Union[TextPart, ToolCallPart, ThinkingPart], pydantic.Discriminator('part_kind')]
```

Response parts include:

1. **TextPart** (`pydantic_ai_slim/pydantic_ai/messages.py:682-696`)
   - `content`: str
   - `part_kind`: 'text'

2. **ToolCallPart** (`pydantic_ai_slim/pydantic_ai/messages.py:725-780`)
   - `tool_name`: str
   - `args`: str | dict[str, Any] | None
   - `tool_call_id`: str
   - `part_kind`: 'tool-call'

3. **ThinkingPart** (`pydantic_ai_slim/pydantic_ai/messages.py:700-722`)
   - `content`: str
   - `id`: str | None
   - `signature`: str | None (Anthropic-specific)
   - `part_kind`: 'thinking'

## Multimodal Content Types

User prompts can contain various types of media content:

- **ImageUrl** (`pydantic_ai_slim/pydantic_ai/messages.py:256-298`)
- **AudioUrl** (`pydantic_ai_slim/pydantic_ai/messages.py:206-253`)
- **DocumentUrl** (`pydantic_ai_slim/pydantic_ai/messages.py:300-358`)
- **VideoUrl** (`pydantic_ai_slim/pydantic_ai/messages.py:142-203`)
- **BinaryContent** (`pydantic_ai_slim/pydantic_ai/messages.py:360-423`)

All URL types inherit from `FileUrl` abstract base class and support:
- URL referencing
- Force download option
- Vendor-specific metadata
- Media type inference

## Message Flow and Agent Graph States

### Graph Architecture Overview
The agent graph system (`pydantic_ai_slim/pydantic_ai/_agent_graph.py`) implements a finite state machine where different nodes process messages at different stages.

### Key Graph Nodes

1. **UserPromptNode** (`pydantic_ai_slim/pydantic_ai/_agent_graph.py:144-244`)
   - Entry point for user interactions
   - Creates initial `ModelRequest` with system prompts and user input
   - Handles dynamic prompt reevaluation
   - Transitions to → `ModelRequestNode`

2. **ModelRequestNode** (`pydantic_ai_slim/pydantic_ai/_agent_graph.py:279-397`)
   - Sends request to the LLM
   - Manages streaming responses
   - Updates usage statistics
   - Appends `ModelRequest` to message history
   - Receives `ModelResponse` from LLM
   - Transitions to → `CallToolsNode`

3. **CallToolsNode** (`pydantic_ai_slim/pydantic_ai/_agent_graph.py:399-555`)
   - Processes model responses
   - Handles tool calls in parallel
   - Manages text responses
   - Determines next action:
     - If tool calls → process tools and loop back to `ModelRequestNode`
     - If final result → transition to `End` node
     - If text response → validate and potentially end or retry

### Message History Management

The `GraphAgentState` (`pydantic_ai_slim/pydantic_ai/_agent_graph.py:73-92`) maintains:
```python
@dataclasses.dataclass
class GraphAgentState:
    message_history: list[_messages.ModelMessage]
    usage: _usage.Usage
    retries: int
    run_step: int
```

Message history accumulates throughout the conversation:
1. System prompts are added at initialization
2. User prompts create new `ModelRequest` entries
3. Model responses append `ModelResponse` entries
4. Tool returns create new `ModelRequest` with tool results
5. Retry prompts create new `ModelRequest` for validation failures

## Message Processing Flow

### Standard Conversation Flow
```
1. UserPromptNode creates ModelRequest with:
   - SystemPromptPart(s) (if first message)
   - UserPromptPart (user input)

2. ModelRequestNode sends to LLM → receives ModelResponse with:
   - TextPart(s) and/or
   - ToolCallPart(s) and/or
   - ThinkingPart(s)

3. CallToolsNode processes response:
   - If text only → validate and potentially end
   - If tool calls → execute tools, create ToolReturnPart(s)
   - Loop back to step 2 with new ModelRequest containing tool returns
```

### Tool Call Flow
```
ModelResponse with ToolCallPart
    ↓
CallToolsNode executes tool
    ↓
Creates ToolReturnPart or RetryPromptPart
    ↓
New ModelRequest with tool results
    ↓
Back to ModelRequestNode
```

### Validation Failure Flow
```
Invalid response or tool args
    ↓
Create RetryPromptPart with error details
    ↓
New ModelRequest with retry instructions
    ↓
Increment retry counter
    ↓
Back to ModelRequestNode (or fail if max retries exceeded)
```

## TLDR: Agent and Agent Graph Relationship

The `Agent` class (`pydantic_ai_slim/pydantic_ai/agent.py`) is the high-level API that users interact with. It:

1. **Encapsulates the graph**: Creates and manages the underlying `Graph[GraphAgentState, GraphAgentDeps, FinalResult]`
2. **Provides run methods**: `run()`, `run_sync()`, `run_stream()` that execute the graph
3. **Manages configuration**: Model selection, tools, output schemas, validators
4. **Handles initialization**: Sets up the graph nodes based on agent configuration

The agent graph (`_agent_graph.py`) is the execution engine that:
1. **Implements the FSM**: Defines state transitions between nodes
2. **Manages message flow**: Handles the actual message passing and transformation
3. **Executes tools**: Runs tool calls in parallel and manages results
4. **Tracks state**: Maintains conversation history, usage, and retry counts

The relationship is that `Agent` is the configuration and API layer, while `_agent_graph` is the runtime execution layer. The `ModelMessage` schema serves as the common language between all components, ensuring type-safe message passing throughout the system.