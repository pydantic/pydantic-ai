# GraphRunContext vs RunContext: Understanding State and Deps

## Overview

You're correct that RunContext in Pydantic AI is the internal implementation used within the agent's execution loop. It's built on top of Pydantic Graph's GraphRunContext but serves a more specific purpose tailored to agent operations.

## Key Architectural Relationship

```
Pydantic Graph (Generic Framework)
├── GraphRunContext[StateT, DepsT]
│   ├── state: StateT (user-defined)
│   └── deps: DepsT (user-defined)
│
Pydantic AI (Agent Implementation)
├── GraphRunContext[GraphAgentState, GraphAgentDeps]
│   ├── state: GraphAgentState (Pydantic AI internals)
│   └── deps: GraphAgentDeps (wraps user deps + internal deps)
│
└── RunContext[AgentDepsT] (exposed to tools)
    └── deps: AgentDepsT (user-defined deps only)
```

## State Management

### In Pydantic Graph
- **State is user-defined**: When creating your own graph, you define the state type and manage it directly
- State is mutable and passed through nodes via `GraphRunContext.state`
- You have full control over what goes into state and how it changes

### In Pydantic AI
- **State is internally managed**: `GraphAgentState` is a Pydantic AI internal dataclass containing:
  - `message_history`: List of model messages
  - `usage`: Token usage tracking
  - `retries`: Retry counter for output validation
  - `run_step`: Current execution step counter
- Users never directly interact with or modify this state
- The state is managed by Pydantic AI's internal nodes (UserPromptNode, ModelRequestNode, CallToolsNode)

## Deps Management

### In Pydantic Graph
- **Deps are user-defined**: You define the dependency type and pass it when running the graph
- Deps are immutable throughout the graph execution
- Accessed via `GraphRunContext.deps`

### In Pydantic AI
- **Deps have dual nature**:
  
  1. **GraphAgentDeps** (internal wrapper):
     - Contains `user_deps` (your actual dependencies)
     - Plus internal dependencies like model, tracer, output specs, tool manager, etc.
     - This is what's stored in `GraphRunContext.deps`

  2. **RunContext.deps** (exposed to tools):
     - Only contains your `user_deps` (type `AgentDepsT`)
     - This is what tools see when they receive `RunContext`
     - Clean separation between user deps and internal machinery

## Tool Usage Pattern

When you define a tool with `@agent.tool`:

```python
@agent.tool
def my_tool(ctx: RunContext[MyDeps], param: str) -> str:
    # ctx.deps is MyDeps - your custom dependencies
    # ctx also provides:
    #   - model: The current model being used
    #   - usage: Token usage statistics
    #   - messages: Conversation history
    #   - prompt: Original user prompt
    return ctx.deps.some_service.process(param)
```

Behind the scenes:
1. Pydantic AI's `CallToolsNode` receives `GraphRunContext[GraphAgentState, GraphAgentDeps[MyDeps, Output]]`
2. It calls `build_run_context()` which extracts:
   - `user_deps` from `GraphAgentDeps` → becomes `RunContext.deps`
   - Plus other relevant info (model, usage, messages, etc.)
3. Your tool receives this simplified `RunContext[MyDeps]`

## Key Insights

1. **State vs Deps Philosophy**:
   - In Pydantic Graph: Both state and deps are user-controlled
   - In Pydantic AI: State is internal (agent mechanics), deps are user-controlled (your dependencies)

2. **Why This Design**:
   - Pydantic AI needs to manage complex agent state (messages, usage, retries)
   - But tools only need access to user dependencies, not internal agent state
   - The `RunContext` provides a clean API that exposes what tools need without internal complexity

3. **Practical Implications**:
   - You define `deps_type` when creating an Agent - these are your service dependencies
   - You pass `deps` when calling `agent.run()` - actual instances of your dependencies
   - Tools receive these deps via `RunContext.deps` - clean, typed access
   - You never see or manage the internal `GraphAgentState` - Pydantic AI handles it

## Example Flow

```python
# User defines deps
class MyDeps:
    db: Database
    api_client: APIClient

# Create agent with deps type
agent = Agent('my-agent', deps_type=MyDeps)

# Tool receives user deps via RunContext
@agent.tool
def fetch_data(ctx: RunContext[MyDeps], query: str):
    return ctx.deps.db.query(query)  # Clean access to user deps

# Run with actual deps
result = agent.run('fetch user data', deps=MyDeps(db=db, api=api))
```

Internally:
- `GraphAgentDeps` wraps your `MyDeps` plus internal stuff
- `GraphAgentState` tracks messages, usage, retries (you never see this)
- `RunContext` exposes just your `MyDeps` to tools (plus useful context like messages)

This separation ensures tools have a clean, focused API while Pydantic AI maintains full control over the agent's execution state.