---
name: pydantic-ai
description: Official Claude Code skill for the PydanticAI agent framework
user-invocable: true
---

# PydanticAI Skill

PydanticAI is a Python agent framework for building production-grade Generative AI applications.
This skill provides patterns, architecture guidance, and tested code examples for working with the codebase.

## Workspace Structure

```
pydantic_ai_slim/          # Core framework (minimal dependencies)
  pydantic_ai/
    agent/                 # Agent class and graph nodes
    models/                # Model provider integrations
    toolsets/              # Toolset abstractions
    tools.py               # Tool decorators and definitions
    output.py              # Output types (ToolOutput, NativeOutput, etc.)
    messages.py            # Message types (ModelRequest, ModelResponse, etc.)
    exceptions.py          # Exception hierarchy
    settings.py            # ModelSettings
    _run_context.py        # RunContext dataclass
    mcp.py                 # MCP server integrations
pydantic_graph/            # Graph execution engine
pydantic_evals/            # Evaluation system
examples/                  # Example applications
clai/                      # CLI tool
tests/                     # Test suite with VCR cassettes
docs/                      # MkDocs documentation source
```

## Development Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies (requires uv, pre-commit, deno) |
| `make test` | Run full test suite (100% coverage required) |
| `pre-commit run --all-files` | Run all checks (format, lint, typecheck) |
| `make docs-serve` | Serve docs locally at http://localhost:8000 |
| `uv run pytest tests/test_file.py::test_name -v` | Run a specific test |
| `uv run pytest tests/test_file.py -v -s` | Run test file with debug output |

## Quick-Start Patterns

### Create a Basic Agent

```python {title="hello_world.py"}
from pydantic_ai import Agent

agent = Agent(
    'anthropic:claude-sonnet-4-0',
    instructions='Be concise, reply with one sentence.',
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

### Add Tools to an Agent

```python {title="dice_game.py"}
import random

from pydantic_ai import Agent, RunContext

agent = Agent(
    'google-gla:gemini-2.5-flash',
    deps_type=str,
    instructions=(
        "You're a dice game, you should roll the die and see if the number "
        "you get back matches the user's guess. If so, tell them they're a winner. "
        "Use the player's name in the response."
    ),
)


@agent.tool_plain
def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))


@agent.tool
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps


dice_result = agent.run_sync('My guess is 4', deps='Anne')
print(dice_result.output)
#> Congratulations Anne, you guessed correctly! You're a winner!
```

### Structured Output with Pydantic Models

```python {title="olympics.py" line_length="90"}
from pydantic import BaseModel

from pydantic_ai import Agent


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent('google-gla:gemini-2.5-flash', output_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
#> city='London' country='United Kingdom'
print(result.usage())
#> RunUsage(input_tokens=57, output_tokens=8, requests=1)
```

### Dependency Injection

```python {title="instructions.py"}
from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    'openai:gpt-5',
    deps_type=str,
    instructions="Use the customer's name while replying to them.",
)


@agent.instructions
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."


@agent.instructions
def add_the_date() -> str:
    return f'The date is {date.today()}.'


result = agent.run_sync('What is the date?', deps='Frank')
print(result.output)
#> Hello Frank, the date today is 2032-01-02.
```

### Testing with TestModel

```python {title="test_model_usage.py" call_name="test_my_agent" noqa="I001"}
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

my_agent = Agent('openai:gpt-5', instructions='...')


async def test_my_agent():
    """Unit test for my_agent, to be run by pytest."""
    m = TestModel()
    with my_agent.override(model=m):
        result = await my_agent.run('Testing my agent...')
        assert result.output == 'success (no tool calls)'
    assert m.last_model_request_parameters.function_tools == []
```

## Task Routing Table

| I want to... | Reference file |
|---|---|
| Create or configure agents | [agents.md](references/agents.md) |
| Add tools to an agent | [tools.md](references/tools.md) |
| Compose or filter toolsets | [toolsets.md](references/toolsets.md) |
| Use web search or code execution | [builtin-tools.md](references/builtin-tools.md) |
| Search with DuckDuckGo/Tavily/Exa | [common-tools.md](references/common-tools.md) |
| Get structured output | [output.md](references/output.md) |
| Inject dependencies | [dependencies.md](references/dependencies.md) |
| Understand RunContext fields | [run-context.md](references/run-context.md) |
| Choose or configure models | [models.md](references/models.md) |
| Use FallbackModel for resilience | [models.md](references/models.md) |
| Stream responses | [streaming.md](references/streaming.md) |
| Work with messages and multimedia | [messages.md](references/messages.md) |
| Process/filter message history | [messages.md](references/messages.md) |
| Summarize long conversations | [messages.md](references/messages.md) |
| Use MCP servers | [mcp.md](references/mcp.md) |
| Build multi-step graphs | [graph.md](references/graph.md) |
| Handle errors and retries | [exceptions.md](references/exceptions.md) |
| Combine FallbackModel with retries | [exceptions.md](references/exceptions.md) |
| Add observability/tracing | [observability.md](references/observability.md) |
| Test my agent | [testing.md](references/testing.md) |
| Enable extended thinking | [thinking.md](references/thinking.md) |
| Evaluate agent performance | [evals.md](references/evals.md) |
| Use embeddings for RAG | [embeddings.md](references/embeddings.md) |
| Use durable execution | [durable.md](references/durable.md) |
| Build multi-agent systems | [multi-agent.md](references/multi-agent.md) |
| Implement router/triage pattern | [multi-agent.md](references/multi-agent.md) |
| Require tool approval (human-in-the-loop) | [deferred-tools.md](references/deferred-tools.md) |
| Use images, audio, video, or documents | [input.md](references/input.md) |
| Use advanced tool features | [tools-advanced.md](references/tools-advanced.md) |
| Secure tools with validation/approval | [tools-advanced.md](references/tools-advanced.md) |
| Make direct model requests | [direct.md](references/direct.md) |
| Expose agents as HTTP servers (A2A) | [a2a.md](references/a2a.md) |
| Handle HTTP retries and rate limits | [retries.md](references/retries.md) |
| Use LangChain or ACI.dev tools | [third-party-tools.md](references/third-party-tools.md) |
| Debug common issues | [troubleshooting.md](references/troubleshooting.md) |
| Migrate from deprecated APIs | [migrations.md](references/migrations.md) |
| See advanced real-world examples | [examples-advanced.md](references/examples-advanced.md) |
| Look up an import path | [api-reference.md](references/api-reference.md) |

## Decision Trees

### Choosing a Tool Registration Method

```
Need RunContext (deps, usage, messages)?
├── Yes → Use @agent.tool
└── No → Pure function, no context needed?
    ├── Yes → Use @agent.tool_plain
    └── Tools defined outside agent file?
        ├── Yes → Use tools=[Tool(...)] in constructor
        └── Dynamic tools based on context?
            ├── Yes → Use ToolPrepareFunc
            └── Multiple related tools as a group?
                └── Yes → Use FunctionToolset
```

### Choosing an Output Mode

```
Need structured data with Pydantic validation?
├── Yes → Does provider support native JSON mode?
│   ├── Yes, and you want it → Use NativeOutput(MyModel)
│   └── No, or prefer consistency → Use ToolOutput(MyModel) [default]
└── No → Need custom parsing logic?
    ├── Yes → Use TextOutput(parser_fn)
    └── No → Just plain text?
        └── Yes → Use output_type=str [default]

Dynamic schema at runtime?
└── Yes → Use StructuredDict(**fields)
```

### Choosing a Multi-Agent Pattern

```
Child agent returns result to parent?
├── Yes → Use agent delegation via tools
└── No → Permanent hand-off to specialist?
    ├── Yes → Use output functions
    └── Application code between agents?
        ├── Yes → Use programmatic hand-off
        └── Complex state machine?
            └── Yes → Use Graph-based control
```

### Choosing a Testing Approach

```
Need deterministic, fast tests?
├── Yes → Use TestModel with agent.override()
└── Need specific tool call behavior?
    ├── Yes → Use FunctionModel
    └── Testing against real API (integration)?
        └── Yes → Use pytest-recording with VCR cassettes
```

## Comparison Tables

### Output Mode Comparison

| Mode | Provider Support | Streaming | Validation | Best For |
|------|-----------------|-----------|------------|----------|
| `ToolOutput` | All providers | Yes | Full Pydantic | Default choice, maximum compatibility |
| `NativeOutput` | OpenAI, Anthropic, Google | Limited | Full Pydantic | When provider JSON mode preferred |
| `PromptedOutput` | All providers | Yes | Full Pydantic | Fallback when tools not available |
| `TextOutput` | All providers | Yes | Custom function | Custom parsing, plain text |

### Model Provider Prefixes

| Provider | Prefix | Example |
|----------|--------|---------|
| OpenAI | `openai:` | `openai:gpt-5` |
| Anthropic | `anthropic:` | `anthropic:claude-sonnet-4-5` |
| Google (AI Studio) | `google-gla:` | `google-gla:gemini-2.5-pro` |
| Google (Vertex) | `google-vertex:` | `google-vertex:gemini-2.5-pro` |
| Groq | `groq:` | `groq:llama-3.3-70b-versatile` |
| Mistral | `mistral:` | `mistral:mistral-large-latest` |
| Cohere | `cohere:` | `cohere:command-r-plus` |
| AWS Bedrock | `bedrock:` | `bedrock:anthropic.claude-sonnet-4-5-v2-0` |
| Azure OpenAI | `azure:` | `azure:gpt-5` |
| OpenRouter | `openrouter:` | `openrouter:anthropic/claude-sonnet-4-5` |
| Ollama (local) | `ollama:` | `ollama:llama3.2` |

### Tool Decorator Comparison

| Decorator | RunContext | Use Case |
|-----------|------------|----------|
| `@agent.tool` | Required first param | Access deps, usage, messages, retry info |
| `@agent.tool_plain` | Not available | Pure functions, no context needed |
| `Tool(fn)` | Auto-detected | Define tools outside agent, pass to constructor |

### When to Use Each Agent Method

| Method | Async | Streaming | Iteration | Best For |
|--------|-------|-----------|-----------|----------|
| `agent.run()` | Yes | No | No | Standard async usage |
| `agent.run_sync()` | No | No | No | Scripts, sync contexts |
| `agent.run_stream()` | Yes | Yes | No | Real-time text output |
| `agent.iter()` | Yes | No | Yes | Fine-grained control, debugging |

## Architecture Overview

**Agent execution flow:**
`Agent.run()` → `UserPromptNode` → `ModelRequestNode` → `CallToolsNode` → (loop or end)

**Key generic types:**
- `Agent[AgentDepsT, OutputDataT]` — dependency type + output type
- `RunContext[AgentDepsT]` — available in tools and system prompts

**Model string format:** `"provider:model-name"` (e.g., `"openai:gpt-5"`, `"anthropic:claude-sonnet-4-5"`, `"google:gemini-2.5-pro"`)

**Output modes:**
- `ToolOutput` — structured data via tool calls (default for Pydantic models)
- `NativeOutput` — provider-specific structured output
- `PromptedOutput` — prompt-based structured extraction
- `TextOutput` — plain text responses

## Key Constraints

- **100% test coverage** required on all PRs
- **Python 3.10+** compatibility required
- **Observability**: For production systems, enable Logfire with `logfire.instrument_httpx(capture_all=True)` to see exact HTTP requests sent to model providers — invaluable for debugging tool schema errors, unexpected model behavior, and understanding what's actually being sent to the API
- **Class renames** must include deprecation shim (see `CLAUDE.md`)
- **Documentation references** use backtick + link format: `` [`Agent`][pydantic_ai.agent.Agent] ``
- **Testing**: Use `TestModel` for deterministic tests, `FunctionModel` for custom logic
- **Examples**: Code blocks in docs must use `{title="filename.py"}` format for `pytest-examples`

## Reference Files

| File | Description |
|---|---|
| [agents.md](references/agents.md) | Agent constructor, run methods, instructions, override, WrapperAgent |
| [tools.md](references/tools.md) | Tool decorators, RunContext, ToolPrepareFunc, ModelRetry, docstrings |
| [toolsets.md](references/toolsets.md) | FunctionToolset, CombinedToolset, FilteredToolset, dynamic toolsets |
| [builtin-tools.md](references/builtin-tools.md) | WebSearchTool, CodeExecutionTool, ImageGenerationTool, provider-native tools |
| [common-tools.md](references/common-tools.md) | DuckDuckGo, Tavily, Exa search integrations |
| [output.md](references/output.md) | Output types, validators, union outputs, StructuredDict |
| [dependencies.md](references/dependencies.md) | Dependency injection, RunContext, testing with override |
| [run-context.md](references/run-context.md) | Complete RunContext field reference (deps, retry, tracing, etc.) |
| [models.md](references/models.md) | Model strings, providers, ModelSettings, TestModel, FallbackModel |
| [streaming.md](references/streaming.md) | run_stream(), StreamedRunResult, run_stream_events(), iter() |
| [messages.md](references/messages.md) | ModelRequest/Response, parts, multimedia types, message history |
| [mcp.md](references/mcp.md) | MCP servers (HTTP, SSE, Stdio), FastMCP, sampling, resources |
| [graph.md](references/graph.md) | Graph, BaseNode, End, state management, agent graph internals |
| [exceptions.md](references/exceptions.md) | Exception hierarchy, ModelRetry, deferred tools, error handling |
| [observability.md](references/observability.md) | Logfire integration, OpenTelemetry, InstrumentedModel |
| [testing.md](references/testing.md) | TestModel, FunctionModel, recording, override patterns |
| [thinking.md](references/thinking.md) | Extended thinking, ThinkingPart, provider-specific reasoning |
| [evals.md](references/evals.md) | Evaluation framework, evaluators, LLM judges |
| [embeddings.md](references/embeddings.md) | Embedder, embedding models, vector search patterns |
| [durable.md](references/durable.md) | Temporal, DBOS, Prefect integration |
| [input.md](references/input.md) | ImageUrl, AudioUrl, VideoUrl, DocumentUrl, BinaryContent |
| [deferred-tools.md](references/deferred-tools.md) | DeferredToolRequests, ApprovalRequired, CallDeferred |
| [tools-advanced.md](references/tools-advanced.md) | ToolReturn, Tool.from_schema, ToolPrepareFunc, timeout |
| [multi-agent.md](references/multi-agent.md) | Agent delegation, hand-off, deep agents |
| [direct.md](references/direct.md) | model_request, model_request_sync, low-level API |
| [a2a.md](references/a2a.md) | agent.to_a2a(), FastA2A, inter-agent communication |
| [retries.md](references/retries.md) | AsyncTenacityTransport, RetryConfig, wait_retry_after |
| [third-party-tools.md](references/third-party-tools.md) | LangChain tools, ACI.dev tools |
| [troubleshooting.md](references/troubleshooting.md) | Common mistakes, anti-patterns, debugging tips |
| [migrations.md](references/migrations.md) | Upgrade patterns from deprecated APIs |
| [examples-advanced.md](references/examples-advanced.md) | Real-world examples: RAG, support bot, code assistant |
| [api-reference.md](references/api-reference.md) | Condensed public API with import paths |
