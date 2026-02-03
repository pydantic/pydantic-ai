---
name: pydantic-ai
description: Build Python AI agents using the PydanticAI framework with type-safe tools, structured outputs, and dependency injection. Use when creating LLM agents, adding tools to agents, configuring model providers, implementing structured output with Pydantic models, testing with TestModel, or building multi-agent systems. Provides patterns for Agent creation, tool registration (@agent.tool, @agent.tool_plain), RunContext usage, streaming, message history, and observability with Logfire.
license: MIT
compatibility: Requires Python 3.10+
---

# PydanticAI Skill

PydanticAI is a Python agent framework for building production-grade Generative AI applications.
This skill provides patterns, architecture guidance, and tested code examples for building applications with PydanticAI.

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

| I want to... | Documentation |
|---|---|
| Create or configure agents | [Agents](https://ai.pydantic.dev/agents/) |
| Add tools to an agent | [Tools](https://ai.pydantic.dev/tools/) |
| Compose or filter toolsets | [Toolsets](https://ai.pydantic.dev/toolsets/) |
| Use web search or code execution | [Built-in Tools](https://ai.pydantic.dev/builtin-tools/) |
| Search with DuckDuckGo/Tavily/Exa | [Common Tools](https://ai.pydantic.dev/common-tools/) |
| Get structured output | [Output](https://ai.pydantic.dev/output/) |
| Inject dependencies | [Dependencies](https://ai.pydantic.dev/dependencies/) |
| Understand RunContext fields | [RunContext](https://ai.pydantic.dev/tools/#runcontext) |
| Choose or configure models | [Models](https://ai.pydantic.dev/models/) |
| Use FallbackModel for resilience | [Models](https://ai.pydantic.dev/models/) |
| Stream responses | [Streaming](https://ai.pydantic.dev/agents/#streaming) |
| Work with messages and multimedia | [Message History](https://ai.pydantic.dev/message-history/) |
| Process/filter message history | [Message History](https://ai.pydantic.dev/message-history/) |
| Summarize long conversations | [Message History](https://ai.pydantic.dev/message-history/) |
| Use MCP servers | [MCP](https://ai.pydantic.dev/mcp/) |
| Build multi-step graphs | [Graph](https://ai.pydantic.dev/graph/) |
| Handle errors and retries | [Exceptions](https://ai.pydantic.dev/api/exceptions/) |
| Combine FallbackModel with retries | [Exceptions](https://ai.pydantic.dev/api/exceptions/) |
| Add observability/tracing | [Logfire](https://ai.pydantic.dev/logfire/) |
| Test my agent | [Testing](https://ai.pydantic.dev/testing/) |
| Enable extended thinking | [Thinking](https://ai.pydantic.dev/thinking/) |
| Evaluate agent performance | [Evals](https://ai.pydantic.dev/evals/) |
| Use embeddings for RAG | [Embeddings](https://ai.pydantic.dev/embeddings/) |
| Use durable execution | [Durable Execution](https://ai.pydantic.dev/durable-execution/) |
| Build multi-agent systems | [Multi-Agent](https://ai.pydantic.dev/multi-agent-applications/) |
| Implement router/triage pattern | [Multi-Agent](https://ai.pydantic.dev/multi-agent-applications/) |
| Require tool approval (human-in-the-loop) | [Deferred Tools](https://ai.pydantic.dev/deferred-tools/) |
| Use images, audio, video, or documents | [Input](https://ai.pydantic.dev/input/) |
| Use advanced tool features | [Advanced Tools](https://ai.pydantic.dev/tools-advanced/) |
| Secure tools with validation/approval | [Advanced Tools](https://ai.pydantic.dev/tools-advanced/) |
| Make direct model requests | [Direct API](https://ai.pydantic.dev/direct/) |
| Expose agents as HTTP servers (A2A) | [A2A](https://ai.pydantic.dev/a2a/) |
| Handle HTTP retries and rate limits | [Retries](https://ai.pydantic.dev/retries/) |
| Use LangChain or ACI.dev tools | [Third-Party Tools](https://ai.pydantic.dev/third-party-tools/) |
| Debug common issues | [Troubleshooting](https://ai.pydantic.dev/troubleshooting/) |
| Migrate from deprecated APIs | [Upgrade Guide](https://ai.pydantic.dev/upgrade-guide/) |
| See advanced real-world examples | [Examples](https://ai.pydantic.dev/examples/) |
| Look up an import path | [API Reference](https://ai.pydantic.dev/api/) |
| Extend framework behavior | [Toolsets](https://ai.pydantic.dev/toolsets/) |
| Build custom toolsets or models | [Toolsets](https://ai.pydantic.dev/toolsets/) |

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

- **Python 3.10+** compatibility required
- **Observability**: For production systems, enable Logfire with `logfire.instrument_httpx(capture_all=True)` to see exact HTTP requests sent to model providers — invaluable for debugging tool schema errors, unexpected model behavior, and understanding what's actually being sent to the API
- **Testing**: Use `TestModel` for deterministic tests, `FunctionModel` for custom logic
