---
name: building-pydantic-ai-agents
description: |
  Python agent framework for building LLM-powered applications. Use for creating
  agents, adding tools, structured output, streaming, testing, and multi-agent systems.
license: MIT
metadata:
  version: "1.0.0"
  author: pydantic
---

# PydanticAI Skill

PydanticAI is a Python agent framework for building production-grade Generative AI applications.
This skill provides patterns, architecture guidance, and tested code examples for building applications with PydanticAI.

## Quick-Start Patterns

### Create a Basic Agent

```python {title="hello_world.py"}
from pydantic_ai import Agent

agent = Agent(
    'anthropic:claude-sonnet-4-5',
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
    'google-gla:gemini-3-flash',
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


agent = Agent('google-gla:gemini-3-flash', output_type=CityLocation)
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
    'openai:gpt-5.2',
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

my_agent = Agent('openai:gpt-5.2', instructions='...')


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
| Get structured output | [Structured Output](https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#structured-output) |
| Inject dependencies | [Dependencies](https://ai.pydantic.dev/dependencies/) |
| Understand RunContext fields | [RunContext](https://ai.pydantic.dev/tools/#runcontext) |
| Choose or configure models | [Models](https://ai.pydantic.dev/models/) |
| Use FallbackModel for resilience | [Fallback Model](https://github.com/pydantic/pydantic-ai/blob/main/docs/models/overview.md#fallback-model) |
| Stream responses | [Streaming Events and Final Output](https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#streaming-events-and-final-output) |
| Work with messages and multimedia | [Message History](https://ai.pydantic.dev/message-history/) |
| Process/filter message history | [Processing Message History](https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#processing-message-history) |
| Summarize long conversations | [Summarize Old Messages](https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#summarize-old-messages) |
| Use MCP servers | [MCP](https://ai.pydantic.dev/mcp/) |
| Build multi-step graphs | [Graph](https://ai.pydantic.dev/graph/) |
| Handle errors and retries | [Exceptions](https://ai.pydantic.dev/api/exceptions/) |
| Combine FallbackModel with retries | [Exceptions](https://ai.pydantic.dev/api/exceptions/) |
| Add observability/tracing | [Using Logfire](https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#using-logfire) |
| Test my agent | [Unit testing with TestModel](https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md#unit-testing-with-testmodel) |
| Enable extended thinking | [Thinking](https://ai.pydantic.dev/thinking/) |
| Evaluate agent performance | [Evals](https://ai.pydantic.dev/evals/) |
| Use embeddings for RAG | [Embeddings](https://ai.pydantic.dev/embeddings/) |
| Use durable execution | [Durable Execution](https://ai.pydantic.dev/durable-execution/) |
| Build multi-agent systems | [Agent Delegation](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#agent-delegation) |
| Implement router/triage pattern | [Programmatic Agent Hand-off](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#programmatic-agent-hand-off) |
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

| Mode | Provider Support | Streaming | Use When |
|------|-----------------|-----------|----------|
| `ToolOutput` | All providers | Yes | Default choice. Works with all providers, supports streaming. Use when you need structured data and don't have provider-specific requirements. |
| `NativeOutput` | OpenAI, Anthropic, Google | Limited | Provider natively forces JSON schema compliance. Use when you want guaranteed schema conformance and don't need tools simultaneously (Gemini limitation). |
| `PromptedOutput` | All providers | Yes | Provider doesn't support tools or JSON mode. Use as fallback for basic providers or when debugging. |
| `TextOutput` | All providers | Yes | Custom parsing required. Use when LLM returns non-JSON structured text (e.g., markdown, YAML, or domain-specific formats). |

### Model Provider Prefixes

| Provider | Prefix | Example |
|----------|--------|---------|
| OpenAI | `openai:` | `openai:gpt-5.2` |
| Anthropic | `anthropic:` | `anthropic:claude-sonnet-4-5` |
| Google (AI Studio) | `google-gla:` | `google-gla:gemini-3-pro-preview` |
| Google (Vertex) | `google-vertex:` | `google-vertex:gemini-3-pro-preview` |
| Groq | `groq:` | `groq:llama-3.3-70b-versatile` |
| Mistral | `mistral:` | `mistral:mistral-large-latest` |
| Cohere | `cohere:` | `cohere:command-r-plus` |
| AWS Bedrock | `bedrock:` | `bedrock:anthropic.claude-sonnet-4-5-v2-0` |
| Azure OpenAI | `azure:` | `azure:gpt-5.2` |
| OpenRouter | `openrouter:` | `openrouter:anthropic/claude-sonnet-4-5` |
| Ollama (local) | `ollama:` | `ollama:llama3.2` |
| Custom Provider | N/A | Use `CustomModel(...)` class |

**Custom Providers:** For providers not listed above, implement `CustomModel`. See [Models](https://ai.pydantic.dev/models/).

### Tool Decorator Comparison

| Decorator | RunContext | Use Case |
|-----------|------------|----------|
| `@agent.tool` | Required first param | Access deps, usage, messages, retry info |
| `@agent.tool_plain` | Not available | Pure functions, no context needed |
| `Tool(fn)` | Auto-detected | Define tools outside agent, pass to constructor |

### When to Use Each Agent Method

| Method | Async | Streaming | Use When |
|--------|-------|-----------|----------|
| `agent.run()` | Yes | No | Standard async. Use for autonomous agents, batch processing, background tasks. Add `event_stream_handler` for real-time UI updates while still running to completion. |
| `agent.run_sync()` | No | No | Sync code required. Use in scripts, CLI tools, Jupyter notebooks, or when async isn't available. |
| `agent.run_stream()` | Yes | Yes | Streaming final output. Use for chatbots where you want text to appear word-by-word. May stop early when text arrives before tool calls complete. |
| `agent.iter()` | Yes | No | Step-by-step control. Use when you need to inspect/modify state between agent nodes, implement human approval for tool calls, or debug agent behavior. |

**Recommendation for Interactive Agents:** Use `agent.run(event_stream_handler=...)` for chatbots and assistants. This streams all events (tool calls, results, output) in real-time while running the agent to completion. See [Streaming All Events](https://ai.pydantic.dev/agents/#streaming-all-events).

## Architecture Overview

**Agent execution flow:**
`Agent.run()` → `UserPromptNode` → `ModelRequestNode` → `CallToolsNode` → (loop or end)

**Key generic types:**

- `Agent[AgentDepsT, OutputDataT]` — dependency type + output type
- `RunContext[AgentDepsT]` — available in tools and system prompts

**Model string format:** `"provider:model-name"` (e.g., `"openai:gpt-5.2"`, `"anthropic:claude-sonnet-4-5"`, `"google-gla:gemini-3-pro-preview"`)

**Output modes:**

- `ToolOutput` — structured data via tool calls (default for Pydantic models)
- `NativeOutput` — provider-specific structured output
- `PromptedOutput` — prompt-based structured extraction
- `TextOutput` — plain text responses

## Key Constraints

- **Python 3.10+** compatibility required
- **Observability**: For production systems, enable Logfire with `logfire.instrument_httpx(capture_all=True)` to see exact HTTP requests sent to model providers — invaluable for debugging tool schema errors, unexpected model behavior, and understanding what's actually being sent to the API
- **Testing**: Use `TestModel` for deterministic tests, `FunctionModel` for custom logic

## Common Tasks

### Manage Context Size

Use `history_processors` to trim or filter messages before each model request.

```python
from pydantic_ai import Agent, ModelMessage

async def keep_recent(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages[-10:] if len(messages) > 10 else messages

agent = Agent('openai:gpt-5.2', history_processors=[keep_recent])
```

**Also use for:** Privacy filtering (remove PII), summarizing old messages, role-based access.

**Docs:** [Processing Message History](https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#processing-message-history) · [Summarize Old Messages](https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#summarize-old-messages)

---

### Show Real-Time Progress

Use `event_stream_handler` with `run()` or `run_stream()` to receive events as they happen.

```python
from collections.abc import AsyncIterable
from pydantic_ai import Agent, AgentStreamEvent, FunctionToolCallEvent, RunContext

agent = Agent('openai:gpt-5.2')

async def stream_handler(ctx: RunContext, events: AsyncIterable[AgentStreamEvent]):
    async for event in events:
        if isinstance(event, FunctionToolCallEvent):
            print(f'Calling {event.part.tool_name}...')

result = await agent.run('Do the task', event_stream_handler=stream_handler)
```

**Also use for:** Logging, analytics, debugging, progress bars in UIs.

**Docs:** [Streaming Events and Final Output](https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#streaming-events-and-final-output) · [Streaming All Events](https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#streaming-all-events)

---

### Handle Provider Failures

Use `FallbackModel` to automatically switch providers on 4xx/5xx errors.

```python
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel

fallback = FallbackModel(
    OpenAIChatModel('gpt-5.2'),
    AnthropicModel('claude-sonnet-4-5'),
)
agent = Agent(fallback)
```

**Also use for:** Cost optimization (expensive → cheap), rate limit handling, regional failover.

**Docs:** [Fallback Model](https://github.com/pydantic/pydantic-ai/blob/main/docs/models/overview.md#fallback-model) · [Per-Model Settings](https://github.com/pydantic/pydantic-ai/blob/main/docs/models/overview.md#per-model-settings)

---

### Test Agent Behavior

Use `TestModel` for fast deterministic tests; `FunctionModel` for custom response logic.

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent('openai:gpt-5.2')

# TestModel: fast, auto-generates valid responses based on schema
with agent.override(model=TestModel()):
    result = agent.run_sync('test prompt')
    assert result.output == 'success (no tool calls)'
```

```python
from pydantic_ai.models.function import FunctionModel
from pydantic_ai import ModelResponse, TextPart

# FunctionModel: capture requests, return custom responses
def custom_model(messages, info):
    return ModelResponse(parts=[TextPart(content='mocked response')])

with agent.override(model=FunctionModel(custom_model)):
    result = agent.run_sync('test prompt')
```

**Also use for:** Capturing requests for assertions, simulating errors, testing retries.

**Docs:** [Unit testing with TestModel](https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md#unit-testing-with-testmodel) · [Unit testing with FunctionModel](https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md#unit-testing-with-functionmodel)

---

### Coordinate Multiple Agents

Use **agent delegation** (via tools) when a child returns results to parent; **output functions** for permanent hand-offs.

```python
from pydantic_ai import Agent, RunContext

parent = Agent('openai:gpt-5.2')
researcher = Agent('openai:gpt-5.2', output_type=str)

@parent.tool
async def research(ctx: RunContext, topic: str) -> str:
    """Delegate research to specialist."""
    result = await researcher.run(f'Research: {topic}', usage=ctx.usage)
    return result.output
```

**Also use for:** Triage/routing, specialist hand-off, graph-based workflows.

**Docs:** [Agent Delegation](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#agent-delegation) · [Programmatic Agent Hand-off](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#programmatic-agent-hand-off)

---

### Debug and Validate Agent Behavior

Instrument with Logfire to see exact model requests, tool calls, and validate LLM outputs.

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

# All agent runs now traced — see tool calls, model requests, and outputs in Logfire dashboard
```

**Use for:** Debugging unexpected behavior, validating tool schemas, understanding what's sent to providers, production monitoring.

**Docs:** [Using Logfire](https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#using-logfire) · [Monitoring HTTP Requests](https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#monitoring-http-requests)
