---
name: building-pydantic-ai-agents
description: |
  Python agent framework for building LLM-powered applications. Use for creating
  agents, adding tools, capabilities (hooks, thinking, web search), structured output,
  agent specs (YAML/JSON), streaming, testing, and multi-agent systems.
license: MIT
metadata:
  version: "1.0.0"
  author: pydantic
---

# Pydantic AI Skill

Pydantic AI is a Python agent framework for building production-grade Generative AI applications.
This skill provides patterns, architecture guidance, and tested code examples for building applications with Pydantic AI.

## Quick-Start Patterns

### Create a Basic Agent

```python
from pydantic_ai import Agent

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    instructions='Be concise, reply with one sentence.',
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

### Add Tools to an Agent

```python
import random

from pydantic_ai import Agent, RunContext

agent = Agent(
    'google-gla:gemini-3-flash-preview',
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

```python {line_length="90"}
from pydantic import BaseModel

from pydantic_ai import Agent


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent('google-gla:gemini-3-flash-preview', output_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
#> city='London' country='United Kingdom'
print(result.usage())
#> RunUsage(input_tokens=57, output_tokens=8, requests=1)
```

### Dependency Injection

```python
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

```python {call_name="test_my_agent" noqa="I001"}
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

### Use Capabilities

Capabilities are reusable, composable units of agent behavior — bundling tools, hooks, instructions, and model settings.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, WebSearch

agent = Agent(
    'anthropic:claude-opus-4-6',
    instructions='You are a research assistant. Be thorough and cite sources.',
    capabilities=[
        Thinking(effort='high'),
        WebSearch(),
    ],
)
```

### Add Lifecycle Hooks

Use `Hooks` to intercept model requests, tool calls, and runs with decorators — no subclassing needed.

```python {test="skip" lint="skip"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models import ModelRequestContext

hooks = Hooks()


@hooks.on.before_model_request
async def log_request(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
    print(f'Sending {len(request_context.messages)} messages')
    return request_context


agent = Agent('openai:gpt-5.2', capabilities=[hooks])
```

### Define Agent from YAML Spec

Use `Agent.from_file` to load agents from YAML or JSON — no Python agent construction code needed.

```python {test="skip"}
from pydantic_ai import Agent

# agent.yaml:
# model: anthropic:claude-opus-4-6
# instructions: You are a helpful research assistant.
# capabilities:
#   - WebSearch
#   - Thinking:
#       effort: high

agent = Agent.from_file('agent.yaml')
```

## Task Routing Table

| I want to... | Documentation |
|---|---|
| Create or configure agents | [Agents](https://ai.pydantic.dev/agents/) |
| Bundle reusable behavior (tools, hooks, instructions) | [Capabilities](https://ai.pydantic.dev/capabilities/) |
| Intercept model requests, tool calls, or runs | [Hooks](https://ai.pydantic.dev/hooks/) |
| Define agents in YAML/JSON without Python code | [Agent Specs](https://ai.pydantic.dev/agent-spec/) |
| Use template strings in agent instructions | [Template Strings](https://ai.pydantic.dev/agent-spec/#template-strings) |
| Let my agent call external APIs or functions | [Tools](https://ai.pydantic.dev/tools/) |
| Organize or restrict which tools an agent can use | [Toolsets](https://ai.pydantic.dev/toolsets/) |
| Give my agent web search with automatic provider fallback | [WebSearch Capability](https://ai.pydantic.dev/capabilities/#web-search) |
| Give my agent URL fetching with automatic provider fallback | [WebFetch Capability](https://ai.pydantic.dev/capabilities/#web-fetch) |
| Give my agent web search or code execution (builtin tools) | [Built-in Tools](https://ai.pydantic.dev/builtin-tools/) |
| Search with DuckDuckGo/Tavily/Exa | [Common Tools](https://ai.pydantic.dev/common-tools/) |
| Ensure my agent returns data in a specific format | [Structured Output](https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#structured-output) |
| Pass database connections, API clients, or config to tools | [Dependencies](https://ai.pydantic.dev/dependencies/) |
| Access usage stats, message history, or retry count in tools | [RunContext](https://ai.pydantic.dev/tools/#runcontext) |
| Choose or configure models | [Models](https://ai.pydantic.dev/models/) |
| Automatically switch to backup model when primary fails | [Fallback Model](https://github.com/pydantic/pydantic-ai/blob/main/docs/models/overview.md#fallback-model) |
| Show real-time progress as my agent works | [Streaming Events and Final Output](https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#streaming-events-and-final-output) |
| Work with messages and multimedia | [Message History](https://ai.pydantic.dev/message-history/) |
| Reduce token costs by trimming or filtering conversation history | [Processing Message History](https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#processing-message-history) |
| Keep long conversations manageable without losing context | [Summarize Old Messages](https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#summarize-old-messages) |
| Use MCP servers | [MCP](https://ai.pydantic.dev/mcp/) |
| Build multi-step graphs | [Graph](https://ai.pydantic.dev/graph/) |
| Debug a failed agent run or see what went wrong | [Model Errors](https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#model-errors) |
| Make my agent resilient to temporary failures | [Retries](https://ai.pydantic.dev/retries/) |
| Understand why my agent made specific decisions | [Using Logfire](https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#using-logfire) |
| Write deterministic tests for my agent | [Unit testing with TestModel](https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md#unit-testing-with-testmodel) |
| Enable thinking/reasoning across any provider | [Thinking](https://ai.pydantic.dev/thinking/) · [Thinking Capability](https://ai.pydantic.dev/capabilities/#thinking) |
| Systematically verify my agent works correctly | [Evals](https://ai.pydantic.dev/evals/) |
| Use embeddings for RAG | [Embeddings](https://ai.pydantic.dev/embeddings/) |
| Use durable execution | [Durable Execution](https://ai.pydantic.dev/durable-execution/) |
| Have one agent delegate tasks to another | [Agent Delegation](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#agent-delegation) |
| Route requests to different agents based on intent | [Programmatic Agent Hand-off](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#programmatic-agent-hand-off) |
| Require tool approval (human-in-the-loop) | [Deferred Tools](https://ai.pydantic.dev/deferred-tools/) |
| Use images, audio, video, or documents | [Input](https://ai.pydantic.dev/input/) |
| Use advanced tool features | [Advanced Tools](https://ai.pydantic.dev/tools-advanced/) |
| Validate or require approval before tool execution | [Advanced Tools](https://ai.pydantic.dev/tools-advanced/) |
| Call the model without using an agent | [Direct API](https://ai.pydantic.dev/direct/) |
| Expose agents as HTTP servers (A2A) | [A2A](https://ai.pydantic.dev/a2a/) |
| Handle network errors and rate limiting automatically | [Retries](https://ai.pydantic.dev/retries/) |
| Use LangChain or ACI.dev tools | [Third-Party Tools](https://ai.pydantic.dev/third-party-tools/) |
| Publish reusable agent extensions as packages | [Extensibility](https://ai.pydantic.dev/extensibility/) |
| Build custom toolsets, models, or agents | [Extensibility](https://ai.pydantic.dev/extensibility/) |
| Debug common issues | [Troubleshooting](https://ai.pydantic.dev/troubleshooting/) |
| Migrate from deprecated APIs | [Upgrade Guide](https://ai.pydantic.dev/upgrade-guide/) |
| See advanced real-world examples | [Examples](https://ai.pydantic.dev/examples/) |
| Look up an import path | [API Reference](https://ai.pydantic.dev/api/) |

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

### Choosing How to Extend Agent Behavior

```
Need reusable behavior across agents (tools + hooks + instructions)?
├── Yes → Build a custom capability (subclass AbstractCapability)
└── No → Just intercepting lifecycle events?
    ├── Yes → Complex interception needing tools/instructions too?
    │   ├── Yes → Subclass AbstractCapability
    │   └── No → Use Hooks capability with decorators
    └── No → Defining agents from config files?
        ├── Yes → Use Agent.from_file() with YAML/JSON specs
        └── No → Just adding tools?
            ├── Yes → Use @agent.tool or Toolset
            └── Pass args directly to Agent constructor
```

### Choosing a Capability

```
Need model thinking/reasoning?
├── Yes → Use Thinking(effort='high')
└── Need web search?
    ├── Yes → Use WebSearch() (auto-fallback to local)
    └── Need URL fetching?
        ├── Yes → Use WebFetch()
        └── Need MCP servers?
            ├── Yes → Use MCP()
            └── Need lifecycle hooks only?
                ├── Yes → Use Hooks()
                └── Need to filter/modify tool defs per step?
                    └── Yes → Use PrepareTools()
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

| Scenario | Mode |
|----------|------|
| Need structured data and want maximum provider compatibility | `ToolOutput` (default) — works with all providers, supports streaming |
| Want the provider to natively enforce JSON schema compliance | `NativeOutput` — OpenAI, Anthropic, Google only; limited streaming |
| Provider doesn't support tools or JSON mode | `PromptedOutput` — works everywhere as a fallback |
| LLM returns non-JSON structured text (markdown, YAML, domain-specific) | `TextOutput` — custom parsing function |

### Model Provider Prefixes

| Provider | Prefix | Example |
|----------|--------|---------|
| OpenAI | `openai:` | `openai:gpt-5.2` |
| Anthropic | `anthropic:` | `anthropic:claude-sonnet-4-6` |
| Google (AI Studio) | `google-gla:` | `google-gla:gemini-3-pro-preview` |
| Google (Vertex) | `google-vertex:` | `google-vertex:gemini-3-pro-preview` |
| Groq | `groq:` | `groq:llama-3.3-70b-versatile` |
| Mistral | `mistral:` | `mistral:mistral-large-latest` |
| Cohere | `cohere:` | `cohere:command-r-plus-08-2024` |
| AWS Bedrock | `bedrock:` | `bedrock:anthropic.claude-sonnet-4-6` |
| Azure OpenAI | `azure:` | `azure:gpt-5.2` |
| OpenRouter | `openrouter:` | `openrouter:anthropic/claude-sonnet-4-6` |
| Ollama (local) | `ollama:` | `ollama:llama3.2` |
| Custom Provider | N/A | Subclass `Model` or use `OpenAIChatModel` with custom base URL |

**Custom Providers:** For providers not listed above, subclass `Model` or use `OpenAIChatModel` with a custom `base_url` for OpenAI-compatible APIs. See [Models](https://ai.pydantic.dev/models/).

### Tool Decorator Comparison

| Scenario | Decorator |
|----------|-----------|
| Tool needs access to deps, usage stats, messages, or retry info | `@agent.tool` — `RunContext` as required first param |
| Pure function, no agent context needed | `@agent.tool_plain` |
| Tools defined in a separate module or shared across agents | `Tool(fn)` — pass to agent constructor via `tools=[...]` |

### Built-in Capabilities

| Capability | What it provides | Usable in YAML Specs |
|---|---|:---:|
| `Thinking` | Model thinking/reasoning at configurable effort | Yes |
| `Hooks` | Decorator-based lifecycle hook registration | No |
| `WebSearch` | Web search — builtin when supported, local fallback | Yes |
| `WebFetch` | URL fetching — builtin when supported, custom fallback | Yes |
| `ImageGeneration` | Image generation — builtin when supported, custom fallback | Yes |
| `MCP` | MCP server — builtin when supported, direct connection | Yes |
| `PrepareTools` | Filters or modifies tool definitions per step | No |
| `PrefixTools` | Wraps a capability and prefixes its tool names | Yes |
| `BuiltinTool` | Registers a builtin tool with the agent | Yes |
| `Toolset` | Wraps an `AbstractToolset` | No |
| `HistoryProcessor` | Wraps a history processor function | No |

### When to Use Each Agent Method

| Scenario | Method |
|----------|--------|
| Building a chatbot or assistant that shows tool calls, progress, and output in real-time | `agent.run(event_stream_handler=...)` — streams all events while running to completion |
| Running an autonomous agent, batch job, or background task | `agent.run()` |
| Writing a CLI tool, script, or Jupyter notebook (no async) | `agent.run_sync()` |
| Streaming final text word-by-word to a UI | `agent.run_stream()` |
| Inspecting or modifying state between agent steps, human-in-the-loop approval | `agent.iter()` |

See [Streaming All Events](https://ai.pydantic.dev/agents/#streaming-all-events) for `event_stream_handler` details.

## Architecture Overview

**Agent execution flow:**
`Agent.run()` → `UserPromptNode` → `ModelRequestNode` → `CallToolsNode` → (loop or end)

**Key generic types:**

- `Agent[AgentDepsT, OutputDataT]` — dependency type + output type
- `RunContext[AgentDepsT]` — available in tools and system prompts
- `AbstractCapability[AgentDepsT]` — base class for reusable behavior bundles

**Agent construction:**

- **Python:** `Agent(model, instructions=..., tools=..., capabilities=...)`
- **Declarative:** `Agent.from_file('agent.yaml')` or `Agent.from_spec({...})`

**Capabilities** are the primary extension point — they bundle tools, lifecycle hooks, instructions, and model settings into reusable units. Built-in capabilities include `Thinking`, `WebSearch`, `WebFetch`, `Hooks`, `MCP`, and more.

**Lifecycle hooks** (via `Hooks` or `AbstractCapability`) intercept every stage: `before_run` → `before_model_request` → `before_tool_execute` → `after_tool_execute` → `after_model_request` → `after_run`

**Model string format:** `"provider:model-name"` (e.g., `"openai:gpt-5.2"`, `"anthropic:claude-sonnet-4-6"`, `"google-gla:gemini-3-pro-preview"`)

**Output modes:**

- `ToolOutput` — structured data via tool calls (default for Pydantic models)
- `NativeOutput` — provider-specific structured output
- `PromptedOutput` — prompt-based structured extraction
- `TextOutput` — plain text responses

## Key Practices

- **Python 3.10+** compatibility required
- **Observability**: For production systems, enable Logfire with `logfire.instrument_httpx(capture_all=True)` to see exact HTTP requests sent to model providers — invaluable for debugging tool schema errors, unexpected model behavior, and understanding what's actually being sent to the API
- **Testing**: Use `TestModel` for deterministic tests, `FunctionModel` for custom logic

## Common Tasks

Load [Common Tasks Reference](./reference/COMMON-TASKS.md) for detailed implementation guidance with code examples:

| Task | Section |
|---|---|
| Add capabilities (Thinking, WebSearch, etc.) | Add Capabilities to an Agent |
| Intercept model requests and tool calls | Intercept Agent Lifecycle with Hooks |
| Define agents from YAML/JSON config files | Define Agents Declaratively with Specs |
| Enable thinking/reasoning across providers | Enable Thinking Across Providers |
| Trim or filter conversation history | Manage Context Size |
| Stream events and show real-time progress | Show Real-Time Progress |
| Auto-switch providers on failure | Handle Provider Failures |
| Write deterministic tests | Test Agent Behavior |
| Delegate tasks between agents | Coordinate Multiple Agents |
| Instrument with Logfire for debugging | Debug and Validate Agent Behavior |
