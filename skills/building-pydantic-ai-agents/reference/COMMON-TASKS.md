# Common Tasks

Detailed implementation guidance with code examples for common Pydantic AI tasks.

## Add Capabilities to an Agent

Use capabilities to bundle reusable behavior. Multiple capabilities compose automatically.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, WebSearch

# Built-in capabilities: just pass instances
agent = Agent(
    'anthropic:claude-opus-4-6',
    capabilities=[
        Thinking(effort='high'),
        WebSearch(),
    ],
)
```

**WebSearch** auto-detects whether the model supports builtin web search. If so, it uses the native tool; otherwise, it falls back to a local implementation (e.g., DuckDuckGo). Same pattern applies to `WebFetch`, `ImageGeneration`, and `MCP`.

**Docs:** [Capabilities](https://ai.pydantic.dev/capabilities/) · [Built-in Capabilities](https://ai.pydantic.dev/capabilities/#built-in-capabilities)

---

## Intercept Agent Lifecycle with Hooks

Use `Hooks` for lightweight lifecycle interception via decorators. For reusable behavior that combines hooks with tools/instructions, subclass `AbstractCapability` instead.

```python {test="skip" lint="skip"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.models import ModelRequestContext

hooks = Hooks()


@hooks.on.before_model_request
async def log_request(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
    print(f'Sending {len(request_context.messages)} messages')
    return request_context


@hooks.on.before_tool_execute(tools=['send_email'])
async def audit_tool(ctx, *, call, tool_def, args):
    print(f'Executing {call.tool_name}')
    return args


agent = Agent('openai:gpt-5.2', capabilities=[hooks])
```

**Hook types:** `before_run`/`after_run`, `before_model_request`/`after_model_request`, `before_tool_execute`/`after_tool_execute`, `wrap_run`, `wrap_model_request`, `on_run_error`, `on_model_request_error`.

**Docs:** [Hooks](https://ai.pydantic.dev/hooks/) · [Hooking into the Lifecycle](https://ai.pydantic.dev/capabilities/#hooking-into-the-lifecycle)

---

## Define Agents Declaratively with Specs

Use YAML/JSON specs to separate agent configuration from application code. Specs support template strings rendered from dependencies at runtime.

```yaml
# agent.yaml
model: anthropic:claude-opus-4-6
instructions: "You are helping {{user_name}} with research."
capabilities:
  - WebSearch
  - Thinking:
      effort: high
model_settings:
  max_tokens: 8192
```

```python {test="skip"}
from dataclasses import dataclass

from pydantic_ai import Agent


@dataclass
class UserContext:
    user_name: str


agent = Agent.from_file('agent.yaml', deps_type=UserContext)
result = agent.run_sync('Find recent papers on AI safety', deps=UserContext(user_name='Alice'))
```

**Capability spec syntax:** `'WebSearch'` (no args), `{'Thinking': {'effort': 'high'}}` (kwargs), `{'Thinking': 'high'}` (single arg).

**Docs:** [Agent Specs](https://ai.pydantic.dev/agent-spec/) · [Template Strings](https://ai.pydantic.dev/agent-spec/#template-strings)

---

## Enable Thinking Across Providers

Use the unified `Thinking` capability or `thinking` model setting for cross-provider reasoning support.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

# Via capability (recommended for spec compatibility)
agent = Agent('anthropic:claude-opus-4-6', capabilities=[Thinking(effort='high')])

# Via model_settings (equivalent)
agent = Agent('anthropic:claude-opus-4-6', model_settings={'thinking': 'high'})
```

**Effort levels:** `True` (default effort), `False` (disable), `'minimal'`, `'low'`, `'medium'`, `'high'`, `'xhigh'`. Automatically mapped to each provider's native format (Anthropic adaptive thinking, OpenAI reasoning_effort, Google thinking_level, etc.).

**Docs:** [Thinking](https://ai.pydantic.dev/thinking/) · [Unified Thinking Settings](https://ai.pydantic.dev/thinking/#unified-thinking-settings)

---

## Manage Context Size

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

## Show Real-Time Progress

Use `event_stream_handler` with `run()` or `run_stream()` to receive events as they happen.

```python {test="skip"}
from collections.abc import AsyncIterable

from pydantic_ai import Agent, AgentStreamEvent, FunctionToolCallEvent, RunContext

agent = Agent('openai:gpt-5.2')


async def stream_handler(ctx: RunContext, events: AsyncIterable[AgentStreamEvent]):
    async for event in events:
        if isinstance(event, FunctionToolCallEvent):
            print(f'Calling {event.part.tool_name}...')


async def main():
    await agent.run('Do the task', event_stream_handler=stream_handler)
```

**Also use for:** Logging, analytics, debugging, progress bars in UIs.

**Docs:** [Streaming Events and Final Output](https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#streaming-events-and-final-output) · [Streaming All Events](https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#streaming-all-events)

---

## Handle Provider Failures

Use `FallbackModel` to automatically switch providers on 4xx/5xx errors.

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

fallback = FallbackModel(
    OpenAIChatModel('gpt-5.2'),
    AnthropicModel('claude-sonnet-4-6'),
)
agent = Agent(fallback)
```

**Also use for:** Cost optimization (expensive → cheap), rate limit handling, regional failover.

**Docs:** [Fallback Model](https://github.com/pydantic/pydantic-ai/blob/main/docs/models/overview.md#fallback-model) · [Per-Model Settings](https://github.com/pydantic/pydantic-ai/blob/main/docs/models/overview.md#per-model-settings)

---

## Test Agent Behavior

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
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

agent = Agent('openai:gpt-5.2')


# FunctionModel: capture requests, return custom responses
def custom_model(messages, info):
    return ModelResponse(parts=[TextPart(content='mocked response')])


with agent.override(model=FunctionModel(custom_model)):
    result = agent.run_sync('test prompt')
```

**Also use for:** Capturing requests for assertions, simulating errors, testing retries.

**Docs:** [Unit testing with TestModel](https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md#unit-testing-with-testmodel) · [Unit testing with FunctionModel](https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md#unit-testing-with-functionmodel)

---

## Coordinate Multiple Agents

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

## Debug and Validate Agent Behavior

Instrument with Logfire to see exact model requests, tool calls, and validate LLM outputs.

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

# All agent runs now traced — see tool calls, model requests, and outputs in Logfire dashboard
```

**Use for:** Debugging unexpected behavior, validating tool schemas, understanding what's sent to providers, production monitoring.

**Docs:** [Using Logfire](https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#using-logfire) · [Monitoring HTTP Requests](https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#monitoring-http-requests)
