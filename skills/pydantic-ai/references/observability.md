# Observability Reference

Source: `pydantic_ai_slim/pydantic_ai/_instrumentation.py`, `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

## Why Observability Matters

Agent systems are inherently non-deterministic — the same prompt can produce different tool calls, outputs, and errors. In production, observability is essential for:

- **Debugging failures**: Trace exactly what the model returned, which tools were called, and where things went wrong
- **Understanding behavior**: See the full conversation flow, token usage, and decision points
- **Cost optimization**: Monitor token consumption across runs to identify expensive patterns
- **Performance tuning**: Identify slow tools, retry storms, and model latency issues

## Logfire Integration

The primary observability integration is with [Pydantic Logfire](https://pydantic.dev/logfire), which provides full tracing of agent runs with a purpose-built UI for exploring AI system behavior:

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

This automatically instruments all agents with OpenTelemetry tracing. Every agent run, model request, and tool call is captured with full context.

## HTTP Request Monitoring

For the deepest visibility into what's actually being sent to model providers, add HTTP instrumentation:

```python
import logfire
from pydantic_ai import Agent

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)  # See exact HTTP requests

agent = Agent('anthropic:claude-sonnet-4-5')
result = agent.run_sync('Hello')
# Check Logfire to see the exact JSON payload sent to the API
```

With `capture_all=True`, you can inspect the raw HTTP request body — the actual JSON being sent to OpenAI, Anthropic, or any other provider. This reveals:

- **Tool schemas** — see the exact JSON schema generated for each tool
- **Message formatting** — how system prompts and user messages are structured
- **Request parameters** — temperature, max_tokens, and other settings
- **Response payloads** — the full model response before parsing

This is invaluable when debugging issues like "tool schema too complex" errors — you can see exactly what schema is being sent and why the provider rejected it.

## InstrumentationSettings

Fine-grained control over instrumentation:

```python
from pydantic_ai import Agent, InstrumentationSettings

agent = Agent(
    'openai:gpt-4o',
    instrument=InstrumentationSettings(
        event_mode='logs',  # 'logs' | 'attributes'
    ),
)
```

Or enable with a boolean:

```python
agent = Agent('openai:gpt-4o', instrument=True)
```

## Per-Agent Instrumentation

```python
# Enable on specific agent
agent = Agent('openai:gpt-4o', instrument=True)

# Override at runtime
with agent.override(instrument=InstrumentationSettings()):
    result = agent.run_sync('prompt')
```

## OpenTelemetry Spans

When instrumentation is enabled, the following spans are created:

- **Agent run span** — covers the entire `run()`/`run_sync()` call
- **Model request span** — each request to the LLM
- **Tool call span** — each tool execution

## InstrumentedModel

Wrap any model with instrumentation:

```python
from pydantic_ai.models.instrumented import InstrumentedModel

model = InstrumentedModel('openai:gpt-4o')
agent = Agent(model)
```

## Production Recommendations

For production deployments:

1. **Always enable instrumentation** — the overhead is minimal and the debugging value is immense
2. **Use Logfire** — purpose-built for AI observability with features like conversation replay and token analytics
3. **Set meaningful agent names** — `Agent('openai:gpt-4o', name='customer-support')` for easier filtering in traces
4. **Monitor usage** — track `result.usage()` to catch runaway token consumption early

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `InstrumentationSettings` | `pydantic_ai.InstrumentationSettings` | Instrumentation config |
