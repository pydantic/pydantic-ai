# Observability Reference

Source: `pydantic_ai_slim/pydantic_ai/_instrumentation.py`, `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

## Logfire Integration

The primary observability integration is with [Pydantic Logfire](https://pydantic.dev/logfire):

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

This automatically instruments all agents with OpenTelemetry tracing.

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

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `InstrumentationSettings` | `pydantic_ai.InstrumentationSettings` | Instrumentation config |
