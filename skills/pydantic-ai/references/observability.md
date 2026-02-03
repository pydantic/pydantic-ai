# Observability Reference

Source: `docs/logfire.md`, `pydantic_ai_slim/pydantic_ai/models/instrumented.py`

## Why Observability Matters

Agent systems are inherently non-deterministic - the same prompt can produce different tool calls, outputs, and errors. In production, observability is essential for:

- **Debugging failures**: Trace exactly what the model returned, which tools were called, and where things went wrong
- **Understanding behavior**: See the full conversation flow, token usage, and decision points
- **Cost optimization**: Monitor token consumption across runs to identify expensive patterns
- **Performance tuning**: Identify slow tools, retry storms, and model latency issues

## Logfire Integration

The primary observability integration is with [Pydantic Logfire](https://pydantic.dev/logfire), which provides full tracing of agent runs with a purpose-built UI for exploring AI system behavior.

### Setup

```bash
# Install with logfire support
pip install pydantic-ai  # or pydantic-ai-slim[logfire]

# Authenticate and configure project
logfire auth
logfire projects new  # or: logfire projects use <existing>
```

### Basic Instrumentation

```python
import logfire

from pydantic_ai import Agent

logfire.configure()  # Reads config from .logfire directory
logfire.instrument_pydantic_ai()  # Enable agent instrumentation

agent = Agent('openai:gpt-5', instructions='Be concise.')
result = agent.run_sync('Where does "hello world" come from?')
# Trace visible in Logfire UI
```

### Per-Agent Instrumentation

Enable instrumentation on specific agents:

```python
from pydantic_ai import Agent, InstrumentationSettings

# Enable with boolean
agent = Agent('openai:gpt-5', instrument=True)

# Enable with settings
agent = Agent(
    'openai:gpt-5',
    instrument=InstrumentationSettings(event_mode='logs'),
)

# Override at runtime
with agent.override(instrument=InstrumentationSettings()):
    result = agent.run_sync('prompt')
```

### Global Instrumentation

Instrument all agents without individual configuration:

```python
from pydantic_ai import Agent

# Before creating any agents
Agent.instrument_all()

# Or with custom settings
Agent.instrument_all(InstrumentationSettings(include_content=False))
```

## HTTP Request Monitoring

For the deepest visibility into what's being sent to model providers, add HTTP instrumentation:

```python
import logfire

from pydantic_ai import Agent

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)  # See exact HTTP requests

agent = Agent('anthropic:claude-sonnet-4-5')
result = agent.run_sync('Hello')
```

With `capture_all=True`, you can inspect the raw HTTP request body - the actual JSON being sent to any provider. This reveals:

- **Tool schemas** - the exact JSON schema generated for each tool
- **Message formatting** - how system prompts and user messages are structured
- **Request parameters** - temperature, max_tokens, and other settings
- **Response payloads** - the full model response before parsing

This is invaluable when debugging issues like "tool schema too complex" errors.

## InstrumentationSettings

Fine-grained control over instrumentation behavior:

```python
from pydantic_ai import Agent, InstrumentationSettings

settings = InstrumentationSettings(
    # Semantic conventions version (1, 2, or 3)
    version=3,  # Use latest spec-compliant naming

    # Event capture mode
    event_mode='attributes',  # 'attributes' (default) | 'logs'

    # Content inclusion
    include_content=True,  # Include prompts/completions (default: True)
    include_binary_content=True,  # Include images/audio (default: True)

    # Custom providers (optional)
    tracer_provider=my_tracer_provider,
    logger_provider=my_logger_provider,
)

agent = Agent('openai:gpt-5', instrument=settings)
```

### Version Differences

| Setting | Version 1-2 | Version 3 |
|---------|-------------|-----------|
| Agent run span | `agent run` | `invoke_agent {name}` |
| Tool call span | `running tool` | `execute_tool {name}` |
| Tool args attr | `tool_arguments` | `gen_ai.tool.call.arguments` |
| Tool result attr | `tool_response` | `gen_ai.tool.call.result` |

### Event Modes

- **`attributes`** (default): Collects events into a JSON array on the request span
- **`logs`**: Emits individual OTel log events as children of the request span (OTel 1.36.0 spec)

## OpenTelemetry Spans

When instrumentation is enabled, the following spans are created:

| Span | Description | Key Attributes |
|------|-------------|----------------|
| **Agent run** | Entire `run()`/`run_sync()` call | `gen_ai.agent.name`, `metadata` |
| **Model request** | Each request to the LLM | `gen_ai.request.model`, `events` |
| **Tool call** | Each tool execution | `gen_ai.tool.name`, `tool_arguments` |

## InstrumentedModel

Wrap any model with instrumentation directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentedModel, InstrumentationSettings

settings = InstrumentationSettings()
model = InstrumentedModel('openai:gpt-5', settings)
agent = Agent(model)
```

## Using OpenTelemetry

Pydantic AI follows the [OpenTelemetry Semantic Conventions for Generative AI systems](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

### With Logfire to Alternative Backend

Use Logfire SDK to send data to any OTel backend:

```python
import os

import logfire

from pydantic_ai import Agent

os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://localhost:4318'
logfire.configure(send_to_logfire=False)  # Disable Logfire backend
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

agent = Agent('openai:gpt-5')
result = agent.run_sync('What is the capital of France?')
```

Example with [otel-tui](https://github.com/ymtdzzz/otel-tui):

```bash
# Run otel-tui
docker run --rm -it -p 4318:4318 --name otel-tui ymtdzzz/otel-tui:latest
```

### Without Logfire (Raw OTel)

Use OpenTelemetry SDK directly:

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider

from pydantic_ai import Agent

exporter = OTLPSpanExporter()
span_processor = BatchSpanProcessor(exporter)
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(span_processor)
set_tracer_provider(tracer_provider)

Agent.instrument_all()
agent = Agent('openai:gpt-5')
result = agent.run_sync('What is the capital of France?')
```

Required packages: `opentelemetry-sdk`, `opentelemetry-exporter-otlp`

### Custom Providers

Set custom tracer and logger providers:

```python
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.trace import TracerProvider

from pydantic_ai import Agent, InstrumentationSettings

settings = InstrumentationSettings(
    tracer_provider=TracerProvider(),
    logger_provider=LoggerProvider(),
)

agent = Agent('openai:gpt-5', instrument=settings)
# or: Agent.instrument_all(settings)
```

## Privacy and Security

### Excluding Prompts and Completions

For privacy/security, exclude sensitive content from telemetry:

```python
from pydantic_ai import Agent, InstrumentationSettings

settings = InstrumentationSettings(include_content=False)
agent = Agent('openai:gpt-5', instrument=settings)
```

This excludes:
- User prompts and model completions
- Tool call arguments and responses
- Any other message content

Structural information (span names, timing, token counts) is preserved.

### Excluding Binary Content

Exclude images, audio, and other binary data:

```python
from pydantic_ai import Agent, InstrumentationSettings

settings = InstrumentationSettings(include_binary_content=False)
agent = Agent('openai:gpt-5', instrument=settings)
```

## Custom Metadata

Attach additional data to agent spans:

```python
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-5',
    instrument=True,
    metadata={'environment': 'production', 'version': '1.0'},
)

# Or at runtime
result = agent.run_sync('prompt', metadata={'user_id': '123'})
```

The computed metadata is recorded on the agent span under the `metadata` attribute.

## Alternative Observability Backends

Because Pydantic AI uses OpenTelemetry, you can send data to any OTel-compatible backend:

- [Langfuse](https://langfuse.com/docs/integrations/pydantic-ai)
- [W&B Weave](https://weave-docs.wandb.ai/guides/integrations/pydantic_ai/)
- [Arize](https://arize.com/docs/ax/observe/tracing-integrations-auto/pydantic-ai)
- [Openlayer](https://www.openlayer.com/docs/integrations/pydantic-ai)
- [OpenLIT](https://docs.openlit.io/latest/integrations/pydantic)
- [LangWatch](https://docs.langwatch.ai/integration/python/integrations/pydantic-ai)
- [Patronus AI](https://docs.patronus.ai/docs/percival/pydantic)
- [Opik](https://www.comet.com/docs/opik/tracing/integrations/pydantic-ai)
- [mlflow](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/pydantic_ai)
- [Agenta](https://docs.agenta.ai/observability/integrations/pydanticai)
- [Confident AI](https://documentation.confident-ai.com/docs/llm-tracing/integrations/pydanticai)
- [Braintrust](https://www.braintrust.dev/docs/integrations/sdk-integrations/pydantic-ai)
- [SigNoz](https://signoz.io/docs/pydantic-ai-observability/)

## Production Recommendations

1. **Always enable instrumentation** - overhead is minimal, debugging value is immense
2. **Use meaningful agent names** - `Agent('openai:gpt-5', name='customer-support')` for easier filtering
3. **Monitor usage** - track `result.usage()` to catch runaway token consumption
4. **Enable HTTP instrumentation** - `logfire.instrument_httpx(capture_all=True)` for full visibility
5. **Consider content exclusion** - use `include_content=False` in production with sensitive data

## Troubleshooting

### No Spans Appearing

1. Verify Logfire is configured: `logfire.configure()`
2. Verify instrumentation is enabled: `logfire.instrument_pydantic_ai()` or `instrument=True`
3. Check `.logfire` directory exists with valid credentials
4. Ensure spans are being exported (check network/firewall)

### Missing HTTP Details

Enable HTTPX instrumentation with full capture:

```python
logfire.instrument_httpx(capture_all=True)
```

### Large Payloads

Binary content (images, audio) can create large spans. Use:

```python
InstrumentationSettings(include_binary_content=False)
```

### Version Compatibility

If your OTel backend requires specific attribute names, adjust the version:

```python
InstrumentationSettings(version=1)  # For older OTel 1.36.0 compatibility
InstrumentationSettings(version=3)  # For latest spec-compliant naming
```

## Integration with Durable Execution

For durable workflows (Temporal, DBOS, Prefect), observability is particularly valuable:

- Trace full lifecycle of workflows spanning hours/days
- Correlate agent runs across workflow restarts
- Monitor workflow health and identify stuck workflows

**Temporal:**
```python
from pydantic_ai.durable_exec.temporal import LogfirePlugin, PydanticAIPlugin

client = await Client.connect(
    'localhost:7233',
    plugins=[PydanticAIPlugin(), LogfirePlugin()],
)
```

**DBOS/Prefect:** Use standard Logfire configuration alongside platform-specific observability.

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `InstrumentationSettings` | `pydantic_ai` | Instrumentation configuration |
| `InstrumentedModel` | `pydantic_ai.models.instrumented` | Model wrapper with instrumentation |
