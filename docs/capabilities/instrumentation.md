# Instrumentation

[`Instrumentation`][pydantic_ai.capabilities.Instrumentation] is a [capability](overview.md) that instruments agent runs with OpenTelemetry tracing: it creates spans for the run itself, each model request, and each tool execution, following the [OpenTelemetry Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/). Combined with [Pydantic Logfire](../logfire.md) (or any OTel backend), it gives you full visibility into what your agent is doing:

```python {title="instrumentation_capability.py" test="skip"}
import logfire

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation

logfire.configure()  # (1)!

agent = Agent('openai:gpt-5.2', capabilities=[Instrumentation()])
```

1. Sets the global `TracerProvider` that `Instrumentation` uses by default. Any OpenTelemetry SDK configuration works too.

Pass [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings] via `Instrumentation(settings=...)` to customize providers, content capture, and the conventions version. To instrument every agent in your application instead of attaching the capability per agent, use [`Agent.instrument_all()`][pydantic_ai.agent.Agent.instrument_all].

Other capabilities can attach attributes to the created spans through the OpenTelemetry API (`opentelemetry.trace.get_current_span().set_attribute(...)`).

See [Debugging and Monitoring](../logfire.md) for the full guide: setup, what gets captured, semantic-conventions versions, and excluding sensitive or binary content.
