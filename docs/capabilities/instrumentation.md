---
description: "Trace Pydantic AI agent runs, model requests and tool calls with OpenTelemetry using the Instrumentation capability, following the GenAI semantic conventions."
---

# Instrumentation

[`Instrumentation`][pydantic_ai.capabilities.Instrumentation] is a [capability](overview.md) that instruments agent runs with OpenTelemetry tracing: it creates spans for the run itself, each model request, and each tool execution, following the [OpenTelemetry Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/). Combined with [Pydantic Logfire](../logfire.md) (or any OTel backend), it gives you full visibility into what your agent is doing:

```python {title="instrumentation_capability.py" test="skip"}
import logfire

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation

logfire.configure()  # (1)!

agent = Agent('{{ openai:medium }}', capabilities=[Instrumentation()])
```

1. Sets the global `TracerProvider` that `Instrumentation` uses by default. Any OpenTelemetry SDK configuration works too.

Pass [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings] via `Instrumentation(settings=...)` to customize providers, content capture, and the conventions version. To instrument every agent in your application instead of attaching the capability per agent, use [`Agent.instrument_all()`][pydantic_ai.agent.Agent.instrument_all].

Other capabilities can add attributes to a tool call's span with [`tool_call_span`][pydantic_ai.capabilities.tool_call_span]. There is no equivalent for the model request span yet, so record what a model request hook does on a span of its own with [`open_span`][pydantic_ai.capabilities.open_span]. See [Tracing what your capability does](custom.md#tracing) for the other helpers.

See [Debugging and Monitoring](../logfire.md#using-logfire) for the full guide: setup, what gets captured, semantic-conventions versions, and excluding sensitive or binary content.
