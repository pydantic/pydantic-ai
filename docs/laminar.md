# Laminar (OpenTelemetry)

Laminar is an OpenTelemetry (OTel) compatible observability backend. Since Pydantic AI emits OTel spans, you can export them directly to Laminar.

If you want background on Pydantic AI's OTel data and configuration options, see [Using OpenTelemetry](logfire.md#using-opentelemetry).

## Quickstart

Install the OpenTelemetry SDK and OTLP exporter alongside Pydantic AI. Then configure the OTLP/gRPC exporter to send traces to Laminar and enable Pydantic AI instrumentation.

```bash
pip/uv-add "pydantic-ai-slim[openai]" opentelemetry-sdk opentelemetry-exporter-otlp
```

```python {title="laminar_otel.py" test="skip"}
import os

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider

from pydantic_ai import Agent

exporter = OTLPSpanExporter(
    endpoint="https://api.lmnr.ai:8443/v1/traces",
    headers={"authorization": f"Bearer {os.environ['LMNR_PROJECT_API_KEY']}"},
)
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
set_tracer_provider(tracer_provider)

Agent.instrument_all()

agent = Agent("openai:gpt-5.2")
result = agent.run_sync("What is the capital of France?")
print(result.output)
```

Laminar requires the `authorization` header to be lowercase when using the gRPC exporter. For self-hosted endpoints, ports, and other exporter options, refer to the [Laminar OpenTelemetry documentation](https://docs.laminar.sh/tracing/otel).
