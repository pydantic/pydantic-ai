## This check

Instrumentation includes content by default. `InstrumentationSettings` controls content with `include_content`, `include_binary_content` and `include_model_request_parameters`. The same content must stay out of every channel when its control is off.

1. **A channel that ignores the content controls.** Flag a new span attribute, event, log record, exception record or status description that carries message content, tool arguments or results, instructions, templates or files and does not honor `include_content=False`.
2. **Content exported as metadata.** Flag a value exported as a name, identifier or type whose text can come from the model or a client rather than from the application's own definitions. Redact such a value as content.
3. **Configuration in spans.** Flag request settings, headers, bodies, credentials or native-tool configuration recorded in spans, such as `extra_headers`, `extra_body` or authorization values.
4. **Identifiers derived from URLs or configuration.** Flag such an identifier in a span name, attribute, durable activity name or frontend event.
5. **A channel the sweep does not cover.** `tests/test_instrumentation_content_redaction.py` sweeps every way content enters an agent. When the diff adds an instrumented family, channel or integration that the sweep does not exercise, report a `medium` finding at the new channel, even when its redaction is correct today.
