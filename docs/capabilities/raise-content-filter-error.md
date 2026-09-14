# Raise Content Filter Error

[`RaiseContentFilterError`][pydantic_ai.capabilities.RaiseContentFilterError] is a [capability](overview.md) that opts into treating any model response with `finish_reason='content_filter'` as a [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError], even when the provider returns partial text or refusal text:

```python {title="raise_content_filter_error.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import RaiseContentFilterError
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def filtered_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content='I cannot help with that.')],
        finish_reason='content_filter',
        provider_details={'finish_reason': 'content_filter'},
    )


agent = Agent(FunctionModel(filtered_response), capabilities=[RaiseContentFilterError()])

try:
    agent.run_sync('Tell me how to make a weapon.')
except ContentFilterError as exc:
    print(exc.message)
    #> Content filter triggered. Finish reason: 'content_filter'
```

_(This example is complete, it can be run "as is")_

It declares a default `id` of `'raise_content_filter_error'`, so two instances merge instead of raising a duplicate-id error — see [building custom capabilities](custom.md) for the merge rules.

By default, Pydantic AI only raises [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] when a `content_filter` response is *empty*: if the provider returns partial text or refusal text alongside `finish_reason='content_filter'`, that text becomes ordinary agent output and no error is raised (see [finish reason handling](../models/overview.md#finish-reason-example)). This capability extends the check to *every* `content_filter` response, so partial and refusal text raise too. When it raises, the full [`ModelResponse`][pydantic_ai.messages.ModelResponse] is serialized into [`ContentFilterError.body`][pydantic_ai.exceptions.UnexpectedModelBehavior.body] so the partial text remains inspectable.

## Per-provider behavior {#per-provider-behavior}

Whether a refusal arrives as an error or as ordinary output therefore depends on whether the response ends up with any text: the OpenAI adapters drop the parts, while the Anthropic and Google adapters keep whatever text the provider sent, and an empty `content_filter` response raises for every provider. The adapters differ:

| Provider | Refusal / safety wire event | Adapter behavior | Default outcome |
|---|---|---|---|
| OpenAI (Chat Completions, not streaming) | `choice.message.refusal` | Response parts are emptied, `finish_reason='content_filter'` is set, the refusal string is kept in `provider_details['refusal']` | Response is empty, so [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] is raised |
| OpenAI (Chat Completions, streaming) | refusal delta | The refusal itself is withheld (no part is emitted for it), `finish_reason='content_filter'` is set and the refusal string kept in `provider_details['refusal']`; any text emitted before the refusal delta is retained | Raises when no text preceded the refusal; otherwise the retained text is returned as ordinary output |
| OpenAI Responses | refusal output item | Same as non-streaming Chat Completions | [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] is raised |
| Anthropic | `stop_reason='refusal'` | `finish_reason='content_filter'` is set and the stop explanation is kept in `provider_details['refusal']`, but any text block is preserved as a part | Responses with text parts are returned as ordinary output; empty responses raise [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] |
| Google | Safety-family `finishReason` values (`SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT`, `MODEL_ARMOR`) | `finish_reason='content_filter'` is set, and any text parts are preserved | Responses with text parts are returned as ordinary output; empty responses raise [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] |

Adding [`RaiseContentFilterError`][pydantic_ai.capabilities.RaiseContentFilterError] makes all providers behave the same way: every response with `finish_reason='content_filter'` raises, including Anthropic and Google responses that would otherwise flow the refusal text through as output.
