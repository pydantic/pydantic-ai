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

By default, Pydantic AI only raises [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] when a `content_filter` response is *empty*: if the provider returns partial text or refusal text alongside `finish_reason='content_filter'`, that text becomes ordinary agent output and no error is raised (see [finish reason handling](../models/overview.md#finish-reason-example)). This capability extends the check to *every* `content_filter` response, so partial and refusal text raise too. When it raises, the full [`ModelResponse`][pydantic_ai.messages.ModelResponse] is serialized into [`ContentFilterError.body`][pydantic_ai.exceptions.UnexpectedModelBehavior.body] so the partial text remains inspectable.
