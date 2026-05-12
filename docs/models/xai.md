# xAI

## Install

To use [`XaiModel`][pydantic_ai.models.xai.XaiModel], you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `xai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[xai]"
```

## Configuration

To use xAI models from [xAI](https://x.ai/api) through their API, go to [console.x.ai](https://console.x.ai/team/default/api-keys) to create an API key.

[docs.x.ai](https://docs.x.ai/docs/models) contains a list of available xAI models.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export XAI_API_KEY='your-api-key'
```

You can then use [`XaiModel`][pydantic_ai.models.xai.XaiModel] by name:

```python
from pydantic_ai import Agent

agent = Agent('xai:grok-4-1-fast-non-reasoning')
...
```

Or initialise the model directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel

# Uses XAI_API_KEY environment variable
model = XaiModel('grok-4-1-fast-non-reasoning')
agent = Agent(model)
...
```

You can also customize the [`XaiModel`][pydantic_ai.models.xai.XaiModel] with a custom provider:

```python
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel
from pydantic_ai.providers.xai import XaiProvider

# Custom API key
provider = XaiProvider(api_key='your-api-key')
model = XaiModel('grok-4-1-fast-non-reasoning', provider=provider)
agent = Agent(model)
...
```

Or with a custom `xai_sdk.AsyncClient`:

```python
from xai_sdk import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel
from pydantic_ai.providers.xai import XaiProvider

xai_client = AsyncClient(api_key='your-api-key')
provider = XaiProvider(xai_client=xai_client)
model = XaiModel('grok-4-1-fast-non-reasoning', provider=provider)
agent = Agent(model)
...
```

## X Search

xAI models support searching X (formerly Twitter) for real-time posts and content. The recommended way to enable it is with the [`XSearch`][pydantic_ai.models.xai.XSearch] capability, which configures the underlying `x_search` native tool and can be passed alongside any other capabilities on the agent. See the [xAI X Search documentation](https://docs.x.ai/developers/tools/x-search) for the full list of supported options.

```py {title="xai_x_search.py"}
from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.models.xai import XSearch

agent = Agent(
    'xai:grok-4-1-fast',
    capabilities=[
        XSearch(
            allowed_x_handles=['OpenAI', 'AnthropicAI', 'dasfacc'],
            from_date=datetime(2024, 1, 1),
            to_date=datetime(2024, 12, 31),
            enable_image_understanding=True,
            enable_video_understanding=True,
            include_output=True,
        )
    ],
)

result = agent.run_sync('What have AI companies been posting about?')
print(result.output)
"""
OpenAI announced their latest model updates, while Anthropic shared research on AI safety...
"""
```

_(This example is complete, it can be run "as is")_

The `XSearch` capability accepts:

- **`allowed_x_handles`** / **`excluded_x_handles`**: filter results to (or away from) up to 10 X handles. These are mutually exclusive.
- **`from_date`** / **`to_date`**: restrict results to posts created within the given datetime range (naive datetimes are interpreted as UTC).
- **`enable_image_understanding`** (default: `False`): analyze images attached to posts.
- **`enable_video_understanding`** (default: `False`): analyze video content attached to posts.
- **`include_output`** (default: `False`): include the raw X search results on the [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] available via [`ModelResponse.native_tool_calls`][pydantic_ai.messages.ModelResponse.native_tool_calls]. Without this, the model uses the search results internally but only returns its text summary; enabling it gives programmatic access to the searched posts, sources, and metadata.

As an alternative to the capability, you can pass the lower-level [`XSearchTool`][pydantic_ai.native_tools.XSearchTool] directly via `capabilities=[NativeTool(XSearchTool(...))]` — see the [X Search Tool documentation](../native-tools.md#x-search-tool) — or enable raw output globally via the [`XaiModelSettings.xai_include_x_search_output`][pydantic_ai.models.xai.XaiModelSettings.xai_include_x_search_output] [model setting](../agent.md#model-run-settings).

## Streaming cancellation

!!! warning "Cancellation limitations"
    The `xai-sdk` SDK exposes streaming responses only as an async iterator, with no separate handle for cancelling the underlying gRPC call. Because of a [Python language rule on async generators](https://peps.python.org/pep-0525/), [`cancel()`][pydantic_ai.result.StreamedRunResult.cancel] cannot interrupt an in-flight chunk read while another coroutine is iterating the stream. Pydantic AI marks the response with `state='interrupted'`, but upstream generation may continue until the surrounding `async with agent.run_stream(...)` block exits.

    For reliable cancellation, either pass `debounce_by=None` to [`stream_text()`][pydantic_ai.result.StreamedRunResult.stream_text], [`stream_output()`][pydantic_ai.result.StreamedRunResult.stream_output], or [`stream_responses()`][pydantic_ai.result.StreamedRunResult.stream_responses] and call `cancel()` from the same task that's iterating:

    ```python {title="cancel_xai.py" test="skip"}
    from pydantic_ai import Agent

    agent = Agent('xai:grok-4-1-fast-non-reasoning')


    def should_stop(chunk: str) -> bool:
        return len(chunk) > 100


    async def main():
        async with agent.run_stream('Write a long essay about Python') as result:
            async for chunk in result.stream_text(debounce_by=None):
                if should_stop(chunk):
                    await result.cancel()
                    break
    ```

    Or, if you need to keep debouncing, wrap the stream with [`contextlib.aclosing`](https://docs.python.org/3/library/contextlib.html#contextlib.aclosing) so the iterator is closed before `cancel()` runs:

    ```python {title="cancel_xai_aclosing.py" test="skip"}
    from contextlib import aclosing

    from pydantic_ai import Agent

    agent = Agent('xai:grok-4-1-fast-non-reasoning')


    def should_stop(chunk: str) -> bool:
        return len(chunk) > 100


    async def main():
        async with agent.run_stream('Write a long essay about Python') as result:
            async with aclosing(result.stream_text()) as stream:
                async for chunk in stream:
                    if should_stop(chunk):
                        break
            await result.cancel()
    ```

    Calling `cancel()` from a different task while iteration is in progress is not currently reliable on this provider.
