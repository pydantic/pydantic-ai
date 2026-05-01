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

xAI models support searching X (formerly Twitter) for real-time posts and content. The recommended way to enable it is with the [`XSearch`][pydantic_ai.models.xai.XSearch] capability, which configures the underlying `x_search` builtin tool and can be passed alongside any other capabilities on the agent. See the [xAI X Search documentation](https://docs.x.ai/developers/tools/x-search) for the full list of supported options.

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
- **`include_output`** (default: `False`): include the raw X search results on the [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart] available via [`ModelResponse.builtin_tool_calls`][pydantic_ai.messages.ModelResponse.builtin_tool_calls]. Without this, the model uses the search results internally but only returns its text summary; enabling it gives programmatic access to the searched posts, sources, and metadata.

As an alternative to the capability, you can pass the lower-level [`XSearchTool`][pydantic_ai.builtin_tools.XSearchTool] directly via `builtin_tools=[...]` — see the [X Search Tool documentation](../builtin-tools.md#x-search-tool) — or enable raw output globally via the [`XaiModelSettings.xai_include_x_search_output`][pydantic_ai.models.xai.XaiModelSettings.xai_include_x_search_output] [model setting](../agent.md#model-run-settings).
