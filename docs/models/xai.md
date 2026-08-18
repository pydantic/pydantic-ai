# xAI

## Install

To use [`XaiModel`][pydantic_ai.models.xai.XaiModel], you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `xai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[xai]"
```

## Configuration

To use xAI models from [xAI](https://x.ai/api) through their API, go to [console.x.ai](https://console.x.ai/team/default/api-keys) to create an API key.

[docs.x.ai](https://docs.x.ai/developers/models) contains a list of available xAI models.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export XAI_API_KEY='your-api-key'
```

You can then use [`XaiModel`][pydantic_ai.models.xai.XaiModel] by name:

```python
from pydantic_ai import Agent

agent = Agent('xai:grok-4.3')
...
```

Or initialise the model directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel

# Uses XAI_API_KEY environment variable
model = XaiModel('grok-4.3')
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
model = XaiModel('grok-4.3', provider=provider)
agent = Agent(model)
...
```

For gateway, regional, or proxy deployments you can also point the provider at a custom host and set a client-level default timeout, both of which are forwarded to the underlying `xai_sdk.AsyncClient`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel
from pydantic_ai.providers.xai import XaiProvider

provider = XaiProvider(
    api_key='your-api-key',
    api_host='gateway.example.com',
    timeout=30,
)
model = XaiModel('grok-4.3', provider=provider)
agent = Agent(model)
...
```

`api_host` is the hostname of the xAI API server (the SDK connects over gRPC), and `timeout` is the default timeout in seconds applied to every request the client makes. Unlike other providers, the xAI SDK does not support per-request timeouts, so [`ModelSettings.timeout`][pydantic_ai.settings.ModelSettings.timeout] is not supported and has no effect. Both options are omitted when left unset, so the SDK's own defaults apply.

You can also attach gRPC `metadata` to every request the client makes. The canonical use is xAI prompt-cache sticky routing, which pins a conversation to a cache node via an `x-grok-conv-id` so repeated prefixes are served from cache instead of reprocessed:

```python
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel
from pydantic_ai.providers.xai import XaiProvider

provider = XaiProvider(
    api_key='your-api-key',
    metadata=(('x-grok-conv-id', 'my-conversation-id'),),
)
model = XaiModel('grok-4.3', provider=provider)
agent = Agent(model)
...
```

`metadata` is a sequence of `(key, value)` string tuples forwarded verbatim to the underlying `xai_sdk.AsyncClient`, so the xAI SDK's own documentation on [maximizing cache hits](https://docs.x.ai/developers/advanced-api-usage/prompt-caching/maximizing-cache-hits) applies. Because it is client-scoped, it applies to *every* request made through the provider — a provider configured with a fixed `x-grok-conv-id` must not be shared between unrelated conversations, or those conversations will collide on the same cache node. Use a separate provider per conversation when the metadata is conversation-specific. Like `api_host` and `timeout`, it is omitted when left unset and ignored when a custom `xai_sdk.AsyncClient` is passed.

Or with a custom `xai_sdk.AsyncClient`:

```python
from xai_sdk import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel
from pydantic_ai.providers.xai import XaiProvider

xai_client = AsyncClient(api_key='your-api-key')
provider = XaiProvider(xai_client=xai_client)
model = XaiModel('grok-4.3', provider=provider)
agent = Agent(model)
...
```

## X Search

xAI models support searching X (formerly Twitter) for real-time posts and content. The recommended way to enable it is with the [`XSearch`][pydantic_ai.capabilities.XSearch] capability — see the [capability documentation](../capabilities/overview.md#provider-adaptive-tools) for more details, including cross-provider usage. For the full list of supported options, see the [xAI X Search documentation](https://docs.x.ai/developers/tools/x-search).

```py {title="xai_x_search.py"}
from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch

agent = Agent(
    'xai:grok-4.3',
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

- **`allowed_x_handles`** / **`excluded_x_handles`**: filter results to (or away from) up to 20 X handles. These are mutually exclusive.
- **`from_date`** / **`to_date`**: restrict results to posts created within the given datetime range (naive datetimes are interpreted as UTC).
- **`enable_image_understanding`** (default: `False`): analyze images attached to posts.
- **`enable_video_understanding`** (default: `False`): analyze video content attached to posts.
- **`include_output`** (default: `False`): include the raw X search results on the [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] available via [`ModelResponse.native_tool_calls`][pydantic_ai.messages.ModelResponse.native_tool_calls]. Without this, the model uses the search results internally but only returns its text summary; enabling it gives programmatic access to the searched posts, sources, and metadata.

As an alternative to the capability, you can pass the lower-level [`XSearchTool`][pydantic_ai.native_tools.XSearchTool] directly via `capabilities=[NativeTool(XSearchTool(...))]` — see the [X Search Tool documentation](../native-tools.md#x-search-tool) — or enable raw output globally via the [`XaiModelSettings.xai_include_x_search_output`][pydantic_ai.models.xai.XaiModelSettings.xai_include_x_search_output] [model setting](../agent.md#model-run-settings).

## File attachments

When you include a document in a user prompt, xAI automatically makes its `attachment_search` tool available on [supported agentic models](https://docs.x.ai/developers/files). If the model uses the tool, Pydantic AI exposes its lifecycle alongside the model's response. xAI limits each file to [48 MB](https://docs.x.ai/developers/files#limitations). See [document input](../input.md#document-input) for supported input forms.

The [`NativeToolCallPart`][pydantic_ai.messages.NativeToolCallPart] and [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] have a `tool_name` of `'attachment_search'`, and the call's `provider_details['function_name']` holds xAI's own name for the operation it ran (for example `'pdf_browse'`). Set [`XaiModelSettings.xai_include_attachment_search_output`][pydantic_ai.models.xai.XaiModelSettings.xai_include_attachment_search_output] to `True` to ask xAI to include the browsed content on the [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart]. It defaults to `False`, so the option is not sent unless you enable it.

Attachment search applies to files attached directly to a conversation. To search persistent xAI collections instead, use [`FileSearchTool`][pydantic_ai.native_tools.FileSearchTool].

## Reasoning effort

Grok 4.3 supports `reasoning_effort` values of `'none'`, `'low'`, `'medium'`, and `'high'`. You can configure it directly with [`XaiModelSettings.xai_reasoning_effort`][pydantic_ai.models.xai.XaiModelSettings.xai_reasoning_effort], or use the cross-provider [`ModelSettings.thinking`][pydantic_ai.settings.ModelSettings.thinking] setting:

```py {title="xai_reasoning_effort.py"}
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModelSettings

agent = Agent(
    'xai:grok-4.3',
    model_settings=XaiModelSettings(xai_reasoning_effort='medium'),
)
```

Set `xai_reasoning_effort='none'` or `thinking=False` to disable reasoning on Grok 4.3. xAI redirects several retired text model slugs to `grok-4.3`; choose `grok-4.3` and an explicit reasoning effort when you need predictable behavior and cost. See the [xAI May 15 retirement guide](https://docs.x.ai/developers/migration/may-15-retirement) for details.

Grok 4.5 supports `'low'`, `'medium'`, and `'high'` but not `'none'`, so it always reasons: `thinking=False` is silently ignored and `thinking=True` maps to `'medium'`.

## Agentic turns

When a request uses xAI's server-side [native tools](../native-tools.md) (e.g. web search, code execution, X search), xAI runs its own loop — calling those tools and processing their results — before returning a final response. You can cap how many turns that server-side loop may take with [`XaiModelSettings.xai_max_turns`][pydantic_ai.models.xai.XaiModelSettings.xai_max_turns]:

```py {title="xai_max_turns.py"}
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModelSettings

agent = Agent(
    'xai:grok-4.3',
    model_settings=XaiModelSettings(xai_max_turns=5),
)
```

`xai_max_turns` only governs xAI's server-side native-tool loop. It has no effect on ordinary client-side tools or on Pydantic AI's own agent loop — to bound those, use [`UsageLimits`][pydantic_ai.usage.UsageLimits].

Note that when parallel tool calls are enabled, multiple tool calls can occur within a single turn, so `xai_max_turns` does not necessarily equal the total number of tool calls made.

## Multi-agent models

xAI's [multi-agent models](https://docs.x.ai/developers/model-capabilities/text/multi-agent) (e.g. `grok-4.20-multi-agent`) research a question with several agents in parallel before answering. You can choose how many agents they use with [`XaiModelSettings.xai_agent_count`][pydantic_ai.models.xai.XaiModelSettings.xai_agent_count], which accepts `4` or `16`:

```py {title="xai_agent_count.py"}
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModelSettings

agent = Agent(
    'xai:grok-4.20-multi-agent',
    model_settings=XaiModelSettings(xai_agent_count=16),
)
```

More agents means deeper research, at the cost of more tokens and higher latency. Other xAI models ignore the setting.

## Streaming cancellation

!!! note "Transport cancellation"
    [`cancel()`][pydantic_ai.result.StreamedRunResult.cancel] safely interrupts an active local stream pull, including one running in another task. Current `xai-sdk` releases use that cancellation to cancel an active gRPC read, but the SDK exposes no documented per-stream RPC handle. It therefore does not guarantee when remote generation or billing stops after the local iterator closes. See [xai-org/xai-sdk-python#142](https://github.com/xai-org/xai-sdk-python/issues/142).
