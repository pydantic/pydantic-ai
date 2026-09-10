# Moonshot AI

[`MoonshotAIModel`][pydantic_ai.models.moonshotai.MoonshotAIModel] uses the [Kimi API](https://platform.kimi.ai/docs).
Install `pydantic-ai`, or `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

Create an API key in the [Moonshot Console](https://platform.moonshot.ai/console) and set `MOONSHOTAI_API_KEY`:

```bash
export MOONSHOTAI_API_KEY='your-api-key'
```

The `moonshotai:` prefix selects `MoonshotAIModel`:

```python
from pydantic_ai import Agent

agent = Agent('moonshotai:kimi-k3')
...
```

To configure credentials or an HTTP client explicitly, pass a [`MoonshotAIProvider`][pydantic_ai.providers.moonshotai.MoonshotAIProvider]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.moonshotai import MoonshotAIModel
from pydantic_ai.providers.moonshotai import MoonshotAIProvider

model = MoonshotAIModel('kimi-k3', provider=MoonshotAIProvider(api_key='your-api-key'))
agent = Agent(model)
...
```

`MoonshotAIModel` accepts [`OpenAIChatModelSettings`][pydantic_ai.models.openai.OpenAIChatModelSettings].

### Parameter limits {#parameter-limits}

Accepted values depend on the model. On `kimi-k3`, `temperature` is fixed at `1.0`,
`top_p` at `0.95`, and `presence_penalty` and `frequency_penalty` at `0`.
Omit these settings: sending other values returns an API error. See the
[Kimi parameter reference](https://platform.kimi.ai/docs/api/models-overview) for the limits of each model.

## Dynamic tool loading

On `kimi-k3`, [deferred tools](../tools-advanced.md#tool-search) and tools from
[on-demand capabilities](../capabilities/on-demand.md) use Kimi's
[native dynamic tool loading](https://platform.kimi.ai/docs/guide/use-dynamic-tool-loading).
When a tool is revealed by tool search, `load_capability`, or [`ToolReturn.tools`][pydantic_ai.messages.ToolReturn],
its full definition is appended to the conversation. The top-level `tools` list stays unchanged,
preserving the existing prompt prefix.

This is a tool-loading channel, not a native tool-search service: Pydantic AI runs tool search locally.
Other Moonshot models keep the ordinary Chat Completions behavior. Kimi models accessed through
OpenRouter do not use this channel. Explicitly constructing `OpenAIChatModel` with a
`MoonshotAIProvider` also retains the ordinary mapping; use `MoonshotAIModel` for native loading.

To disable the native channel, pass `profile={'tool_addition_mode': None}` to `MoonshotAIModel`.
