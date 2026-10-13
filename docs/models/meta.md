---
description: "Use Meta's Muse models with Pydantic AI through MetaProvider and the OpenAI-compatible Responses API, with the meta: prefix."
---

# Meta

Use Meta's Muse models through [`MetaProvider`][pydantic_ai.providers.meta.MetaProvider] and the [Meta Model API](https://dev.meta.ai/docs). The provider prefix is `meta:`, and it uses the OpenAI-compatible Responses API.

## Install

Install Pydantic AI with the OpenAI SDK used by this integration:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

Create an API key on the [Meta developer platform](https://dev.meta.ai).

You can set the `META_API_KEY` environment variable and use [`MetaProvider`][pydantic_ai.providers.meta.MetaProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('meta:muse-spark-1.3')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.meta import MetaProvider

model = OpenAIResponsesModel(
    'muse-spark-1.3',
    provider=MetaProvider(api_key='your-meta-api-key'),
)
agent = Agent(model)
...
```

Contributor mode models (`muse-spark-1.3-contributor`, `muse-spark-1.2-contributor`) are separate model names with their own [pricing](https://dev.meta.ai/docs/pricing-rate-limits).

## Responses and Chat Completions

The `meta:` prefix uses [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel]. Muse labels the progress updates it writes before a tool call as `commentary`, and Pydantic AI sends that label and Muse's encrypted reasoning back on later requests, which keeps long tool-calling runs from ending on a progress update.

Meta also serves the Chat Completions API, which you can use with [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] and the same provider. It doesn't return reasoning (only a `reasoning_tokens` count in usage), so you won't get [thinking](../capabilities/thinking.md) parts.

## Reasoning

Muse models always reason: Meta rejects `reasoning_effort='none'`, so `thinking=False` is ignored. Efforts from `'minimal'` to `'max'` are supported through the unified `thinking` setting or `openai_reasoning_effort`. Meta only streams reasoning summaries when you request them:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

agent = Agent(
    'meta:muse-spark-1.3',
    model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='auto'),
)
...
```

Unlike OpenAI's reasoning models, Muse accepts `temperature` and `top_p` while reasoning, so they're sent as configured.

## Tool choice

Meta only accepts `tool_choice='auto'`. When Pydantic AI would force a tool itself, such as an [output tool](../output.md#tool-output), it falls back to `'auto'`. An explicit forcing `tool_choice` (`'required'` or specific tools) raises a `UserError`, and `tool_choice='none'` is rejected by the API: it's sent as `'none'`, or as an `allowed_tools` restriction to the output tools when the Responses API has some, and the API only takes `'auto'`.

## Web search

Muse supports the [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool], billed per search on top of token usage.
