---
description: "Use Azure OpenAI and Microsoft Foundry model deployments with Pydantic AI through the Chat Completions or Responses API, or Claude via the Anthropic SDK."
---

# Microsoft Azure / Foundry

Pydantic AI supports Azure OpenAI and other model deployments in Microsoft Foundry (formerly Azure AI Foundry), as well as Claude through the Anthropic SDK.

| API | Model selection |
| --- | --- |
| Chat Completions | `azure:<deployment-name>` |
| Responses | `azure-responses:<deployment-name>` |
| Anthropic Messages | [`AnthropicModel` with a Foundry client](#claude-on-microsoft-foundry) |

Use your Azure deployment name as the model name. The examples below assume a deployment named `gpt-5.2`.

## Install

Install Pydantic AI with the OpenAI SDK used by this integration:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

To use [Microsoft Foundry](https://ai.azure.com/) as your provider, set `AZURE_OPENAI_ENDPOINT` to a URL whose path ends in `/v1` (for example `https://<resource>.openai.azure.com/openai/v1/` or `https://<resource>.services.ai.azure.com/openai/v1/`), set `AZURE_OPENAI_API_KEY`, and use [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('azure:gpt-5.2')
...
```

!!! tip
    For voice agents, use an Azure OpenAI realtime deployment or Azure AI Voice Live with a [realtime session](../realtime/azure.md) instead.

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider

model = OpenAIChatModel(
    'gpt-5.2',
    provider=AzureProvider(
        azure_endpoint='https://your-resource.openai.azure.com/openai/v1/',
        api_key='your-api-key',
    ),
)
agent = Agent(model)
...
```

This targets the [Azure OpenAI v1 API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle), which Microsoft recommends for all new projects. It also pairs naturally with the Responses API — see [Using Azure with the Responses API](#using-azure-with-the-responses-api) below.

[`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] also recognises [Microsoft Foundry serverless model deployments](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/concepts/endpoints) at `https://<model>.<region>.models.ai.azure.com` and connects to them the same way.

## Connecting to an existing `api-version`-based deployment

If your resource still uses the dated `api-version` API, pass `api_version` (or set the `OPENAI_API_VERSION` environment variable) and point `azure_endpoint` at the resource root instead:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider

model = OpenAIChatModel(
    'gpt-5.2',
    provider=AzureProvider(
        azure_endpoint='https://your-resource.openai.azure.com/',
        api_version='2024-12-01-preview',
        api_key='your-api-key',
    ),
)
agent = Agent(model)
...
```

## Using Azure with the Responses API

Microsoft Foundry also supports the OpenAI Responses API through [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel]. This is particularly recommended when working with document inputs ([`DocumentUrl`][pydantic_ai.DocumentUrl] and [`BinaryContent`][pydantic_ai.BinaryContent]), as Azure's Chat Completions API does not support these input types.

Use the `azure-responses:` prefix to select the Responses API by name (the `azure:` prefix uses the Chat Completions API):

```python
from pydantic_ai import Agent

agent = Agent('azure-responses:gpt-5.2')
...
```

!!! note
    Azure's Responses API doesn't yet support every feature of OpenAI's Responses API — for example, native web search is unavailable, and there are limits around image editing and file uploads. See [Microsoft's Responses API docs](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses) for the current list. This applies whether you use the `azure-responses:` shorthand or construct `OpenAIResponsesModel` with `AzureProvider` directly.

Or initialise the model and provider directly, for example to process a document:

??? example "Document processing with Azure using Responses API"
    ```python
    from pydantic_ai import Agent, BinaryContent
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.azure import AzureProvider

    pdf_bytes = b'%PDF-1.4 ...'  # Your PDF content

    model = OpenAIResponsesModel(
        'gpt-5.2',
        provider=AzureProvider(
            azure_endpoint='https://your-resource.openai.azure.com/openai/v1/',
            api_key='your-api-key',
        ),
    )
    agent = Agent(model)
    result = agent.run_sync([
        'Summarize this document',
        BinaryContent(data=pdf_bytes, media_type='application/pdf'),
    ])
    ```

## Prompt caching

On GPT-5.6 and later deployments, Azure's v1 API honors OpenAI's explicit [prompt caching](openai.md#prompt-caching) settings: [`openai_prompt_cache_options`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_prompt_cache_options], [`openai_cache_instructions`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_cache_instructions] and [`CachePoint`][pydantic_ai.messages.CachePoint] markers, with both the Chat Completions and Responses APIs. Cache writes and reads are reported as [`cache_write_tokens`][pydantic_ai.usage.RequestUsage.cache_write_tokens] and [`cache_read_tokens`][pydantic_ai.usage.RequestUsage.cache_read_tokens] in the response's usage, as on OpenAI. See [Monitoring cache efficiency](../capabilities/caching.md#monitoring-cache-efficiency).

The unified [`cache`][pydantic_ai.settings.ModelSettings.cache] setting and the [`Caching`][pydantic_ai.capabilities.Caching] capability don't turn these on with `AzureProvider`, so set them yourself:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.azure import AzureProvider

model = OpenAIResponsesModel(
    'gpt-5.6-sol',
    provider=AzureProvider(
        azure_endpoint='https://your-resource.openai.azure.com/openai/v1/',
        api_key='your-api-key',
    ),
)
settings = OpenAIResponsesModelSettings(
    openai_prompt_cache_options={'mode': 'implicit', 'ttl': '30m'},
    openai_cache_instructions=True,
)
agent = Agent(model, instructions='...', model_settings=settings)
...
```

This has been verified on the v1 API with `gpt-5.6-luna`, `gpt-6-sol`, `gpt-6-luna`, `gpt-6.1-sol` and `gpt-6-astra`. It hasn't been verified with `gpt-6.1-terra`, which wasn't available to test, or on [`api-version`-based deployments](#connecting-to-an-existing-api-version-based-deployment).

## Grok tool schemas

For deployments whose names start with `grok` (case-insensitive), [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] inlines non-recursive tool-schema definitions while retaining OpenAI's schema conversion, including nested `additionalProperties: false` where applicable. This applies with both [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] and [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel].

Native structured output using [`NativeOutput`][pydantic_ai.output.NativeOutput] with nested models still sends `$defs` and `$ref`. Tool-schema inlining does not apply to these output schemas.

!!! warning "Recursive tool schemas"
    Recursive tool argument types cannot be fully inlined, so their schemas retain `$defs` and `$ref`. If Azure Grok rejects a recursive tool schema with a 400 response, use a non-recursive tool argument type.

### Custom profile with `OpenAIProvider`

The generic `openai:` prefix and [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] select OpenAI profiles even when `OPENAI_BASE_URL` or `base_url` points to Azure. To keep that provider, explicitly pass the profile returned by [`AzureProvider.model_profile`][pydantic_ai.providers.azure.AzureProvider]. This reuses the Azure Grok tool-schema conversion and preserves native structured-output schemas.

```python {gateway_toggle="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider

model_name = 'grok-4-1-fast-reasoning'
model = OpenAIResponsesModel(
    model_name,
    provider=OpenAIProvider(
        base_url='https://your-resource.services.ai.azure.com/openai/v1/',
        api_key='your-api-key',
    ),
    profile=AzureProvider.model_profile(model_name),
)
agent = Agent(model)
```

## Claude on Microsoft Foundry

For Claude, install the `anthropic` optional group and pass an `AsyncAnthropicFoundry` client to [`AnthropicProvider`][pydantic_ai.providers.anthropic.AnthropicProvider]. See [Claude on Microsoft Foundry](anthropic.md#microsoft-foundry) for the example and Entra ID authentication guidance.
