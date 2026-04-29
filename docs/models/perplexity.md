# Perplexity

[Perplexity](https://docs.perplexity.ai/) exposes an OpenAI-compatible chat completions endpoint backed by models that ground their answers in live web search.

Pydantic AI talks to Perplexity through [`PerplexityProvider`][pydantic_ai.providers.perplexity.PerplexityProvider] paired with [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel].

## Install

`PerplexityProvider` reuses the OpenAI client, so install Pydantic AI with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

Create an API key from the [Perplexity API key dashboard](https://www.perplexity.ai/account/api/keys), then set the `PERPLEXITY_API_KEY` environment variable. The `PPLX_API_KEY` alias used by Perplexity's own SDKs is also accepted.

You can then refer to a Perplexity model by name:

```python {test="skip"}
from pydantic_ai import Agent

agent = Agent('perplexity:sonar-pro')
result = agent.run_sync('What was announced at the latest Pydantic AI release?')
print(result.output)
```

Or initialise the provider directly to pass in an explicit API key or a custom HTTP client:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.perplexity import PerplexityProvider

model = OpenAIChatModel(
    'sonar-pro',
    provider=PerplexityProvider(api_key='your-perplexity-api-key'),
)
agent = Agent(model)
```

Pick a model name from the [Perplexity model directory](https://docs.perplexity.ai/getting-started/models).

## Web Search

Perplexity's chat models perform web search natively on every request — there's no opt-in flag to send. Pydantic AI exposes this through the standard [`WebSearchTool`][pydantic_ai.builtin_tools.WebSearchTool] so the same agent code works whether the underlying provider is Perplexity, OpenAI, Anthropic, or any other supported web-search provider:

```python {test="skip"}
from pydantic_ai import Agent, WebSearchTool

agent = Agent('perplexity:sonar-pro', builtin_tools=[WebSearchTool()])
result = agent.run_sync("Summarise this week's biggest AI announcements.")
print(result.output)
```

For richer search behaviour (domain allow/deny lists, recency windows, custom result counts) outside an LLM loop, call Perplexity's standalone [Search API](https://docs.perplexity.ai/docs/search/quickstart) directly — it lives at `POST https://api.perplexity.ai/search` and returns ranked results without invoking a chat model.

## Agent API

In addition to chat completions, Perplexity offers an [Agent API](https://docs.perplexity.ai/docs/agent/quickstart) — preset-driven research workflows that run multi-step searches before responding. The Agent API is available at `POST https://api.perplexity.ai/v1/responses` (an OpenAI Responses-compatible alias) and is accessed today via the dedicated [`perplexity` Python SDK](https://pypi.org/project/perplexity/) or raw HTTP. Direct support inside Pydantic AI's `OpenAIResponsesModel` is not yet wired up; track [the Perplexity provider issue](https://github.com/pydantic/pydantic-ai/issues) for updates.
