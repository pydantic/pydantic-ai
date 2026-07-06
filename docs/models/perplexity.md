# Perplexity

[Perplexity](https://docs.perplexity.ai/) exposes an OpenAI-compatible chat completions endpoint backed by models that ground their answers in live web search.

Pydantic AI talks to Perplexity through [`PerplexityProvider`][pydantic_ai.providers.perplexity.PerplexityProvider] paired with [`PerplexityModel`][pydantic_ai.models.perplexity.PerplexityModel].

## Install

`PerplexityProvider` reuses the OpenAI client, so install Pydantic AI with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

Create an API key from the [Perplexity API key dashboard](https://www.perplexity.ai/account/api/keys), then set the `PERPLEXITY_API_KEY` environment variable. The `PPLX_API_KEY` alias used by Perplexity's own SDKs is also accepted.

You can then refer to a Perplexity model by name:

```python
from pydantic_ai import Agent

agent = Agent('perplexity:sonar-pro')
...
```

Or initialise the provider directly to pass in an explicit API key or a custom HTTP client:

```python
from pydantic_ai import Agent
from pydantic_ai.models.perplexity import PerplexityModel
from pydantic_ai.providers.perplexity import PerplexityProvider

model = PerplexityModel(
    'sonar-pro',
    provider=PerplexityProvider(api_key='your-perplexity-api-key'),
)
agent = Agent(model)
```

Pick a model name from the [Perplexity model directory](https://docs.perplexity.ai/getting-started/models).

## Web Search

Perplexity's chat models perform web search natively on every request. There is no opt-in flag to send and no per-request configuration, so [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool] is not supported.

The sources Perplexity used are returned on each response in [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details]: `citations` (a list of URLs) and `search_results` (title, url, snippet, and dates). Read them from `result.all_messages()[-1].provider_details`.

To control the search itself (domain allow/deny lists, recency windows, custom result counts), call Perplexity's standalone [Search API](https://docs.perplexity.ai/docs/search/quickstart) directly — it lives at `POST https://api.perplexity.ai/search` and returns ranked results without invoking a chat model.

## Agent API

In addition to chat completions, Perplexity offers an [Agent API](https://docs.perplexity.ai/docs/agent/quickstart) — preset-driven research workflows that run multi-step searches before responding. The Agent API is available at `POST https://api.perplexity.ai/v1/responses` (an OpenAI Responses-compatible alias) and is accessed today via the dedicated [`perplexity` Python SDK](https://pypi.org/project/perplexity/) or raw HTTP. Direct support inside Pydantic AI's `OpenAIResponsesModel` is not yet wired up.
