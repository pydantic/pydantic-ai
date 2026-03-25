# `pydantic_ai.usage`

::: pydantic_ai.usage

## Pricing

Pydantic AI uses [`genai-prices`](https://github.com/pydantic/genai-prices) to calculate the cost of model requests and embeddings. By default, pricing data comes from the version of `genai-prices` installed in your environment, which may lag behind newly released models or price changes.

To fetch the latest pricing data at runtime, call [`update_prices()`][pydantic_ai.update_prices] once at application startup:

```py {test="skip"}
import pydantic_ai

pydantic_ai.update_prices()

agent = pydantic_ai.Agent('openai:gpt-4o')
```

This starts a background daemon thread that downloads pricing data from GitHub and refreshes it hourly. If the fetch fails (e.g. no network access), pricing silently falls back to the data bundled with the installed `genai-prices` package.

::: pydantic_ai.prices
