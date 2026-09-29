---
description: "Run Pydantic AI agents on any decision model behind the /v1/systemone API, such as Contrastive Language Models (CLM) and Laya."
---

# System One API

A [decision model](decision.md) answers typed questions about a text, each with a probability or a distribution over the options, rather than writing text: the fast, one-look "System 1" judgement, next to a language model's step-by-step "System 2" reasoning. TypeSafe's Jev answers these questions over a `POST /v1/systemone` API, and other decision models are available over the same API, such as [Contrastive Language Models](https://github.com/Contrastive-LM/CLM) (CLM) and [Laya](https://huggingface.co/convaiinnovations/laya).

[`SystemOneModel`][pydantic_ai.models.system_one.SystemOneModel] is the Pydantic AI model class for any decision model behind this API, and a subclass of [`DecisionModel`][pydantic_ai.models.decision.DecisionModel], like [`TypeSafeModel`](typesafe.md). An agent built for one runs on the other by changing the model. All it needs is the API's URL, and its key if it has one.

!!! tip "Start with Decision models"
    **[Decision models](decision.md) is where to learn how an agent's output types, tools and message history map
    onto a decision model's questions**, and what the answers mean.

## Install

`SystemOneModel` talks to the API over HTTP and needs nothing beyond `pydantic-ai-slim` itself.

## Configuration

Set the API's URL, and its key if it has one, as environment variables:

```bash
export SYSTEM_ONE_BASE_URL='https://decisions.example.com'
export SYSTEM_ONE_API_KEY='your-api-key'
```

Then use `SystemOneModel` by name, as `system-one:` followed by the name the API serves the model under, such as `system-one:clm-latest`, or initialise the model directly with just that name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel

model = SystemOneModel('clm-latest')
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.system_one import SystemOneProvider

model = SystemOneModel(
    'clm-latest',
    provider=SystemOneProvider(base_url='https://decisions.example.com', api_key='your-api-key'),
)
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

You can also customize the [`SystemOneProvider`][pydantic_ai.providers.system_one.SystemOneProvider] with a custom `http_client`:

```python
from httpx2 import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.system_one import SystemOneProvider

custom_http_client = AsyncClient(timeout=30)
model = SystemOneModel(
    'clm-latest',
    provider=SystemOneProvider(base_url='https://decisions.example.com', http_client=custom_http_client),
)
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

## Model settings

`temperature`, `timeout`, `extra_headers` and `extra_body` are forwarded to the request, and the other generic settings, such as `top_p`, are ignored. Whether `temperature` has an effect depends on the model: where it does, it moves every probability a [threshold](decision.md#confidence-and-thresholds) reads. [`SystemOneModelSettings`][pydantic_ai.models.system_one.SystemOneModelSettings] adds the two thresholds every decision model has, `decision_boolean_threshold` and `decision_route_threshold`.

```python
from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel

model = SystemOneModel('clm-latest')
agent = Agent(
    model,
    output_type=bool,
    instructions='Is this request harmful?',
    model_settings={'temperature': 0.5, 'timeout': 5},
)
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

Each model has limits of its own, such as how many options a pick-one can have or how long a text it reads, documented by whoever publishes it. `SystemOneModel` leaves them to the API: a request over them gets an error response, which is raised as a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError], so a [`FallbackModel`](overview.md#fallback-model) can take over. Where a model reads a limited number of tokens, set its [`context_window`][pydantic_ai.profiles.ModelProfile.context_window] with `profile={'context_window': ...}`, and a processor that [compacts when the context window fills](../message-history.md#compact-when-the-context-window-fills) keeps the history under it.

!!! note "Measure on your own data"
    Each model's confidence is its own, and a threshold tuned on one model does not carry over to another. Measure
    accuracy, the hand-off rate and any threshold on labelled examples of your own before relying on them.
