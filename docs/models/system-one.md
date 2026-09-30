---
description: "Run Pydantic AI agents on any decision model behind the /v1/systemone API, such as Contrastive Language Models (CLM), Laya, and the decision models Ollama runs locally."
---

# System One API

A [decision model](decision.md) answers typed questions about a text, each with a probability or a distribution over the options, rather than writing text: the fast, one-look "System 1" judgement, next to a language model's step-by-step "System 2" reasoning. TypeSafe's Jev answers these questions over a `POST /v1/systemone` API, and other decision models are available over the same API, such as [Contrastive Language Models](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) (CLM) and [Laya](https://huggingface.co/convaiinnovations/laya), and [Ollama](#ollama) runs decision models locally over it.

[`SystemOneModel`][pydantic_ai.models.system_one.SystemOneModel] is the Pydantic AI model class for any decision model behind this API, and a subclass of [`DecisionModel`][pydantic_ai.models.decision.DecisionModel], like [`TypeSafeModel`](typesafe.md). An agent built for one runs on the other by changing the model. All it needs is the API's URL, and its key if it has one.

As with every model in Pydantic AI, the work is split in two:

- The **model**, [`SystemOneModel`][pydantic_ai.models.system_one.SystemOneModel], speaks the API: it turns an agent run into `/v1/systemone` requests and reads the answers.
- The **provider**, [`SystemOneProvider`][pydantic_ai.providers.system_one.SystemOneProvider], says where the API is and how to authenticate. What the model there can be asked goes in its [profile](#limits).

[`TypeSafeModel`](typesafe.md) and its provider split the same way, for Jev through TypeSafe's SDK.

!!! tip "Start with Decision models"
    **[Decision models](decision.md) is where to learn how an agent's output types, tools and message history map
    onto a decision model's questions**, and what the answers mean.

## Install

`SystemOneModel` talks to the API over HTTP and needs nothing beyond `pydantic-ai-slim` itself.

## Quick start with Ollama {#ollama}

[Ollama](https://ollama.com) v0.35.0 and later runs decision models locally over this API, such as [Nimble](https://ollama.com/library/nimble) from Bespoke Labs and [Tev1](https://ollama.com/library/tev1) from Together AI. With Ollama running, pull Nimble:

```bash
ollama pull nimble
```

Point Pydantic AI at Ollama. Local requests need no API key, and the URL has no `/v1` suffix, unlike the one for Ollama's [OpenAI-compatible API](ollama.md):

```bash
export SYSTEM_ONE_BASE_URL='http://localhost:11434'
```

Then have Nimble label a support ticket:

```python
from typing import Literal

from pydantic_ai import Agent

agent = Agent(
    'system-one:nimble',
    output_type=Literal['billing', 'bug', 'account'],
    instructions='Which label fits this support ticket?',
)
result = agent.run_sync('Our checkout has returned 500 errors since 9am.')
print(result.output)
#> bug
```

Nimble picks one label in one request, and reports how sure it is of each in [`provider_details`](decision.md#confidence-and-thresholds).

## Configuration

To use any other server, set the API's URL, with or without a trailing `/v1`, and its key if it has one, as environment variables:

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

## Limits

Each model has limits of its own, such as how many options a pick-one can have or how long a text it reads, documented by whoever publishes it. They belong to the model, not to the client, so they go in the model's [profile](../api/profiles.md) as a [`DecisionModelProfile`][pydantic_ai.profiles.decision.DecisionModelProfile]:

- `decision_max_choice_options`: a pick-one with more options is refused before a request is sent.
- `decision_max_score_levels`: whole numbers with more levels are [asked as a pick-one](decision.md#what-each-field-type-does) instead of a rubric.

Set them with `profile=`. [Ollama](#ollama), for example, takes at most 26 options in a pick-one and 26 levels in a rubric:

```python
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.profiles.decision import DecisionModelProfile
from pydantic_ai.providers.system_one import SystemOneProvider

model = SystemOneModel(
    'nimble',
    provider=SystemOneProvider(base_url='http://localhost:11434'),
    profile=DecisionModelProfile(decision_max_choice_options=26, decision_max_score_levels=26),
)
```

Ollama also takes at most 64 questions and a 64 KiB request, and answers only with models pulled locally; see [Ollama's decision docs](https://docs.ollama.com/capabilities/decision).

A request over a limit the profile does not know about gets an error response from the API, which is raised as a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError], so a [`FallbackModel`](overview.md#fallback-model) can take over. Where a model reads a limited number of tokens, set its [`context_window`][pydantic_ai.profiles.ModelProfile.context_window] the same way, and a processor that [compacts when the context window fills](../message-history.md#compact-when-the-context-window-fills) keeps the history under it.

!!! note "Measure on your own data"
    Each model's confidence is its own, and a threshold tuned on one model does not carry over to another. Measure
    accuracy, the hand-off rate and any threshold on labelled examples of your own before relying on them.
