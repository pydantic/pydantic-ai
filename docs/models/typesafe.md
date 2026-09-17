# TypeSafe (Jev)

[Jev](https://typesafe.ai) is not a language model. You give it a text and typed questions, and it answers each one with a confidence. It does not write text.

`TypeSafeModel` lets an agent whose job is to decide something run on Jev like on any other model. Each field of the `output_type` becomes one question, the prompt is the text, and the answers come back as the output. Change the model name and the same agent runs on a language model, so you can compare the two.

## Install

To use `TypeSafeModel`, install `pydantic-ai-slim` (or `pydantic-ai`) with the `typesafe` optional group:

```bash
pip/uv-add "pydantic-ai-slim[typesafe]"
```

## Configuration

To use Jev through the [TypeSafe](https://typesafe.ai) API, get an API key from your TypeSafe account.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export TYPESAFE_API_KEY='your-api-key'
```

You can then use `TypeSafeModel` by name, with the `output_type` Jev should fill:

```python
from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent


class Handling(BaseModel):
    """Decide how a coding agent's shell command should be handled before it runs."""

    verdict: Literal['run', 'reject', 'ask'] = Field(
        description='How to handle this command.',
        json_schema_extra={
            'typesafe_criteria': {
                'run': 'Reads, builds, tests or edits inside the project. Reversible.',
                'reject': 'Destroys data, rewrites shared history, or sends secrets over the network.',
                'ask': 'Legitimate but consequential enough that a human should confirm.',
            }
        },
    )
    irreversible: bool = Field(description='Would running this destroy data or leak secrets?')


agent = Agent('typesafe:jev-latest', output_type=Handling)
result = agent.run_sync('rm -rf ./build')
print(result.output)
#> verdict='ask' irreversible=False
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel

model = TypeSafeModel('jev-latest')
agent = Agent(model, output_type=bool)
...
```

## What Jev can answer

Every field of the output type is one question, and all of them go out in a single request:

| Field type | Question | Answer |
|---|---|---|
| `bool` | yes or no | `True` when Jev's probability is at least 0.5 |
| `Literal[...]` or `Enum` of strings | pick one | the chosen option |
| `float` with `ge=0` and `le=1` | yes or no | Jev's probability |

The field description is the question text. The output type's docstring and the agent's instructions are passed along as context, so put the framing there and the per-field wording in the descriptions. Give each option of a `Literal` or `Enum` a description in `json_schema_extra={'typesafe_criteria': {...}}`; without them Jev only sees the option names.

Jev's confidence for every field is on the response, so you can act on how sure it was, for example by asking a human below a threshold:

```python
from pydantic_ai import Agent

agent = Agent('typesafe:jev-latest', output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
print(result.response.provider_details)
#> {'confidence': {'response': 0.95}, 'probabilities': {}}
```

## What fails, and how

Jev cannot write text, call tools, look at images, or change an answer. Anything that needs one of those is refused with a `UserError` before a request is sent, so a wrong agent costs nothing:

- Output: text output, `str` in the output types, more than one output type, `NativeOutput`, `PromptedOutput`, an output type with no fields, or a field that is not one of the types above.
- Tools: function tools, toolsets and native tools.
- Prompt: images, audio, video, documents, or no text at all.
- History: tool calls and tool results from another model, native tools included, and deferred tool results. Earlier user prompts are sent as `previous_prompts`; earlier answers, from Jev or another model, are not.
- Retries: an output validator that raises `ModelRetry`, or an answer Pydantic rejects. Jev cannot revise, so the run stops after the first request.

A `FallbackModel` does not skip past a `UserError`, because it means the agent cannot run on Jev at all. It does fall back on `ModelHTTPError` and `ModelAPIError`, which Jev raises like any other model when the API returns an error or cannot be reached. A response the SDK cannot parse is `UnexpectedModelBehavior`, which is not skipped either.

Jev does not stream. `run_stream`, `event_stream_handler` and the AG-UI and Vercel AI adapters all need streaming, so they are not supported, and neither is `count_tokens_before_request`.

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

model = TypeSafeModel('jev-latest', provider=TypeSafeProvider(api_key='your-api-key'))
agent = Agent(model, output_type=bool)
...
```

You can also customize the `TypeSafeProvider` with a custom `http_client`:

```python
from httpx2 import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

custom_http_client = AsyncClient(timeout=30)
model = TypeSafeModel(
    'jev-latest',
    provider=TypeSafeProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model, output_type=bool)
...
```

## SDK retries {#sdk-retries}

The TypeSafe SDK retries connection errors, timeouts and retryable HTTP statuses twice by default, with backoff. To change that, build the client yourself and hand it to the provider:

```python
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

client = AsyncTypeSafeClient(api_key='your-api-key', retry=RetryPolicy(max_retries=0))
model = TypeSafeModel('jev-latest', provider=TypeSafeProvider(typesafe_client=client))
agent = Agent(model, output_type=bool)
...
```

See [Provider SDK retries](../retries.md#provider-sdk-retries) for how this interacts with Pydantic AI's own retries.

## Model settings

Jev has no sampling knobs, so the generic `temperature`, `top_p` and similar settings are ignored. `timeout`, `extra_headers` and `extra_body` are forwarded to the request:

```python
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel

model = TypeSafeModel('jev-latest')
agent = Agent(model, output_type=bool, model_settings={'timeout': 5})
...
```
