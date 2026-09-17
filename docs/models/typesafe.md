# TypeSafe (Jev)

[Jev](https://typesafe.ai) is not a language model. You give it a text and typed questions, and it answers each one with a confidence. It does not write text.

[`TypeSafeModel`][pydantic_ai.models.typesafe.TypeSafeModel] lets an agent whose job is to decide something run on Jev like on any other model. Each field of the `output_type` becomes one question, the prompt is the text, and the answers come back as the output, so a Pydantic model with several fields extracts several values in one request. Change the model name and the same agent runs on a language model, so you can compare the two.

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
from enum import Enum

from pydantic import BaseModel, Field

from pydantic_ai import Agent


class Verdict(str, Enum):
    """How to handle this command."""

    run = 'run'
    """Reads, builds, tests or edits inside the project. Reversible."""
    reject = 'reject'
    """Destroys data, rewrites shared history, or sends secrets over the network."""
    ask = 'ask'
    """Legitimate but consequential enough that a human should confirm."""


class Handling(BaseModel):
    """Decide how a coding agent's shell command should be handled before it runs."""

    verdict: Verdict
    irreversible: bool = Field(description='Would running this destroy data or leak secrets?')


agent = Agent('typesafe:jev-latest', output_type=Handling)
result = agent.run_sync('rm -rf ./build')
print(result.output)
#> verdict=<Verdict.ask: 'ask'> irreversible=False
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

The field description is the question text; an `Enum` field without one uses the enum's class docstring. The output type's docstring and the agent's instructions are context, so put the framing there and the per-field wording in the descriptions. A docstring under an `Enum` member, as in the example above, describes that option. A `Literal` has nowhere to put descriptions, so Jev only sees its option names.

A bare `bool`, `Literal` or `float` as the `output_type` is a single question with no field to describe, so the agent's instructions are the question, as in the example below.

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

## Judging a conversation

The latest user prompt is the text Jev judges. Everything before it in the message history goes along as context: user prompts, answers, tool calls and their results, from whichever model produced them. That makes Jev a cheap judge of another agent's run:

```python
from pydantic_ai import Agent

assistant = Agent('openai:gpt-5.6-sol')
conversation = assistant.run_sync('hello')

judge = Agent('typesafe:jev-latest', output_type=bool, instructions='Was the assistant polite?')
result = judge.run_sync('Judge the conversation above.', message_history=conversation.all_messages())
print(result.output)
#> True
```

## Files

Jev reads text only. To ask about a document, image or video, add the [File Understanding](../capabilities/file-understanding.md) capability with a model that can read it, and Jev gets a description in the file's place:

```python
from enum import Enum

from pydantic_ai import Agent, DocumentUrl
from pydantic_ai.capabilities import FileUnderstanding


class DocumentSubject(str, Enum):
    """What is this document about?"""

    animals = 'animals'
    vehicles = 'vehicles'
    other = 'other'


agent = Agent(
    'typesafe:jev-latest',
    output_type=DocumentSubject,
    capabilities=[FileUnderstanding(fallback_model='openai:gpt-5.6-sol')],
)
result = agent.run_sync([DocumentUrl('https://example.com/field-guide.pdf')])
print(result.output.value)
#> animals
```

## What Jev cannot do

Jev does not write text, call tools, read files or stream. An agent that needs any of those is refused with a [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent:

- The `output_type` must be one structured type made of the field types above: no `str`, no union of output types, no [`NativeOutput`][pydantic_ai.output.NativeOutput] or [`PromptedOutput`][pydantic_ai.output.PromptedOutput].
- No function tools, toolsets or native tools.
- No image, audio, video or document in the prompt or the history, unless [File Understanding](../capabilities/file-understanding.md) has described it first.
- No streaming: `run_stream`, `event_stream_handler` and the AG-UI and Vercel AI adapters do not work with it.

Jev does not revise an answer either. An output validator that raises [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] gets the same answer again, so a validator that keeps rejecting runs the agent out of retries.

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

You can also customize the [`TypeSafeProvider`][pydantic_ai.providers.typesafe.TypeSafeProvider] with a custom `http_client`:

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
