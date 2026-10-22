---
description: 'Use native Cloudflare Workers AI chat, decision, embedding and image models with shared account configuration.'
---

# Cloudflare Workers AI

Use the `cloudflare:` prefix with the interface for your task:

| Interface | Native Workers AI task | Adapter |
| --- | --- | --- |
| `Agent` | Chat, including supported vision inputs | [`CloudflareChatModel`][pydantic_ai.models.cloudflare.CloudflareChatModel] |
| `Agent` | Clef decisions | [`CloudflareClefModel`][pydantic_ai.models.cloudflare_clef.CloudflareClefModel] |
| `Embedder` | Text embeddings | [`CloudflareEmbeddingModel`][pydantic_ai.embeddings.cloudflare.CloudflareEmbeddingModel] |
| `ImageGenerator` | Image generation and editing | [`CloudflareImageGenerationModel`][pydantic_ai.images.cloudflare.CloudflareImageGenerationModel] |

These models run on Cloudflare. Third-party model routing through AI Gateway, speech/realtime, and
standalone reranking, classification, translation and image-to-text APIs are outside these adapters.

## Install and authenticate

For chat and embeddings, install `pydantic-ai-slim[openai]`. Clef and native image generation need
only `pydantic-ai-slim`; importing their adapters does not import the OpenAI SDK.

Create a [Cloudflare API token with Workers AI access](https://developers.cloudflare.com/workers-ai/get-started/rest-api/)
and set the token and account ID:

```bash
export CLOUDFLARE_API_TOKEN='your-api-token'
export CLOUDFLARE_ACCOUNT_ID='your-account-id'
```

## Chat and vision

```python {test="skip" gateway_toggle="skip"}
from pydantic_ai import Agent

agent = Agent('cloudflare:@cf/meta/llama-4-scout-17b-16e-instruct')
result = agent.run_sync('Explain how a shipping delay affects an order.')
print(result.output)
```

Cloudflare's [Chat Completions endpoint](https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/)
supports most Workers AI text-generation models. Tool calling, vision and structured-output behavior
depend on the selected model. Profiles are conservative for unlisted models; pass an explicit
`profile` when enabling additional capabilities you have verified.

`Agent` selects the decision adapter for `@cf/cloudflare/clef` and `@cf/cloudflare/clef-flash` and the
chat adapter for other model IDs. An ID from an unrelated task does not become a chat model; use the
appropriate interface. For a future decision model, construct `CloudflareClefModel` explicitly.

## Embeddings

```python {test="skip" gateway_toggle="skip"}
from pydantic_ai import Embedder

embedder = Embedder('cloudflare:@cf/baai/bge-small-en-v1.5')
result = embedder.embed_query_sync('How can I track my order?')
print(len(result.embeddings[0]))
```

`Embedder` uses the OpenAI-compatible embeddings endpoint with the same account credentials.
Model-specific dimension limits are enforced by Workers AI.

## Image generation

```python {test="skip" gateway_toggle="skip"}
from pydantic_ai import ImageGenerator

generator = ImageGenerator('cloudflare:@cf/black-forest-labs/flux-2-klein-4b')
result = generator.generate_sync(
    'A watercolor illustration of a delivery bicycle.',
    settings={'dimensions': (1024, 1024)},
)
```

FLUX.2 uses multipart requests, including for text-only generation, and accepts up to four reference
images named by their position. Pass binary images or image URLs through `images`; URLs are downloaded
and sent inline. Cloudflare's reference-image size limits apply; the adapter does not resize inputs.
Other JSON image models, including Stable Diffusion XL, accept text prompts through this adapter.
Provider-side uploaded file IDs are unsupported.

[`CloudflareImageGenerationSettings`][pydantic_ai.images.cloudflare.CloudflareImageGenerationSettings]
exposes native steps, seed, guidance and negative-prompt controls. Their
availability and ranges depend on the selected model. `dimensions` maps to width/height except on
FLUX.1 Schnell, which has no size control and rejects geometry settings before making a request.
`aspect_ratio` maps to integer dimensions preserving the requested ratio with the longest edge at most
1024 pixels; model-specific size limits are enforced by Workers AI. Explicit `dimensions` are forwarded unchanged.

## Model discovery

Any full Workers AI model ID is accepted; the autocomplete list is intentionally representative.
Constructing a model does not make a catalog request. To discover current IDs explicitly:

```python {test="skip" gateway_toggle="skip"}
from pydantic_ai.providers.cloudflare import CloudflareProvider


async def main():
    async with CloudflareProvider() as provider:
        chat_and_decision_ids = await provider.discover()
        embedding_ids = await provider.discover(task='Text Embeddings')
        image_ids = await provider.discover(task='Text-to-Image')
        print(chat_and_decision_ids, embedding_ids, image_ids)
```

Discovery excludes deprecated models and filters by catalog task. Cloudflare also labels Clef as
text generation, so supported Clef IDs are retained and unsupported Clef variants are excluded.
Results are cached on the provider; pass `refresh=True` to fetch the catalog again. Task membership
does not guarantee every optional capability or Chat Completions compatibility.

## Decisions

[`CloudflareClefModel`][pydantic_ai.models.cloudflare_clef.CloudflareClefModel] runs agents on
[Clef](https://developers.cloudflare.com/workers-ai/models/clef/) and
[Clef Flash](https://developers.cloudflare.com/workers-ai/models/clef-flash/). These are
[decision models](decision.md): they answer yes/no, pick-one, and rubric questions with probabilities.
Use an agent's `output_type` to define those questions.

Select either model with the `cloudflare:` prefix:

```python {test="skip" gateway_toggle="skip"}
from pydantic_ai import Agent

agent = Agent(
    'cloudflare:@cf/cloudflare/clef-flash',
    output_type=bool,
    instructions='Does this report describe an urgent production incident?',
)
result = agent.run_sync('All production API requests return 503. Please help urgently.')
print(result.output)
```

The [Decision models guide](decision.md) explains supported output types, tool routing, confidence,
and fallback to a language model.

## Configure the provider

Pass a [`CloudflareProvider`][pydantic_ai.providers.cloudflare.CloudflareProvider] to set the token,
account ID, or HTTP client explicitly:

```python {gateway_toggle="skip"}
from pydantic_ai.models.cloudflare_clef import CloudflareClefModel
from pydantic_ai.providers.cloudflare import CloudflareProvider

model = CloudflareClefModel(
    '@cf/cloudflare/clef',
    provider=CloudflareProvider(api_key='your-api-token', account_id='your-account-id'),
)
```

For connection-based configuration, put the account in `base_url` and pass the token as `api_key`.
An explicit `base_url` takes precedence over `account_id` and its environment variable:

```python {gateway_toggle="skip"}
from pydantic_ai.models.cloudflare_clef import CloudflareClefModel
from pydantic_ai.providers.cloudflare import CloudflareProvider

model = CloudflareClefModel(
    '@cf/cloudflare/clef-flash',
    provider=CloudflareProvider(
        api_key='your-api-token',
        base_url='https://api.cloudflare.com/client/v4/accounts/your-account-id',
    ),
)
```

The provider also accepts a custom `httpx2.AsyncClient` through `http_client`.

## Decision image inputs

Both models accept up to four PNG, JPEG, or WebP images. Pass
[`BinaryContent`][pydantic_ai.messages.BinaryContent] or [`ImageUrl`][pydantic_ai.messages.ImageUrl]
in the prompt; image URLs are downloaded and sent inline. Cloudflare's
[image limits](https://developers.cloudflare.com/workers-ai/models/clef/) apply to the decoded images.

```python {test="skip" gateway_toggle="skip"}
from pathlib import Path
from typing import Literal

from pydantic_ai import Agent, BinaryContent

agent = Agent(
    'cloudflare:@cf/cloudflare/clef-flash',
    output_type=Literal['kiwi', 'apple'],
    instructions='Which fruit is shown?',
)
result = agent.run_sync(
    ['Identify this fruit.', BinaryContent(data=Path('fruit.jpg').read_bytes(), media_type='image/jpeg')]
)
print(result.output)
```

## Decision settings and limits

[`DecisionModelSettings`][pydantic_ai.models.decision.DecisionModelSettings] provides the shared decision
thresholds. `timeout`, `extra_headers`, and `extra_body` are forwarded to Workers AI. Generic sampling
settings, including `temperature`, are ignored because Clef rejects them.

Field aliases and route names are mapped to Cloudflare-compatible question IDs when needed; returned answers
retain their original names.

The [model profile](../api/profiles.md) sets these context windows:

| Model ID | Context window |
| --- | --- |
| `@cf/cloudflare/clef` | 65,536 tokens |
| `@cf/cloudflare/clef-flash` | 24,576 tokens |

Both profiles allow up to 255 options in a pick-one question and 10 levels in a rubric. Workers AI accepts
at most 64 questions and four images per request. Requests over those transport limits raise
[`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] before being sent, so a
[`FallbackModel`](overview.md#fallback-model) can take over.
