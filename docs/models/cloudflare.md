---
description: "Run Pydantic AI agents on Cloudflare's Clef and Clef Flash decision models, with typed outputs and image evidence."
---

# Cloudflare Workers AI

[`CloudflareClefModel`][pydantic_ai.models.cloudflare.CloudflareClefModel] runs agents on
[Clef](https://developers.cloudflare.com/workers-ai/models/clef/) and
[Clef Flash](https://developers.cloudflare.com/workers-ai/models/clef-flash/). These are
[decision models](decision.md): they answer yes/no, pick-one, and rubric questions with probabilities.
Use an agent's `output_type` to define those questions.

## Install and authenticate

The adapter needs nothing beyond `pydantic-ai-slim`. Create a
[Cloudflare API token with Workers AI access](https://developers.cloudflare.com/workers-ai/get-started/rest-api/)
and set the token and account ID:

```bash
export CLOUDFLARE_API_TOKEN='your-api-token'
export CLOUDFLARE_ACCOUNT_ID='your-account-id'
```

Then select either model with the `cloudflare:` prefix:

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
from pydantic_ai.models.cloudflare import CloudflareClefModel
from pydantic_ai.providers.cloudflare import CloudflareProvider

model = CloudflareClefModel(
    '@cf/cloudflare/clef',
    provider=CloudflareProvider(api_key='your-api-token', account_id='your-account-id'),
)
```

For connection-based configuration, put the account in `base_url` and pass the token as `api_key`.
An explicit `base_url` takes precedence over `account_id` and its environment variable:

```python {gateway_toggle="skip"}
from pydantic_ai.models.cloudflare import CloudflareClefModel
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

## Images

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

## Settings and limits

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
