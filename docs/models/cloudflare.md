---
description: "Use Cloudflare Clef decisions and Clairic extraction, computation, estimates and forecasts in Pydantic AI."
---

# Cloudflare Workers AI

[`CloudflareClefModel`][pydantic_ai.models.cloudflare.CloudflareClefModel] runs agents on
[Clef](https://developers.cloudflare.com/workers-ai/models/clef/) and
[Clef Flash](https://developers.cloudflare.com/workers-ai/models/clef-flash/). These are
[decision models](decision.md): they answer yes/no, pick-one, and rubric questions with probabilities.
Use an agent's `output_type` to define those questions.

For typed values and distributions, use [Clairic](#clairic-extraction-and-computation).

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

## Clairic extraction, computation, estimates and forecasts {#clairic-extraction-and-computation}

[`CloudflareClairicModel`][pydantic_ai.models.cloudflare_clairic.CloudflareClairicModel] runs
[Clairic](https://developers.cloudflare.com/workers-ai/models/clairic/), Cloudflare's preview model
for typed extraction, computation, estimation and forecasting. It uses the same authentication and
[`CloudflareProvider`][pydantic_ai.providers.cloudflare.CloudflareProvider] configuration as Clef.

```python {test="skip" gateway_toggle="skip"}
from datetime import date
from typing import Annotated

from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.cloudflare_clairic import ClairicField


class Invoice(BaseModel):
    """Values read or computed from an invoice."""

    invoice_number: str | None
    total: Annotated[float, ClairicField(unit='USD', decimals=2)] = Field(
        description='Amount due including shipping'
    )
    due_date: date | None = Field(description='Issue date plus the net payment days')
    purchase_order: str | None


agent = Agent('cloudflare:@cf/cloudflare/clairic', output_type=Invoice)
result = agent.run_sync(
    'Invoice INV-2291, issued 2026-09-02, net 30. Four pallets at $250 each; shipping $80.'
)
print(result.output)
print(result.response.provider_details)
```

Clairic defaults to native structured output. You can also explicitly use
[`NativeOutput`](../output.md#native-output). Supply one structured output type; function tools,
provider-native tools, tool output, prompted output, free-text generation and streaming are unsupported.
Use field names and `Field(description=...)` to describe each task.

| Output field | Native Clairic field | Notes |
| --- | --- | --- |
| `str` | `text` | Extracted text, with source spans when available. |
| `int` | `number`, `integer=true` | Extract or compute an integer. |
| `float` | `number` | Extract or compute a number. |
| `date`, `time`, `datetime` | `date`, `time`, `datetime` | Native ISO values validated by Pydantic. No timezone is added to naive timestamps. |
| `timedelta` | `duration`, `unit=seconds` | Native seconds become a Python duration. |
| `Annotated[float, ClairicField(kind='duration', unit='months')]` | `duration` | Numeric duration retains its requested unit, including calendar months and years. |
| `list[str]`, `list[float]`, `list[date]` | `list` with `items=text`, `number`, `date` | Lists normally return `[]` when no values match. |
| `ClairicNumberEstimate` | `estimate` | Numeric point prediction and full distribution. |
| `ClairicDateEstimate` | `estimate` | Date point prediction and full distribution. |
| `ClairicForecast` | `forecast` | Numeric forecast arrays and per-step distributions. |
| Nullable supported fields | Same native field | Native `null` remains `None`; guesses never replace missing values. |

Nested objects other than the distribution types, booleans, enums, non-null unions, arbitrary JSON Schema
constraints, nullable list items and integer lists are unsupported. Clairic's `integer` flag applies to
scalar numbers and durations; `list[int]` is rejected because the provider does not guarantee integer
list values. Unsupported field schemas raise [`UserError`][pydantic_ai.exceptions.UserError] before HTTP.
A required scalar receiving `null` enters normal output-validation retries; use nullable fields for
information that may be absent.

### Field controls

[`ClairicField`][pydantic_ai.models.cloudflare_clairic.ClairicField] is a frozen marker for `Annotated`
fields. Python result types infer native `type` and list `items`; the marker supplies native controls.
Controls incompatible with the result type fail before HTTP. The marker leaves ordinary Python values
as ordinary values; use the distribution result types for estimates and forecasts.

| Control | Applies to | Meaning |
| --- | --- | --- |
| `kind='duration'` | Numeric durations | Distinguish a numeric duration from a number; other native kinds follow the Python type. |
| `unit` | Numbers, durations, number lists, numeric estimates, forecasts | Unit label. Duration units: seconds, minutes, hours, days, weeks, months, years. |
| `integer` | Scalar numbers, numeric durations | Integer precision; inferred for `int`. |
| `decimals` | Numbers, durations, number lists | Zero to six decimal places. Integer precision permits zero decimal places. |
| `min`, `max` | Estimates, forecasts | Required bounds, with `min < max`. Date estimates take Python `date` bounds. |
| `step` | Estimates, forecasts | Positive grid spacing; dates also accept `'month'` or `'year'`. Numeric date spacing is in days. |
| `scale` | Estimates, forecasts | `'linear'` or `'log'`; log grids require positive numeric bounds. |
| `null_threshold` | Scalar extraction and computation | Probability threshold above which `value` becomes null; greater than zero and at most one. Native default: 0.5. |
| `horizon` | Forecasts | Required number of steps, from one to 64. |
| `interval` | Forecasts | Required: `'step'`, `'minute'`, `'hour'`, `'day'`, `'week'`, `'month'`, `'quarter'`, `'year'`. |

For `timedelta`, the adapter requests seconds. To retain another unit, use a numeric duration.
It does not convert calendar months or years into a fixed number of seconds.

### Estimates and forecasts

Use [`ClairicNumberEstimate`][pydantic_ai.models.cloudflare_clairic.ClairicNumberEstimate],
[`ClairicDateEstimate`][pydantic_ai.models.cloudflare_clairic.ClairicDateEstimate] or
[`ClairicForecast`][pydantic_ai.models.cloudflare_clairic.ClairicForecast] directly as output fields:

```python {test="skip" gateway_toggle="skip"}
from datetime import date, timedelta
from typing import Annotated

from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.cloudflare_clairic import (
    ClairicDateEstimate,
    ClairicField,
    ClairicForecast,
    ClairicNumberEstimate,
    CloudflareClairicModelSettings,
)


class Predictions(BaseModel):
    """Payment estimates and a weekly collections forecast."""

    payment_days: Annotated[
        ClairicNumberEstimate, ClairicField(min=0, max=120, step=1)
    ] = Field(description='Days after issue until payment')
    payment_date: Annotated[
        ClairicDateEstimate,
        ClairicField(min=date(2026, 10, 1), max=date(2027, 1, 31), step=1),
    ] = Field(description='Date on which the invoice will be paid')
    collections: Annotated[
        ClairicForecast,
        ClairicField(min=0, max=1000, step=10, horizon=3, interval='week', unit='USD'),
    ] = Field(description='Collections for each of the next three weeks')
    service_duration: timedelta


settings = CloudflareClairicModelSettings(
    cloudflare_clairic_reference_date=date(2026, 10, 19),
    cloudflare_clairic_include_quantiles=True,
    cloudflare_clairic_return_state_text=True,
)
agent = Agent('cloudflare:@cf/cloudflare/clairic', output_type=Predictions, model_settings=settings)
result = agent.run_sync(
    'Invoice issued 2026-10-01, net 30. Past payment times: 24, 28, 31, 29 and 27 days. '
    'Weekly collections: 100, 120 and 150 USD. Service duration: 90 minutes.'
)
print(result.output.payment_date.value)
print(result.output.payment_date.median)
print(result.output.collections.steps[0].p10)
```

Estimates preserve `value`, `mean`, `median`, `p10`, `p90`, optional `quantiles`, `p_below` and `p_above`.
Date estimates parse dates throughout and may omit tail probabilities. Forecasts preserve independent
`value`, `median`, `p10` and `p90` arrays plus `steps`, each a numeric estimate. Native forecast steps do
not need a `type` discriminator. Point values can differ from medians; the adapter never substitutes
one for the other. Requesting quantiles adds 99 percentiles to each estimate, including forecast steps.
Extra native distribution fields are retained. Raw answers remain available in `provider_details`.

### Uncertainty and source spans

`result.response.provider_details` retains native `answers`, including `value`, `guess`, `p_null`,
`confidence`, `confidences`, `span`, `spans` and complete distributions when returned. It retains native
`usage`, `truncated` and additional provider response fields. `RequestUsage` includes input/output tokens
and additional integer usage counters. Confidence and null probability describe different uncertainties;
a confident guess can accompany a null value. Typed values and uncertainty scores do not guarantee
correct answers, including for text lists. See Cloudflare's
[interpretation guidance](https://developers.cloudflare.com/workers-ai/concepts/extraction-models/).

`field_names` maps native answer IDs to output field aliases. Valid aliases are retained; if an alias
is invalid, all fields receive safe IDs to avoid collisions. Original field names and descriptions
remain in each field's instructions.

Source offsets index `state_text`. The adapter requests that text by default and preserves the exact
submitted string as `request_state`; if Cloudflare omits `state_text`, submitted text is used. For one
user text, `request_state` is the original text. Multipart conversation history is serialized as a JSON
array of role/content entries, including system prompts, assistant text and validation retries. Inspect
`state_text` before applying offsets to documents. Truncation sets the response finish reason to `length`.

### Images and request settings

Clairic accepts up to four inline PNG, JPEG or WebP images through `BinaryContent` or `ImageUrl`.
Image URLs are downloaded before inference. Image-only prompts use empty state; mixed text/history
includes numbered image markers. Cloudflare's
[image size limits](https://developers.cloudflare.com/workers-ai/models/clairic/) apply.

[`CloudflareClairicModelSettings`][pydantic_ai.models.cloudflare_clairic.CloudflareClairicModelSettings]
exposes `cloudflare_clairic_reference_date` (Python date or ISO string),
`cloudflare_clairic_include_quantiles` (default false) and `cloudflare_clairic_return_state_text`
(default true). Disabling returned text retains `request_state` as the span-reference fallback.

Clairic forwards standard `timeout`, `extra_headers` and `extra_body`; generic tuning settings are ignored.
To preserve output validation and span correspondence, `extra_body` cannot replace `model`, `state`,
`fields`, `images` or `options`. Other body overrides apply after typed settings, including `reference_date`.
The model's context window is 65,536 tokens.
