---
title: Batch Processing
description: "Send agent runs and model requests through provider batch APIs, at about half the price, with Pydantic AI's BatchModelRequests capability and model_request_batch."
---

# Batch Processing

Anthropic, OpenAI and Google offer batch APIs: you submit many requests as one job, and the provider works through them when it has capacity, at about half the standard price. You get results later rather than straight away: Anthropic and OpenAI give a batch a 24-hour window, and Google targets 24 hours and expires a job after 48. Many batches finish much sooner, but none of these windows is a guarantee, and a request the provider didn't get to in time is [submitted again](#failed-requests).

That trade-off suits work that isn't waiting on a user, such as running [evals](evals.md), classifying or extracting data from many records, and generating data offline. It doesn't suit interactive use, or agents that take many turns: every model request of a run, including each one that follows a tool call, is a separate batch round trip.

Pydantic AI supports batch APIs at three levels:

- The [`BatchModelRequests`](#agents) [capability](capabilities/overview.md) sends an agent's model requests through the batch API, without changing how you run the agent. With [`PendingModelResponse`](#ending-a-run-while-it-waits), a run can end once its request is submitted, and be resumed later to collect the result.
- [`model_request_batch`](#direct-batch-requests) sends a list of model requests as a batch and waits for the responses.
- The [`batch_*` methods on `Model`](#submit-now-collect-later) are the low-level primitives the other two are built on: use them to manage batch jobs yourself, without agents.

## Supported providers and APIs {#supported-models}

| Model | Notes |
|---|---|
| [`AnthropicModel`](models/anthropic.md#batch-api) | The Claude API's Message Batches. Not available on Bedrock, Vertex AI or Microsoft Foundry. |
| [`OpenAIResponsesModel`](models/openai.md#batch-api) | OpenAI's Batch API. Not available with `OpenAIChatModel` or Azure OpenAI. On by default with `OpenAIProvider`, including with a custom `base_url`; providers with their own class can [opt in](models/openai.md#batch-api). |
| [`GoogleModel`](models/google.md#batch-api) | The Gemini API's Batch Mode. Not available on Vertex AI. |
| [`TestModel`][pydantic_ai.models.test.TestModel] and [`FunctionModel`][pydantic_ai.models.function.FunctionModel] | Run a batch's requests immediately and in memory, for [testing](testing.md#testing-batch-processing). |

Batching isn't available through [Pydantic AI Gateway](gateway.md) yet: use the provider directly, with its own API key.

These are the APIs Pydantic AI can batch through. Which models a provider serves in batches is up to the provider: see its batch API documentation, linked from each provider's page, which also lists its limits on batch size.

Whether a model has a batch API is a [profile](models/overview.md#models-and-providers) flag: [`anthropic_supports_message_batches`][pydantic_ai.profiles.anthropic.AnthropicModelProfile.anthropic_supports_message_batches], [`openai_supports_batch_api`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_supports_batch_api] or [`google_supports_batch_api`][pydantic_ai.profiles.google.GoogleModelProfile.google_supports_batch_api]. Pass `profile=` to the model to set it where the default is wrong for your endpoint. Batching a model without a batch API raises a [`UserError`][pydantic_ai.exceptions.UserError].

### Cost

Responses from a batch carry `provider_details['processing'] == 'batch'`, so [`ModelResponse.cost()`][pydantic_ai.messages.ModelResponse.cost], `usage.cost` and [usage limits](agent.md#usage-limits) use batch prices from [`genai-prices`](https://github.com/pydantic/genai-prices).

## Batching agent runs {#agents}

Add a [`BatchModelRequests`][pydantic_ai.capabilities.BatchModelRequests] capability to an agent and run it as usual. Runs started together, here with `asyncio.gather`, have their requests collected into one batch:

```python {title="batch_agent.py" gateway_toggle="skip"}
import asyncio
from typing import Literal

from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.capabilities import BatchModelRequests


class Triage(BaseModel):
    category: Literal['billing', 'bug', 'account', 'other']
    urgent: bool


agent = Agent(
    '{{ anthropic:medium }}',
    output_type=Triage,
    instructions='Triage this support ticket.',
    capabilities=[BatchModelRequests()],
)

tickets = [
    'I was charged twice for my subscription this month.',
    'The CSV export stops after 1,000 rows.',
    'How do I change the email address on my account?',
]


async def main():
    results = await asyncio.gather(*(agent.run(ticket) for ticket in tickets), return_exceptions=True)  # (1)!
    for ticket, result in zip(tickets, results):
        if isinstance(result, BaseException):
            print(f'{ticket!r} failed: {result!r}')
        else:
            print(result.output)
            #> category='billing' urgent=True
            #> category='bug' urgent=False
            #> category='account' urgent=False
```

1. A request that fails only fails its own run, not the others in its batch. With `return_exceptions=True`, you get each run's result or error.

### What happens while a request waits

Each model request is queued until its batch holds `max_requests` requests or its oldest request has waited `max_wait`, and then submitted. The run waits while the batch is processed, checking on it at growing intervals, and continues when the result arrives, as if the request had been answered live. That can take minutes or hours, and so can the `await` above: to end a run once its request is submitted and collect the result later, use [`PendingModelResponse`](#ending-a-run-while-it-waits).

Tools, structured output, output validation and retries work as usual, and each model request in a run goes into whichever batch is being collected when it's made. A run that needs several model requests, for example to call tools or to retry after an output validator raised [`ModelRetry`][pydantic_ai.exceptions.ModelRetry], takes several batch round trips, each of which can take as long as the provider's batch window. [Usage limits](agent.md#usage-limits) apply to each run as usual.

Only requests made at about the same time share a batch: runs awaited one after another each wait `max_wait` alone and get a batch each.

### Settings

| Setting | Default | What it does |
|---|---|---|
| `max_requests` | 1,000 | The most requests in one batch. The provider's own limit caps it. |
| `max_wait` | 30 seconds | How long the oldest queued request waits for others before its batch is submitted. Later requests never extend it. `None` submits a batch only when it's full or when you call [`flush()`][pydantic_ai.capabilities.BatchModelRequests.flush]. |
| `max_resubmissions` | 2 | How many new batches a request that [failed on the provider's side](#failed-requests) is submitted in before its error is raised. |
| `poll_interval` | 30 seconds | How long a request waits before first checking its batch. The interval grows by half after every check. |
| `max_poll_interval` | 10 minutes | The longest wait between checks. |

A batch is also submitted early when the next request would take it past the provider's size limit. Raise `max_requests` when you make many more requests at a time, to submit fewer, larger batches: providers limit how many batches can be in progress at once.

A run that takes several batch round trips waits up to `max_wait`, plus the time until the next check, for each, on top of the provider's own time. Lower `max_poll_interval` when that adds up, at the cost of more status requests.

To try another model once a request's error reaches the run, add [`ModelFallback`](capabilities/model-fallback.md): the request then goes into a batch on the next model in its chain. When a run resumed from history finds that the batch it was waiting on failed, it also moves on to the next model, rather than starting over with a new batch on the first.

### Sharing batches between agents

Each `BatchModelRequests` instance collects its own batches. Add the same instance to several agents, or pass it to runs with `capabilities=`, and their requests share batches:

```python {title="batch_shared.py" gateway_toggle="skip"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.capabilities import BatchModelRequests

batching = BatchModelRequests()
replier = Agent('{{ anthropic:medium }}', instructions='Draft a reply to this support ticket.', capabilities=[batching])
summarizer = Agent('{{ anthropic:medium }}', instructions='Summarize this customer feedback in a few words.')


async def main():
    reply, summary = await asyncio.gather(
        replier.run('The app crashes whenever I upload a photo.'),
        summarizer.run(
            'I love the new dashboard, but it takes ages to load on my laptop.',
            capabilities=[batching],  # (1)!
        ),
    )
    print(reply.output)
    #> Sorry about that! We're looking into photo uploads and will update you soon.
    print(summary.output)
    #> Likes the new dashboard, but it loads slowly.
```

1. An agent without the capability can opt individual runs into batching by passing it to `run()`.

### Agent specs {#agent-specs}

`BatchModelRequests` can be declared in an [agent spec](agent-spec.md), with its settings as keyword arguments. Give each duration as a number of seconds or as an ISO 8601 duration, like `PT5M`:

```yaml {title="batch_agent.yaml" test="skip"}
model: {{ anthropic:medium }}
instructions: Triage this support ticket.
capabilities:
  - BatchModelRequests:
      max_requests: 5000
      max_wait: PT5M
```

Each agent built from a spec gets a `BatchModelRequests` instance of its own: its runs share batches with each other, but not with the runs of other agents, even ones built from the same spec. To share batches between agents built from specs, leave the capability out of the specs and pass one instance to their runs with `capabilities=`, as above.

## Submitting now and collecting later {#ending-a-run-while-it-waits}

A batch can take hours, which is longer than many processes should wait. Add [`PendingModelResponse`][pydantic_ai.output.PendingModelResponse] to the agent's `output_type`, and a run ends as soon as its request has been submitted, with a `PendingModelResponse` as its output. The run's message history ends in the pending response, which records the batch, so you can store the history and run the agent again with it later, in this process or another. That run checks on the batch once: if the result has arrived, the run continues as usual, and otherwise it ends with a `PendingModelResponse` again.

```python {title="batch_pending.py" gateway_toggle="skip"}
import asyncio
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel

from pydantic_ai import Agent, ModelMessagesTypeAdapter, PendingModelResponse
from pydantic_ai.capabilities import BatchModelRequests


class Triage(BaseModel):
    category: Literal['billing', 'bug', 'account', 'other']
    urgent: bool


agent = Agent(
    '{{ anthropic:medium }}',
    output_type=[Triage, PendingModelResponse],
    instructions='Triage this support ticket.',
    capabilities=[BatchModelRequests(max_wait=timedelta(seconds=5))],  # (1)!
)


async def submit(tickets: list[str]) -> list[bytes]:
    results = await asyncio.gather(*(agent.run(ticket) for ticket in tickets))  # (2)!
    return [result.all_messages_json() for result in results]  # (3)!


async def collect(stored_messages: bytes) -> Triage | bytes:
    result = await agent.run(message_history=ModelMessagesTypeAdapter.validate_json(stored_messages))
    if isinstance(result.output, PendingModelResponse):
        return result.all_messages_json()  # (4)!
    return result.output


async def main():
    stored = await submit(
        [
            'I was charged twice for my subscription this month.',
            'The CSV export stops after 1,000 rows.',
        ]
    )
    # Later, for example from a scheduled job:
    for messages in stored:
        print(await collect(messages))
        #> category='billing' urgent=True
        #> category='bug' urgent=False
```

1. Submitting a request waits up to `max_wait` for more requests to join its batch. A script that starts its runs together and exits can lower it.
2. Start the runs together, so their requests share a batch: runs awaited one after another each wait `max_wait` alone and get a batch each.
3. Store these, for example in a database, and load them wherever you collect the results.
4. The batch is still being processed: store the new history in place of the old one, and check again later.

Collect with `BatchModelRequests` too: resuming the history without it raises a [`UserError`][pydantic_ai.exceptions.UserError], rather than sending the waiting request again as a new request at the full price. If the agent's model was chosen dynamically, for example by [`SelectModel`](capabilities/select-model.md), resuming also raises a `UserError` unless you pass the model that submitted the request to the run as `model=`, so store the model's name with the history.

A run that needs several model requests, for example to call tools, ends with a `PendingModelResponse` for each of them. Each run in a submit/collect cycle counts the request in its own [`usage.requests`][pydantic_ai.usage.RunUsage.requests], so adding up `usage.requests` across the runs counts a request once per run, while its tokens and cost are only counted by the run that received the result. This works for any request the provider keeps working on after it's been sent, including [OpenAI background mode](models/openai.md#background-mode).

The process that collects a batch doesn't need the API key that submitted it: the provider decides who may check on a batch, so a rotated key of the same project or workspace works. Checks of one batch within `poll_interval` of each other share one status request, and its results are downloaded once while they're cached, so collecting many stored histories costs about one check per batch rather than one per run. The cache keeps the results of the last 32 batches that ended, up to 256 MB of them, so collect the histories of one batch together rather than interleaved with those of many others.

A batch left for a later run like this keeps running whether or not anything collects it, so collect results well within the provider's [retention window](#retention).

## Durable execution

Without `PendingModelResponse` in the output type, a run waits for its batch, and you only get its message history once it ends. If your process may stop while a batch is processed, end runs with a `PendingModelResponse`, or use [durable execution](durable_execution/overview.md).

With Temporal, DBOS or Prefect, add `BatchModelRequests` to the agent alongside its durability capability. Submitting a request and every check on its batch run as separate durable steps, so a run that's interrupted while its batch is processed, for example by a deployment, continues from the same batch when it's recovered. Under Temporal and DBOS, the waits between checks are durable too. Requests are collected into batches per worker process: the requests of concurrent workflow runs on one worker share batches. Temporal and DBOS don't support capabilities passed to a run inside a workflow, so there the capability has to be on the agent.

```python {title="batch_temporal.py" gateway_toggle="skip"}
from pydantic import BaseModel
from temporalio import workflow

from pydantic_ai import Agent
from pydantic_ai.capabilities import BatchModelRequests
from pydantic_ai.durable_exec.temporal import PydanticAIWorkflow, TemporalDurability


class Extraction(BaseModel):
    remote: bool
    countries: list[str]


agent = Agent(
    '{{ google:medium }}',
    name='extractor',
    output_type=Extraction,
    instructions='Extract the remote-work details from this job posting.',
    capabilities=[BatchModelRequests(), TemporalDurability()],
)


@workflow.defn
class ExtractPosting(PydanticAIWorkflow):  # (1)!
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, posting: str) -> Extraction:
        result = await agent.run(posting)
        return result.output
```

1. Start one workflow per posting: the requests of the workflows running on a worker share batches, and no worker slot is held while a batch is processed. See [Temporal](durable_execution/temporal.md) for running the worker and starting workflows.

With [DBOS](durable_execution/dbos.md) or [Prefect](durable_execution/prefect.md), put [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability] or [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability] next to `BatchModelRequests` instead, and run the agent in a DBOS workflow or a Prefect flow. Under Prefect, the waits between checks are plain sleeps in the flow, which a retried flow run sleeps again.

Once the step that submitted a request has completed, replaying or recovering the workflow reuses its recorded result, so the request isn't submitted again. The one exception is a step that's retried after the provider created its batch but before the step completed, for example because the worker crashed, or because the step timed out: like any model request, the retry sends the request again, which submits it in another batch. A request waits for its batch to be submitted inside its step, so keep `max_wait` well below the step's timeout (for example, the Temporal model activity's `start_to_close_timeout`). For the same reason, `max_wait=None` doesn't suit durable execution: [`flush()`][pydantic_ai.capabilities.BatchModelRequests.flush] can't be called from workflow code. A [resubmission](#failed-requests) happens inside the step that found the failure, with the same exception.

Other engines record less:

- With the durability capabilities in [Pydantic AI Harness](harness/index.md) for AWS Lambda, Absurd, pgtask and Render Workflows, submitting a request and each check on its batch are durable steps too, but the waits between checks are plain sleeps in the process running the agent. That process has to stay up for as long as the batch takes, within its invocation, claim or task timeout. On AWS Lambda, Absurd and pgtask, a recovered run continues with the same batch, after sleeping again through the waits it already did. A retried Render Workflows entry task, or a run continued from a [Step Persistence](harness/step-persistence.md) snapshot taken while it waited, submits the request in a new batch instead.
- Restate and Airflow record model requests through model wrappers of their own, which batched requests don't go through, so batching isn't recorded there and recovery submits the request again, in a new batch that's billed too.

To wait out long batches on any of these, add [`PendingModelResponse`](#ending-a-run-while-it-waits) to the `output_type` and schedule the next run from the returned message history yourself.

## Errors and resubmission {#failed-requests}

A request that fails in a batch raises the same error, including its category such as [`ModelRateLimitError`][pydantic_ai.exceptions.ModelRateLimitError] or [`ModelContextWindowExceededError`][pydantic_ai.exceptions.ModelContextWindowExceededError], as it would outside one, and fails only its own run.

A request that failed on the provider's side is submitted again in a new batch, as a provider SDK retries a live request. That's a request that expired before the provider got to it, including a Google job that expired, or that's missing from the batch's results, and a request that failed with a rate limit, timeout, conflict, overloaded or other server error. Up to `max_resubmissions` (default 2) new batches are tried, after which the run raises the request's error. A request that failed in a way that would repeat, such as an invalid request, a context window overflow or an exhausted quota, raises its error straight away. So does a request whose batch was cancelled: providers don't cancel batches of their own accord, so a cancelled batch, for example from the provider's console, means someone wanted its work stopped. With `max_resubmissions=0`, every error is raised.

A batch the provider refuses to create, for example because too many requests are already waiting in the project's batches, fails all of its requests with the provider's error, which isn't resubmitted: wait for earlier batches to end, or submit fewer requests at a time.

A check on a batch that fails the way a live request is retried on, for example with a dropped connection or a server error, says nothing about the batch, so the request keeps waiting and the batch is checked again at the next poll. After 3 failed checks in a row, the next failure raises the error.

The number of resubmissions and of failed checks is recorded in the message history, so a run resumed from a history whose batch has failed submits the request again, rather than reading the same failed result every time it's resumed. A resubmitted request waits for its new batch as usual, or, with [`PendingModelResponse`](#ending-a-run-while-it-waits), ends the run with a new `PendingModelResponse`.

## Direct batch requests

[`model_request_batch`][pydantic_ai.direct.model_request_batch] sends a list of [`BatchRequest`][pydantic_ai.models.batch.BatchRequest]s through a model's batch API and waits for all of them. Every request can have its own messages, settings and [request parameters][pydantic_ai.models.ModelRequestParameters], such as tools and output schema. It returns, in request order, each request's response or the [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] it failed with:

```python {title="batch_direct.py" gateway_toggle="skip"}
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_batch
from pydantic_ai.models.batch import BatchRequest

INSTRUCTIONS = 'Classify the sentiment of this product review as positive, negative or mixed. Answer with one word.'
reviews = {
    'review-1': 'Arrived quickly and works perfectly.',
    'review-2': 'Broke after two days, and support never replied.',
}


async def main():
    responses = await model_request_batch(
        '{{ anthropic:medium }}',
        [
            BatchRequest(
                custom_id=review_id,
                messages=[ModelRequest.user_text_prompt(review, instructions=INSTRUCTIONS)],
                model_settings={'max_tokens': 10},
            )
            for review_id, review in reviews.items()
        ],
    )
    for review_id, response in zip(reviews, responses):
        if isinstance(response, Exception):
            print(f'{review_id} failed: {response}')
        else:
            print(f'{review_id}: {response.text}')
            #> review-1: positive
            #> review-2: negative
```

Requests are split into as many batches as the provider requires, by count and size. Unlike with `BatchModelRequests`, `max_requests` defaults to `None`, which takes the provider's own limit. `poll_interval` and `max_poll_interval` set how often batches are checked on, as for `BatchModelRequests`. Requests a batch didn't run, or that failed with a transient error, are [submitted again](#failed-requests) in new batches, up to `max_resubmissions` times, and the call returns once every request has a response or an error that resubmitting wouldn't fix.

A request's own error is returned in its place, but an error creating a batch, checking on one or downloading its results isn't a request's error: the call raises it, after cancelling the batches it created that haven't ended. A check that fails the way a live request is retried on is first tried again at the next poll, as with `BatchModelRequests`. Cancelling the call cancels the batches that haven't ended too.

A request's `request_timeout` setting raises a [`UserError`][pydantic_ai.exceptions.UserError], since one request in a batch can't be timed out separately: to limit how long the call waits, cancel it, for example with `asyncio.timeout`. Unlike [`model_request`][pydantic_ai.direct.model_request], `model_request_batch` and the `batch_*` methods below aren't [traced](#tracing).

### The low-level `Model` API {#submit-now-collect-later}

To manage batch jobs yourself, use the `batch_*` methods of [`Model`][pydantic_ai.models.Model] directly. [`batch_encode`][pydantic_ai.models.Model.batch_encode] turns a `BatchRequest` into the provider's format, and [`batch_create`][pydantic_ai.models.Model.batch_create] submits encoded requests and returns a [`Batch`][pydantic_ai.models.batch.Batch] handle. The handle holds only plain data, so you can store it, together with your requests' messages, settings and parameters, which are needed to read the results:

```python {title="batch_submit_collect.py" gateway_toggle="skip"}
import asyncio

from pydantic import TypeAdapter

from pydantic_ai import ModelRequest
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models import infer_model
from pydantic_ai.models.batch import Batch, BatchRequest

INSTRUCTIONS = 'Classify the sentiment of this product review as positive, negative or mixed. Answer with one word.'
reviews = {
    'review-1': 'Arrived quickly and works perfectly.',
    'review-2': 'Broke after two days, and support never replied.',
}
batch_adapter = TypeAdapter(Batch)
model = infer_model('{{ anthropic:medium }}')
requests = [
    BatchRequest(custom_id=review_id, messages=[ModelRequest.user_text_prompt(review, instructions=INSTRUCTIONS)])
    for review_id, review in reviews.items()
]


async def submit() -> bytes:
    batch = await model.batch_create([await model.batch_encode(request) for request in requests])  # (1)!
    return batch_adapter.dump_json(batch)  # (2)!


async def collect(stored_batch: bytes) -> bool:
    batch = await model.batch_status(batch_adapter.validate_json(stored_batch))  # (3)!
    if batch.status != 'ended':
        return False
    requests_by_id = {request.custom_id: request for request in requests}
    for result in await model.batch_results(batch):
        try:
            text = model.batch_decode(result, requests_by_id[result.custom_id]).text
        except ModelAPIError as error:  # (4)!
            text = f'failed: {error}'
        print(f'{result.custom_id}: {text}')
        #> review-1: positive
        #> review-2: negative
    return True


async def main():
    stored = await submit()
    while not await collect(stored):  # (5)!
        await asyncio.sleep(600)
```

1. The requests passed to one `batch_create` call must share a [`group`][pydantic_ai.models.batch.EncodedBatchRequest.group] and fit in one batch, within [`batch_max_requests`][pydantic_ai.models.Model.batch_max_requests] and [`batch_max_bytes`][pydantic_ai.models.Model.batch_max_bytes]; otherwise it raises a `UserError`. `model_request_batch` does that grouping and splitting for you.
2. Store this, for example in a database, and load it wherever you collect the results.
3. Returns the batch's current state, which is the handle to pass to `batch_results` once it has ended.
4. `batch_decode` raises the [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] the request failed with, if it did. Resubmitting failed requests is up to you, as is handling a request the batch ended without a result for.
5. Or check from a scheduled job.

[`batch_cancel`][pydantic_ai.models.Model.batch_cancel] asks the provider to cancel a batch. `batch_create` doesn't retry a failed submission, because the provider may have created the batch before the connection failed: check the provider's batches before submitting the requests again.

## Details

### Which requests share a batch

Each model request is batched with the model that serves it, however that model was chosen: the agent's model, one passed to the run as `model=`, one a capability's `get_model` such as [`SelectModel`](capabilities/select-model.md) selects, or one a `before_model_request` hook swaps in, whatever the order of the capabilities. A run that switches models partway through puts each request in a batch for the model serving it, and a switch to a model without a [batch API](#supported-models) raises a [`UserError`][pydantic_ai.exceptions.UserError] at that request. A [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] has no batch API of its own, so batching one raises a `UserError` too.

Requests go into separate batches when the provider can't take them in one: when they use different credentials or endpoints, with OpenAI and Google when they use different models, and with Anthropic when they need different beta flags. Anthropic batches can mix models and settings otherwise. A batch is also split when it would exceed the provider's size limit, [`Model.batch_max_bytes`][pydantic_ai.models.Model.batch_max_bytes], and the provider's limit on requests per batch, [`Model.batch_max_requests`][pydantic_ai.models.Model.batch_max_requests], caps `max_requests`; each provider's page lists its limits. A request too large for any batch raises a `UserError` when it's made.

Since a batch is submitted, checked and collected with one credential, batching refuses clients whose credential can vary or that it can't see: an OpenAI client with an API key callback or a workload identity, an HTTP client with an `auth` of its own, and per-request `extra_headers`. Configure the credential on the provider's client instead. Other settings that only apply to a single live request, such as a per-request `timeout`, raise a `UserError` too; configure them on the provider's client. [`max_retries`][pydantic_ai.settings.ModelSettings.max_retries] is ignored, as the provider runs a batched request itself and no SDK client retries it. See each provider's page for its other limitations.

OpenAI stores batched responses by default, as it does other responses, so `openai_previous_response_id` works across batches; set `openai_store=False` to opt out.

### Request timeouts

A [`request_timeout`][pydantic_ai.settings.ModelSettings.request_timeout] in the model settings covers the whole wait, from when the request is queued until its result arrives. When it passes, the run raises [`ModelRequestTimeout`][pydantic_ai.exceptions.ModelRequestTimeout] and the request is withdrawn, which cancels its batch if no other request is waiting on it.

### Streaming and event loops

A batch result arrives all at once, so a streamed run, like one with [`run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] or [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events], an [`event_stream_handler`](agent.md#streaming-all-events) or a [UI adapter](ui/overview.md), waits for its batch as any run does and then gets each part of the response in one event.

Batching needs an `asyncio` event loop, so it doesn't run under Trio. One `BatchModelRequests` instance can serve runs on several event loops, for example in several threads: each loop's requests go into batches of their own.

### Model wrappers and hooks

`BatchModelRequests` sends requests through the model's batch API methods rather than its `request()`. So a model wrapper inside it, from a capability listed after it, never has `request()` called. A wrapper outside it, from a capability listed before it, has `request()` called for the submission and again for every check of the batch, receiving a response with `state='suspended'` until the result arrives. Put behavior that should run once per model request, like counting or metering requests, in a [`wrap_model_request`](capabilities/custom.md#model-request-hooks) hook instead: model request hooks run once per request, wherever the capability is listed.

### Cancellation and `async with`

Cancelling a run that's waiting abandons its request: Pydantic AI cancels the provider batch once every request it submitted in that batch has been abandoned, and requests the provider already completed may still be billed. A batch submitted by another process, or by another `BatchModelRequests` instance, is left running, since a message history can't be trusted to say which other requests share the batch.

A request a run left for a later run, by ending with a [`PendingModelResponse`](#ending-a-run-while-it-waits), keeps its batch running too. A later run that resumes it and is then cancelled leaves the batch running, since the same history may be resumed again elsewhere. A request is only left for a later run once the run has ended with its `PendingModelResponse` as the output. A pending response that an [`after_model_request`](hooks.md#model-request-hooks) hook rejects, by raising or by returning another response, isn't left for a later run: its request is abandoned like a cancelled run's. So is one whose run fails or is cancelled afterwards, ends with another output, for example from an [`after_run`](hooks.md#run-hooks) hook, or is iterated with [`iter()`][pydantic_ai.agent.Agent.iter] and left before it ends.

Requests waiting on the same submission or check share one task. Using the capability as an async context manager, `async with batching:`, cancels those tasks when it exits, and requests still queued or waiting on them fail with a `UserError`. A request waiting between two checks of its batch isn't affected: its next check runs as usual.

### Tracing

With [instrumentation](logfire.md) enabled, a batch is traced once, in a trace of its own, rather than under whichever run's request happened to start it. Its submission is a `batch <model>` span that links to the `chat` span of every request in the batch, and each check is a `batch check <model>` span within that trace, holding the provider's HTTP requests. Each request's span records the batch's ID as `pydantic_ai.batch.id` and links to the `batch` span, including in a later run that resumes waiting from the message history. That's the `chat` span, or under Temporal, the model activity's span within it.

### Retention

Providers only keep a batch's results for a limited time. OpenAI [deletes a batch's output file](https://developers.openai.com/api/docs/guides/batch) 30 days after the batch completes. Anthropic makes [results available](https://platform.claude.com/docs/en/build-with-claude/batch-processing) for 29 days after the batch was created, counted from its creation rather than from when it ended; the batch itself can still be checked after that. Google [stores results](https://ai.google.dev/gemini-api/docs/batch-api) for 6 weeks by default.

A run resumed from a history whose batch the provider no longer has raises [`SuspendedResponseExpired`][pydantic_ai.exceptions.SuspendedResponseExpired], as a resumed [OpenAI background response](models/openai.md#background-mode) does. The request isn't submitted again, since the provider may have run and billed it: to run it again, resume from the history without the pending response at its end. A key from another project or workspace also gets this error, since the provider won't show it the batch.
