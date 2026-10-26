---
title: Batch Processing
description: "Send agent runs and model requests through provider batch APIs, at half the price, with Pydantic AI's BatchModelRequests capability and model_request_batch."
---

# Batch Processing

Anthropic, OpenAI and Google offer batch APIs: you submit many requests as one job, and the provider processes them within 24 hours (Google: 48 hours) at about half the standard price. That suits work that isn't waiting on a user, such as evals, data extraction and classification pipelines, and offline data generation.

Pydantic AI supports batch APIs at three levels:

- The [`BatchModelRequests`](#agents) [capability](capabilities/overview.md) sends an agent's model requests through the batch API, without changing how you run the agent.
- [`model_request_batch`](#direct-batch-requests) sends a list of model requests as a batch and waits for the responses.
- The [`batch_*` methods on `Model`](#submit-now-collect-later) submit a batch and collect its results separately, for example from another process.

## Agents

Add a [`BatchModelRequests`][pydantic_ai.capabilities.BatchModelRequests] capability to an agent and run it as usual. Requests made at about the same time in the same process, by any number of agent runs, are collected into one batch:

```python {title="batch_agent.py" gateway_toggle="skip"}
import asyncio
from datetime import timedelta

from pydantic_ai import Agent
from pydantic_ai.capabilities import BatchModelRequests

batching = BatchModelRequests(max_wait=timedelta(minutes=1))
agent = Agent('{{ anthropic:medium }}', instructions='Answer in one sentence.', capabilities=[batching])


async def main():
    questions = ['What is the capital of France?', 'What is the capital of Italy?']
    results = await asyncio.gather(*(agent.run(question) for question in questions))
    for result in results:
        print(result.output)
        #> The capital of France is Paris.
        #> The capital of Italy is Rome.
```

Tools, structured output, output validation and retries all work as usual; each model request in a run goes into whichever batch is being collected when it's made. A run that needs several model requests, for example to call tools or to retry after an output validator raised [`ModelRetry`][pydantic_ai.exceptions.ModelRetry], takes several batch round trips, each of which can take as long as the provider's batch window. [Usage limits](agent.md#usage-limits) apply to each run as usual. A request that fails only fails its own run, not the others in its batch: pass `return_exceptions=True` to `asyncio.gather` to get each run's result or error.

A batch is submitted when it holds `max_requests` requests (default 1,000), when the next request would take it past the provider's size limit, or when its oldest request has waited `max_wait` (default 30 seconds), whichever comes first. Later requests never extend `max_wait`. With `max_wait=None`, a batch is only submitted when it's full or when you call [`flush()`][pydantic_ai.capabilities.BatchModelRequests.flush]. Raise `max_requests` when you make many more requests at a time, to submit fewer, larger batches: providers limit how many batches can be in progress at once. Each request then waits for its batch to end, checking after `poll_interval` (default 30 seconds) and then at intervals that grow up to `max_poll_interval` (default 10 minutes). A run that takes several batch round trips can wait up to `max_wait` plus the time until the next check for each, on top of the provider's own time, so lower `max_poll_interval` when that adds up, at the cost of more status requests.

A [`request_timeout`][pydantic_ai.settings.ModelSettings.request_timeout] in the model settings covers the whole wait, from when the request is queued until its result arrives: when it passes, the run raises [`ModelRequestTimeout`][pydantic_ai.exceptions.ModelRequestTimeout] and the request is withdrawn, which cancels its batch if no other request is waiting on it.

A batch result arrives all at once, so a streamed run, like one with [`run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] or [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events], an [`event_stream_handler`](agent.md#streaming-all-events) or a [UI adapter](ui/overview.md), waits for its batch as any run does and then gets each part of the response in one event. Batching needs an `asyncio` event loop, so it doesn't run under Trio. One `BatchModelRequests` instance can serve runs on several event loops, for example in several threads: each loop's requests go into batches of their own.

`BatchModelRequests` sends requests through the model's batch API methods rather than its `request()`, so a model wrapper inside it, from a capability listed after it, never has `request()` called. Put behavior that should run once per model request in a [`wrap_model_request`](capabilities/custom.md#model-request-hooks) hook instead.

Requests waiting on the same submission or check share one task. Using the capability as an async context manager, `async with batching:`, cancels those tasks when it exits, and requests still waiting on them fail with a [`UserError`][pydantic_ai.exceptions.UserError]. A request waiting between two checks of its batch isn't affected: its next check runs as usual.

### Which requests share a batch

Each model request is batched with the model that serves it, however that model was chosen: the agent's model, one passed to the run as `model=`, one a capability's `get_model` such as [`SelectModel`](capabilities/select-model.md) selects, or one a `before_model_request` hook swaps in, whatever the order of the capabilities. A run that switches models partway through puts each request in a batch for the model serving it, and a switch to a model without a [batch API](#supported-models) raises a [`UserError`][pydantic_ai.exceptions.UserError] at that request. A [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] has no batch API of its own, so it can't be batched.

Requests go into separate batches when the provider can't take them in one: when they use different credentials or endpoints, with OpenAI and Google when they use different models, and with Anthropic when they need different beta flags. Anthropic batches can mix models and settings otherwise. A batch is also split when it would exceed the provider's size limit ([`Model.batch_max_bytes`][pydantic_ai.models.Model.batch_max_bytes]: OpenAI 200 MB, Anthropic 256 MB, Google 2 GB), and the provider's limit on requests per batch caps `max_requests`: OpenAI takes 50,000 and Anthropic 100,000, while Google only limits a batch's size. A request too large for any batch raises a `UserError` when it's made.

A request that fails in a batch raises the same error, including its category such as [`ModelRateLimitError`][pydantic_ai.exceptions.ModelRateLimitError] or [`ModelContextWindowExceededError`][pydantic_ai.exceptions.ModelContextWindowExceededError], as it would outside one. OpenAI stores batched responses by default, as it does other responses, so `openai_previous_response_id` works across batches; set `openai_store=False` to opt out.

Since a batch is submitted, checked and collected with one credential, batching refuses clients whose credential can vary or that it can't see: an OpenAI client with an API key callback or a workload identity, an HTTP client with an `auth` of its own, and per-request `extra_headers`. Configure the credential on the provider's client instead. See each provider's page for its other limitations.

### Failed requests

A request that failed on the provider's side is submitted again in a new batch, as a provider SDK retries a live request. That's a request that expired before the provider got to it, including a Google job that expired, or that's missing from the batch's results, and a request that failed with a rate limit, timeout, conflict, overloaded or other server error. Up to `max_resubmissions` (default 2) new batches are tried, after which the run raises the request's error. A request that failed in a way that would repeat, such as an invalid request, a context window overflow or an exhausted quota, raises its error straight away. So does a request whose batch was canceled: providers don't cancel batches of their own accord, so a canceled batch, for example from the provider's console, means someone wanted its work stopped. With `max_resubmissions=0`, every error is raised.

A batch the provider refuses to create, for example because too many requests are already waiting in the project's batches, fails all of its requests with the provider's error, which isn't resubmitted: wait for earlier batches to end, or submit fewer requests at a time.

A check on a batch that fails the way a live request is retried on, for example with a dropped connection or a server error, says nothing about the batch, so the request keeps waiting and the batch is checked again at the next poll, up to 3 times in a row before the run raises the error.

The count is recorded in the message history, so a run resumed from a history whose batch has failed this way submits the request again, rather than reading the same failed result every time it's resumed. A resubmitted request waits for its new batch as usual, or, with [`PendingModelResponse`](#ending-a-run-while-it-waits), ends the run with a new `PendingModelResponse`.

To try another model once a request's error reaches the run, add [`ModelFallback`](capabilities/model-fallback.md): the request then goes into a batch on the next model in its chain. When a run resumed from history finds that the batch it was waiting on failed, it also moves on to the next model, rather than starting over with a new batch on the first.

### Sharing batches between agents

Each `BatchModelRequests` instance collects its own batches. Add the same instance to several agents, or pass it to runs with `capabilities=`, and their requests share batches:

```python {title="batch_shared.py" gateway_toggle="skip"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.capabilities import BatchModelRequests

batching = BatchModelRequests()
geographer = Agent('{{ anthropic:medium }}', instructions='Answer in one sentence.', capabilities=[batching])
historian = Agent('{{ anthropic:medium }}', instructions='Answer in one sentence.')


async def main():
    france, italy = await asyncio.gather(
        geographer.run('What is the capital of France?'),
        historian.run('What is the capital of Italy?', capabilities=[batching]),  # (1)!
    )
    print(france.output)
    #> The capital of France is Paris.
    print(italy.output)
    #> The capital of Italy is Rome.
```

1. Runs of an agent without the capability can be batched one by one.

### Agent specs {#agent-specs}

`BatchModelRequests` can be declared in an [agent spec](agent-spec.md), with its settings as keyword arguments. Give each duration as a number of seconds or as an ISO 8601 duration, like `PT5M`:

```yaml {title="batch_agent.yaml" test="skip"}
model: {{ anthropic:medium }}
instructions: Answer in one sentence.
capabilities:
  - BatchModelRequests:
      max_requests: 5000
      max_wait: PT5M
```

Each agent built from a spec gets a `BatchModelRequests` instance of its own: its runs share batches with each other, but not with the runs of other agents, even ones built from the same spec. To share batches between agents built from specs, leave the capability out of the specs and pass one instance to their runs with `capabilities=`, as above.

### Ending a run while it waits

A batch can take up to a day, which is longer than many processes should wait. Add [`PendingModelResponse`][pydantic_ai.output.PendingModelResponse] to the agent's `output_type`, and a run ends as soon as its request has been submitted, with a `PendingModelResponse` as its output. The run's message history ends in the pending response, which records the batch, so you can store the history and run the agent again with it later, in this process or another. That run checks on the batch once: if the result has arrived, the run continues as usual, and otherwise it ends with a `PendingModelResponse` again.

```python {title="batch_pending.py" gateway_toggle="skip"}
from pydantic import BaseModel

from pydantic_ai import Agent, ModelMessagesTypeAdapter, PendingModelResponse
from pydantic_ai.capabilities import BatchModelRequests


class City(BaseModel):
    city: str
    country: str


agent = Agent(
    '{{ anthropic:medium }}', output_type=[City, PendingModelResponse], capabilities=[BatchModelRequests()]
)


async def submit(question: str) -> bytes:
    result = await agent.run(question)
    return result.all_messages_json()  # (1)!


async def collect(stored_messages: bytes) -> City | bytes:
    result = await agent.run(message_history=ModelMessagesTypeAdapter.validate_json(stored_messages))
    if isinstance(result.output, PendingModelResponse):
        return result.all_messages_json()  # (2)!
    return result.output


async def main():
    stored = await submit('Where is the Eiffel Tower?')
    # Later, for example from a scheduled job:
    print(await collect(stored))
    #> city='Paris' country='France'
```

1. Store this, for example in a database, and load it wherever you collect the results.
2. The batch is still being processed: store the new history in place of the old one, and check again later.

Submitting still waits for the batch to be submitted, up to `max_wait` (30 seconds by default) for more requests to join it. A script that starts its runs together, submits them and exits can set a lower `max_wait`, or `max_wait=None` and call [`flush()`][pydantic_ai.capabilities.BatchModelRequests.flush] once every run's request is queued. With `max_wait=None`, every later request waits for a `flush()` too, like a tool call's follow-up request or the next request of a run that collected its result, so collect with a capability that has a `max_wait`. Collect with `BatchModelRequests` too: resuming the history without it raises a [`UserError`][pydantic_ai.exceptions.UserError], rather than sending the waiting request again as a new request at the full price.

A run that needs several model requests, for example to call tools, ends with a `PendingModelResponse` for each of them. Each run in the cycle counts the response it ends with or continues from in [`usage.requests`][pydantic_ai.usage.RunUsage.requests], so adding up the usage of every run counts a request once per run, while its tokens are only counted by the run that received them. This works for any request the provider keeps working on after it's been sent, including [OpenAI background mode](models/openai.md#background-mode).

The process that collects a batch doesn't need the API key that submitted it: the provider decides who may check on a batch, so a rotated key of the same project or workspace works. Checks of one batch within `poll_interval` of each other share one status request, and its results are downloaded once while they're cached, so collecting many stored histories costs about one check per batch rather than one per run. The cache keeps the results of the last 32 batches that ended, up to 256 MB of them, so collect the histories of one batch together rather than interleaved with those of many others.

Without `PendingModelResponse` in the output type, a run waits for its batch, and you only get its message history once it ends. If your process may stop while a batch is processed, end runs with a `PendingModelResponse` or use [durable execution](#durable-execution). Cancelling a run that's waiting abandons its request: Pydantic AI cancels the provider batch once every request it submitted in that batch has been abandoned, and requests the provider already completed may still be billed. A batch submitted by another process, or by another `BatchModelRequests` instance, is left running, since a message history can't be trusted to say which other requests share the batch. So is a batch a run left for a later run by ending with a `PendingModelResponse`. A pending response that an [`after_model_request`](hooks.md#model-request-hooks) hook rejects, by raising or by returning another response, isn't left for a later run: its request is abandoned like a cancelled run's.

### Retention

Providers only keep a batch's results for a limited time. OpenAI [deletes a batch's output file](https://developers.openai.com/api/docs/guides/batch) 30 days after the batch completes. Anthropic makes [results available](https://platform.claude.com/docs/en/build-with-claude/batch-processing) for 29 days after the batch was created, counted from its creation rather than from when it ended; the batch itself can still be checked after that. Google [stores results](https://ai.google.dev/gemini-api/docs/batch-api) for 6 weeks by default. Collect results well within these windows. A run resumed from a history whose batch the provider no longer has raises [`SuspendedResponseExpired`][pydantic_ai.exceptions.SuspendedResponseExpired], as a resumed [OpenAI background response](models/openai.md#background-mode) does. The request isn't submitted again, since the provider may have run and billed it: to run it again, resume from the history without the pending response at its end. A key from another project or workspace also gets this error, since the provider won't show it the batch.

### Tracing

With [instrumentation](logfire.md) enabled, a batch is traced once, in a trace of its own, rather than under whichever run's request happened to start it. Its submission is a `batch <model>` span that links to the `chat` span of every request in the batch, and each check is a `batch check <model>` span within that trace, holding the provider's HTTP requests. Each request's span records the batch's ID as `pydantic_ai.batch.id` and links to the `batch` span, including in a later run that resumes waiting from the message history. That's the `chat` span, or under Temporal, the model activity's span within it.

### Durable execution

Under [durable execution](durable_execution/overview.md) with Temporal, DBOS or Prefect, add `BatchModelRequests` to the agent alongside its durability capability. Submitting a request and every check on its batch run as separate durable steps, so a run that's interrupted while its batch is processed, for example by a deployment, continues from the same batch when it's recovered. Under Temporal and DBOS, the waits between checks are durable too. Requests are collected into batches per worker process: the requests of concurrent workflow runs on one worker share batches. Temporal and DBOS don't support capabilities passed to a run inside a workflow, so there the capability has to be on the agent.

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

With other durability capabilities, such as those in [Pydantic AI Harness](harness/index.md) for AWS Lambda, Absurd, pgtask and Render Workflows, submitting a request and each check on its batch are durable steps too, but the waits between checks are plain sleeps in the process running the agent. That process has to stay up for as long as the batch takes, which can be up to a day, within its invocation, claim or task timeout. On AWS Lambda, Absurd and pgtask, a recovered run continues with the same batch, after sleeping again through the waits it already did. A retried Render Workflows entry task, or a run continued from a [Step Persistence](harness/step-persistence.md) snapshot taken while it waited, submits the request in a new batch instead. Restate and Airflow record model requests through model wrappers of their own, which batched requests don't go through, so batching isn't recorded there and recovery submits the request again. To wait out long batches on any of these, add [`PendingModelResponse`](#ending-a-run-while-it-waits) to the `output_type` and schedule the next run from the returned message history yourself.

Once the step that submitted a request has completed, replaying or recovering the workflow reuses its recorded result, so the request isn't submitted again. The one exception is a step that's retried after the provider created its batch but before the step completed, for example because the worker crashed, or because the step timed out: like any model request, the retry sends the request again, which submits it in another batch. A request waits for its batch to be submitted inside its step, so keep `max_wait` well below the step's timeout (for example, the Temporal model activity's `start_to_close_timeout`). For the same reason, `max_wait=None` doesn't suit durable execution: [`flush()`][pydantic_ai.capabilities.BatchModelRequests.flush] can't be called from workflow code. A [resubmission](#failed-requests) happens inside the step that found the failure, with the same exception.

## Direct batch requests

[`model_request_batch`][pydantic_ai.direct.model_request_batch] sends a list of [`BatchRequest`][pydantic_ai.models.batch.BatchRequest]s through a model's batch API and waits for all of them. Every request can have its own messages, settings and [request parameters][pydantic_ai.models.ModelRequestParameters], such as tools and output schema. It returns, in request order, each request's response or the [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] it failed with:

```python {title="batch_direct.py" gateway_toggle="skip"}
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_batch
from pydantic_ai.models.batch import BatchRequest


async def main():
    questions = {'france': 'What is the capital of France?', 'italy': 'What is the capital of Italy?'}
    responses = await model_request_batch(
        '{{ anthropic:medium }}',
        [
            BatchRequest(
                custom_id=key,
                messages=[ModelRequest.user_text_prompt(question)],
                model_settings={'max_tokens': 100},
            )
            for key, question in questions.items()
        ],
    )
    for key, response in zip(questions, responses):
        if isinstance(response, Exception):
            print(f'{key} failed: {response}')
        else:
            print(f'{key}: {response.text}')
            #> france: The capital of France is Paris.
            #> italy: The capital of Italy is Rome.
```

Requests are split into as many batches as the provider requires, by count and size, or as `max_requests` limits them to, and `poll_interval` and `max_poll_interval` set how often batches are checked on, as for `BatchModelRequests`. Requests a batch didn't run, or that failed with a transient error, are [submitted again](#failed-requests) in new batches, up to `max_resubmissions` times, and the call returns once every request has a response or an error that resubmitting wouldn't fix. Cancelling the call cancels the batches that haven't ended. A request's `request_timeout` setting raises a [`UserError`][pydantic_ai.exceptions.UserError], since one request in a batch can't be timed out separately: to limit how long the call waits, cancel it, for example with `asyncio.timeout`.

## Submit now, collect later

To submit a batch from one process and collect its results from another, use the `batch_*` methods of [`Model`][pydantic_ai.models.Model] directly. [`batch_create`][pydantic_ai.models.Model.batch_create] returns a [`Batch`][pydantic_ai.models.batch.Batch] handle that holds only plain data, so you can store it, together with your requests' settings and parameters, which are needed to read the results:

```python {title="batch_submit_collect.py" gateway_toggle="skip"}
from pydantic import TypeAdapter

from pydantic_ai import ModelRequest
from pydantic_ai.models import infer_model
from pydantic_ai.models.batch import Batch, BatchRequest

batch_adapter = TypeAdapter(Batch)
model = infer_model('{{ anthropic:medium }}')
requests = [
    BatchRequest(custom_id='france', messages=[ModelRequest.user_text_prompt('What is the capital of France?')]),
    BatchRequest(custom_id='italy', messages=[ModelRequest.user_text_prompt('What is the capital of Italy?')]),
]


async def submit() -> bytes:
    batch = await model.batch_create([await model.batch_encode(request) for request in requests])
    return batch_adapter.dump_json(batch)  # (1)!


async def collect(stored_batch: bytes) -> None:
    batch = await model.batch_status(batch_adapter.validate_json(stored_batch))
    if batch.status != 'ended':
        print('Still processing')
        return
    requests_by_id = {request.custom_id: request for request in requests}
    for result in await model.batch_results(batch):
        response = model.batch_decode(result, requests_by_id[result.custom_id])  # (2)!
        print(f'{result.custom_id}: {response.text}')
        #> france: The capital of France is Paris.
        #> italy: The capital of Italy is Rome.


async def main():
    await collect(await submit())
```

1. Store this, for example in a database, and load it wherever you collect the results.
2. Raises the [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] the request failed with, if it did.

[`batch_cancel`][pydantic_ai.models.Model.batch_cancel] asks the provider to cancel a batch. [`batch_create`][pydantic_ai.models.Model.batch_create] doesn't retry a failed submission, because the provider may have created the batch before the connection failed: check the provider's batches before submitting the requests again.

## Cost

Responses from a batch carry `provider_details['processing'] == 'batch'`, so [`ModelResponse.cost()`][pydantic_ai.messages.ModelResponse.cost], `usage.cost` and [usage limits](agent.md#usage-limits) use batch prices from [`genai-prices`](https://github.com/pydantic/genai-prices).

## Supported models

| Model | Notes |
|---|---|
| [`AnthropicModel`](models/anthropic.md#batch-api) | The Claude API's Message Batches. Not available on Bedrock, Vertex AI or Microsoft Foundry. |
| [`OpenAIResponsesModel`](models/openai.md#batch-api) | OpenAI's own API. Not available with `OpenAIChatModel` or Azure OpenAI. An OpenAI-compatible provider that serves the Batch API can [opt in](models/openai.md#batch-api). |
| [`GoogleModel`](models/google.md#batch-api) | The Gemini API. Not available on Vertex AI. |
| [`TestModel`][pydantic_ai.models.test.TestModel] and [`FunctionModel`][pydantic_ai.models.function.FunctionModel] | Run a batch's requests immediately and in memory, for [testing](testing.md). |

Whether a model has a batch API is a [profile](models/overview.md#models-and-providers) flag: [`anthropic_supports_message_batches`][pydantic_ai.profiles.anthropic.AnthropicModelProfile.anthropic_supports_message_batches], [`openai_supports_batch_api`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_supports_batch_api] or [`google_supports_batch_api`][pydantic_ai.profiles.google.GoogleModelProfile.google_supports_batch_api]. Pass `profile=` to the model to set it where the default is wrong for your endpoint.

Settings that only apply to a single live request, such as a per-request `timeout`, `max_retries` or `extra_headers`, raise a [`UserError`][pydantic_ai.exceptions.UserError]; configure them on the provider's client instead. See each provider's page for its other limitations.
