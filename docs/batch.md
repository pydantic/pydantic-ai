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

batching = BatchModelRequests(max_requests=500, max_wait=timedelta(seconds=10))
agent = Agent('{{ anthropic:medium }}', instructions='Answer in one sentence.', capabilities=[batching])


async def main():
    questions = ['What is the capital of France?', 'What is the capital of Italy?']
    results = await asyncio.gather(*(agent.run(question) for question in questions))
    for result in results:
        print(result.output)
        #> The capital of France is Paris.
        #> The capital of Italy is Rome.
```

Tools, structured output, output validation and retries all work as usual; each model request in a run goes into whichever batch is being collected when it's made. A run that needs several model requests, for example to call tools, takes several batch round trips.

The capability batches whichever model serves each request: the agent's model, one passed to the run as `model=`, or one a [capability](capabilities/select-model.md) or hook selects. That model needs a [batch API](#supported-models). Requests go into separate batches when the provider can't take them in one, for example when they use different credentials, or, with OpenAI and Google, different models.

A batch is submitted when it holds `max_requests` requests (default 100, capped by the provider's limit), or when its oldest request has waited `max_wait` (default 30 seconds), whichever comes first. With `max_wait=None`, a batch is only submitted when it's full or when you call [`flush()`][pydantic_ai.capabilities.BatchModelRequests.flush]. Each request then waits for its batch to end, checking after `poll_interval` (default 30 seconds) and then at intervals that grow up to `max_poll_interval` (default 10 minutes).

A [`request_timeout`][pydantic_ai.settings.ModelSettings.request_timeout] in the model settings covers the whole wait, from when the request is queued until its result arrives: when it passes, the run raises [`ModelRequestTimeout`][pydantic_ai.exceptions.ModelRequestTimeout] and the request is withdrawn, which cancels its batch if no other request is waiting on it.

Streaming isn't supported: a batch result arrives all at once. Batching needs an `asyncio` event loop, so it doesn't run under Trio.

`BatchModelRequests` sends requests through the model's batch API methods rather than its `request()`, so a model wrapper inside it, from a capability listed after it, never has `request()` called. Put behavior that should run once per model request in a [`wrap_model_request`](capabilities/custom.md#model-request-hooks) hook instead.

Requests waiting on the same submission or check share one task. Using the capability as an async context manager, `async with batching:`, cancels those tasks when it exits, and requests still waiting on them fail with a [`UserError`][pydantic_ai.exceptions.UserError].

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
```

1. Store this, for example in a database, and load it wherever you collect the results.
2. The batch is still being processed: store the new history in place of the old one, and check again later.

A run that needs several model requests, for example to call tools, ends with a `PendingModelResponse` for each of them. This works for any request the provider keeps working on after it's been sent, including [OpenAI background mode](models/openai.md#background-mode).

Without `PendingModelResponse` in the output type, a run waits for its batch, and you only get its message history once it ends. If your process may stop while a batch is processed, end runs with a `PendingModelResponse` or use [durable execution](#durable-execution). Cancelling a run that's waiting abandons its request: Pydantic AI cancels the provider batch once every request it submitted in that batch has been abandoned, and requests the provider already completed may still be billed. A batch submitted by another process, or by another `BatchModelRequests` instance, is left running, since a message history can't be trusted to say which other requests share the batch.

### Tracing

With [instrumentation](logfire.md) enabled, a batch is traced once, in a trace of its own, rather than under whichever run's request happened to start it. Its submission is a `batch <model>` span that links to the `chat` span of every request in the batch, and each check is a `batch check <model>` span within that trace, holding the provider's HTTP requests. Each request's span records the batch's ID as `pydantic_ai.batch.id` and links to the `batch` span, including in a later run that resumes waiting from the message history. That's the `chat` span, or under Temporal, the model activity's span within it.

### Durable execution

Under [durable execution](durable_execution/overview.md) with Temporal, DBOS or Prefect, add `BatchModelRequests` to the agent alongside its durability capability. Submitting a request and every check on its batch run as separate durable steps, so a run that's interrupted while its batch is processed, for example by a deployment, continues from the same batch when it's recovered. Under Temporal and DBOS, the waits between checks are durable too. Requests are collected into batches per worker process: the requests of concurrent workflow runs on one worker share batches. Temporal and DBOS don't support capabilities passed to a run inside a workflow, so there the capability has to be on the agent.

A request waits for its batch to be submitted inside its durable step, so keep `max_wait` well below the step's timeout (for example, the Temporal model activity's `start_to_close_timeout`), or a slow submission can time out and be retried, submitting the request again. For the same reason, `max_wait=None` doesn't suit durable execution: [`flush()`][pydantic_ai.capabilities.BatchModelRequests.flush] can't be called from workflow code.

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

Requests are split into as many batches as the provider requires, or as `max_requests` limits them to, and `poll_interval` and `max_poll_interval` set how often batches are checked on, as for `BatchModelRequests`. Cancelling the call cancels the batches that haven't ended. A request's `request_timeout` setting raises a [`UserError`][pydantic_ai.exceptions.UserError], since one request in a batch can't be timed out separately: to limit how long the call waits, cancel it, for example with `asyncio.timeout`.

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
| [`OpenAIResponsesModel`](models/openai.md#batch-api) | OpenAI's own API. Not available with `OpenAIChatModel` or OpenAI-compatible providers. |
| [`GoogleModel`](models/google.md#batch-api) | The Gemini API. Not available on Vertex AI. |
| [`TestModel`][pydantic_ai.models.test.TestModel] and [`FunctionModel`][pydantic_ai.models.function.FunctionModel] | Run a batch's requests immediately and in memory, for [testing](testing.md). |

Settings that only apply to a single live request, such as a per-request `timeout`, `max_retries` or `extra_headers`, raise a [`UserError`][pydantic_ai.exceptions.UserError]; configure them on the provider's client instead. See each provider's page for its other limitations.
