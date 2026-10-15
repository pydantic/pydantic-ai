---
title: Batch Processing
description: "Send agent runs and model requests through provider batch APIs, at half the price, with Pydantic AI's BatchModel and model_request_batch."
---

# Batch Processing

Anthropic, OpenAI and Google offer batch APIs: you submit many requests as one job, and the provider processes them within 24 hours (Google: 48 hours) at about half the standard price. That suits work that isn't waiting on a user, such as evals, data extraction and classification pipelines, and offline data generation.

Pydantic AI supports batch APIs at three levels:

- [`BatchModel`](#agents) sends an agent's model requests through the batch API, without changing how you run the agent.
- [`model_request_batch`](#direct-batch-requests) sends a list of model requests as a batch and waits for the responses.
- The [`batch_*` methods on `Model`](#submit-now-collect-later) submit a batch and collect its results separately, for example from another process.

## Agents

Wrap a model in [`BatchModel`][pydantic_ai.models.batch.BatchModel] and run the agent as usual. Requests made at about the same time in the same process, by any number of agent runs, are collected into one batch:

```python {title="batch_agent.py" gateway_toggle="skip"}
import asyncio
from datetime import timedelta

from pydantic_ai import Agent
from pydantic_ai.models.batch import BatchModel

model = BatchModel('{{ anthropic:medium }}', batch_settings={'max_requests': 500, 'max_wait': timedelta(seconds=10)})
agent = Agent(model, instructions='Answer in one sentence.')


async def main():
    questions = ['What is the capital of France?', 'What is the capital of Italy?']
    results = await asyncio.gather(*(agent.run(question) for question in questions))
    for result in results:
        print(result.output)
        #> The capital of France is Paris.
        #> The capital of Italy is Rome.
```

Tools, structured output, output validation and retries all work as usual; each model request in a run goes into whichever batch is being collected when it's made. A run that needs several model requests, for example to call tools, takes several batch round trips.

[`BatchSettings`][pydantic_ai.models.batch.BatchSettings] controls when a batch is submitted: when it holds `max_requests` requests (default 100, capped by the provider's limit), or when its oldest request has waited `max_wait` (default 30 seconds), whichever comes first. With `max_wait=None`, a batch is only submitted when it's full or when you call [`BatchModel.flush()`][pydantic_ai.models.batch.BatchModel.flush]. Each request then waits for its batch to end, checking after `poll_interval` (default 30 seconds) and then at intervals that grow up to `max_poll_interval` (default 10 minutes).

Streaming isn't supported: a batch result arrives all at once.

### Waiting and resuming

While a request waits for its batch, the run's message history ends in a [`ModelResponse`][pydantic_ai.messages.ModelResponse] with `state='suspended'`, which records the batch. If the process stops while the run is waiting, run the agent again with that message history, in this process or another, and it picks the same batch back up instead of submitting the request again:

```python {title="batch_resume.py" gateway_toggle="skip"}
from pydantic_ai import Agent, ModelMessagesTypeAdapter


async def resume(agent: Agent, stored_messages: bytes) -> str:
    result = await agent.run(message_history=ModelMessagesTypeAdapter.validate_json(stored_messages))
    return result.output
```

To get hold of that message history while the run is waiting, use [message history storage](message-history.md) or run the agent under [durable execution](#durable-execution).

Cancelling a run that's waiting abandons its request. Pydantic AI cancels the provider batch once every request in it has been abandoned in the same process; requests the provider already completed may still be billed.

### Durable execution

Under [durable execution](durable_execution/overview.md) with Temporal, DBOS or Prefect, `BatchModel` needs no extra setup. Submitting a request and every check on its batch run as separate durable steps, so a run that's interrupted while its batch is processed, for example by a deployment, continues from the same batch when it's recovered. Under Temporal, the waits between checks are durable timers too. Requests are collected into batches per worker process.

A request waits for its batch to be submitted inside its durable step, so keep `max_wait` well below the step's timeout (under Temporal, the model activity's `start_to_close_timeout`), or a slow submission can time out and be retried, submitting the request again. For the same reason, `max_wait=None` doesn't suit durable execution: [`flush()`][pydantic_ai.models.batch.BatchModel.flush] can't be called from workflow code.

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

Requests are split into as many batches as the provider requires, or as `batch_settings` limits them to; it also sets how often batches are checked on, as for `BatchModel`. Cancelling the call cancels the batches that haven't ended.

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

Settings that only apply to a single live request, such as a per-request `timeout` or `extra_headers`, raise a [`UserError`][pydantic_ai.exceptions.UserError]; configure them on the provider's client instead. See each provider's page for its other limitations.
