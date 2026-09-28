---
description: "Run Pydantic AI agents on open-weight Contrastive Language Models (CLM) served on your own hardware: typed answers with confidence scores for routing, guards and classification."
---

# Contrastive Language Models (CLM)

A [Contrastive Language Model](https://github.com/Contrastive-LM/CLM) (CLM) is an open-weight [decision model](decision.md): it answers typed questions about a text, each with a probability or a distribution over the options, rather than writing text. It scores each option by how well it matches the text, so it only ever picks between the options you give it. The reference checkpoint, [`Contrastive-LM/CLM-v0.1-8B`](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), is two small projection heads on top of a frozen [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) encoder, released under the Apache 2.0 license, and you run it on your own hardware.

[`ContrastiveModel`][pydantic_ai.models.contrastive.ContrastiveModel] is the Pydantic AI model class for CLMs, and a subclass of [`DecisionModel`][pydantic_ai.models.decision.DecisionModel], like [`TypeSafeModel`](typesafe.md). An agent built for one runs on the other by changing the model name. This page covers what is specific to CLMs: serving a checkpoint, connecting to it, and its limits.

!!! tip "Start with Decision models"
    **[Decision models](decision.md) is where to learn how an agent's output types, tools and message history map
    onto a CLM's questions**, and what the answers mean.

## Install

`ContrastiveModel` talks to the CLM server over HTTP and needs nothing beyond `pydantic-ai-slim` itself. The model runs in a separate server, `clm-serve`, from the [`contrastive-lm`](https://pypi.org/project/contrastive-lm/) package, which needs a GPU machine with [vLLM](https://docs.vllm.ai) for the encoder. It does not need to be the machine your agent runs on.

## Serving a checkpoint

Start the Qwen3-8B encoder with vLLM, then `clm-serve` in front of it:

```bash
pip install contrastive-lm

vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --max-model-len 2048 --port 8090 &
clm-serve
```

On first start, `clm-serve` downloads the `Contrastive-LM/CLM-v0.1-8B` heads from Hugging Face into `~/.cache/clm/` and serves them as `clm-latest` on `http://127.0.0.1:8700`. Where the weights come from and where they run is the server's to decide:

- `--ckpt PATH` serves a local checkpoint as `clm-latest` instead, such as heads you have [fine-tuned](https://github.com/Contrastive-LM/CLM/blob/main/docs/FINETUNING.md).
- `--model NAME=PATH` serves another checkpoint beside it under `NAME`, and `--ckpt-dir DIR` every `.pt` file in `DIR` under its file name.
- `--device cpu` or `--device cuda` picks where the heads run; by default they use CUDA when it is available. The encoder's device and dtype are vLLM's, set with `CUDA_VISIBLE_DEVICES` and `--dtype`.
- With `CLM_API_KEY` set, the server only answers requests that carry that key.

The heads only work with Qwen3-8B embeddings, so the encoder cannot be swapped for another model.

## Configuration

With `clm-serve` running on the same machine with its defaults, use `ContrastiveModel` by name, as `contrastive:clm-latest`, or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.contrastive import ContrastiveModel

model = ContrastiveModel('clm-latest')
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

To reach a server elsewhere, set its address, and the key it was started with if any, as environment variables:

```bash
export CLM_BASE_URL='http://gpu-box:8700'
export CLM_API_KEY='your-api-key'
```

These are the variables `clm-serve` and the `contrastive-lm` client read too.

### Model names

`clm-latest` is the checkpoint `clm-serve` was started with, the reference `CLM-v0.1-8B` heads unless you passed `--ckpt`. A checkpoint served with `--model NAME=PATH` is addressed by its `NAME`, as `contrastive:NAME`. `clm-raw` compares the encoder's embeddings without the heads, which is only useful as a baseline.

`clm-latest` names whatever the server was started with, so a threshold you have [tuned](decision.md#tuning-a-threshold-on-your-own-data) against one checkpoint does not carry over to another: serve each checkpoint under its own `NAME` and tune against that.

## Limits

- **No limit on options.** A CLM scores every option of a question in one pass, so a pick-one question can have as many options as you like, and the [route question](decision.md#routes-which-thing-to-do) as many tools.
- **2048 tokens per text by default.** The server cuts off each text it embeds, the state together with one question, at `--max-tokens`, which has to match vLLM's `--max-model-len`. vLLM keeps the last tokens and drops the rest without an error, so a long conversation loses its start. Set the model's [`context_window`][pydantic_ai.profiles.ModelProfile.context_window] to the same number, and a processor that [compacts when the context window fills](../message-history.md#compact-when-the-context-window-fills) keeps the history under it:

```python {test="skip"}
from pydantic_ai.models.contrastive import ContrastiveModel

model = ContrastiveModel('clm-latest', profile={'context_window': 2048})
```

## What CLMs answer badly

- **Probabilities are relative to the options.** A CLM compares the options with each other, so it will always favour one of them, even when none fits. Give a pick-one an option for "none of these" where that is a real answer, and read a yes/no's probability as how much better yes fits than no.
- **Zero-shot is not the benchmark.** The published agentic benchmark results come from heads fine-tuned for the task, not from the `CLM-v0.1-8B` checkpoint zero-shot. Fine-tuning only trains the heads, so it is cheap; see the [fine-tuning guide](https://github.com/Contrastive-LM/CLM/blob/main/docs/FINETUNING.md).
- **English.** The reference checkpoint's model card lists English as its language.

!!! note "Measure on your own data"
    Measure accuracy, the hand-off rate and any threshold on labelled examples of your own before relying on them.

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.contrastive import ContrastiveModel
from pydantic_ai.providers.contrastive import ContrastiveProvider

model = ContrastiveModel(
    'clm-latest',
    provider=ContrastiveProvider(base_url='http://gpu-box:8700', api_key='your-api-key'),
)
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

You can also customize the [`ContrastiveProvider`][pydantic_ai.providers.contrastive.ContrastiveProvider] with a custom `http_client`:

```python
from httpx2 import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.contrastive import ContrastiveModel
from pydantic_ai.providers.contrastive import ContrastiveProvider

custom_http_client = AsyncClient(timeout=30)
model = ContrastiveModel('clm-latest', provider=ContrastiveProvider(http_client=custom_http_client))
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

## Model settings

`temperature` is forwarded to the server, which divides the scores by it before turning them into probabilities: above 1 flattens every distribution, below 1 sharpens it. The server takes values above 0 and up to 100, and answers anything else with a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError], so an agent-wide `temperature` of `0` meant for a language model fails every CLM request. It moves every probability a [threshold](decision.md#confidence-and-thresholds) reads, so tune thresholds at the temperature you run with. `timeout`, `extra_headers` and `extra_body` are forwarded to the request, and the other generic settings, such as `top_p`, are ignored. [`ContrastiveModelSettings`][pydantic_ai.models.contrastive.ContrastiveModelSettings] adds the two thresholds every decision model has, `decision_boolean_threshold` and `decision_route_threshold`.

```python
from pydantic_ai import Agent
from pydantic_ai.models.contrastive import ContrastiveModel

model = ContrastiveModel('clm-latest')
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
