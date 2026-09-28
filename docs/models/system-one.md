---
description: "Run Pydantic AI agents on open-weight decision models, such as Contrastive Language Models (CLM) and Laya, served on your own hardware over the /v1/systemone API."
---

# System One servers

A [decision model](decision.md) answers typed questions about a text, each with a probability or a distribution over the options, rather than writing text: the fast, one-look "System 1" judgement, next to a language model's step-by-step "System 2" reasoning. TypeSafe's hosted Jev answers these questions over a `POST /v1/systemone` API, and open-weight decision models ship servers for the same API that you run on your own hardware:

- [Contrastive Language Models](https://github.com/Contrastive-LM/CLM) (CLM), such as [`Contrastive-LM/CLM-v0.1-8B`](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), served by `clm-serve`.
- [Laya](https://huggingface.co/convaiinnovations/laya), served by `laya-serve`.

[`SystemOneModel`][pydantic_ai.models.system_one.SystemOneModel] is the Pydantic AI model class for any server that speaks this API, and a subclass of [`DecisionModel`][pydantic_ai.models.decision.DecisionModel], like [`TypeSafeModel`](typesafe.md). An agent built for one runs on the other by changing the model. This page covers connecting to a server, serving each model, and each model's limits.

!!! tip "Start with Decision models"
    **[Decision models](decision.md) is where to learn how an agent's output types, tools and message history map
    onto a decision model's questions**, and what the answers mean.

## Install

`SystemOneModel` talks to the server over HTTP and needs nothing beyond `pydantic-ai-slim` itself. The model runs in the server, which you install separately, as [below](#serving-a-model), and which does not need to be on the machine your agent runs on.

## Configuration

Point `SystemOneModel` at the server by setting its address, and the key it was started with if any, as environment variables:

```bash
export SYSTEM_ONE_BASE_URL='http://127.0.0.1:8700'
export SYSTEM_ONE_API_KEY='your-api-key'
```

Then use it by name, as `system-one:` followed by the name the server serves the model under, such as `system-one:clm-latest`, or initialise the model directly with just that name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel

model = SystemOneModel('clm-latest')
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

The servers only speak plain HTTP, so put a server you reach over a network you do not control behind a reverse proxy that terminates HTTPS, before sending it prompts and a key.

### `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.system_one import SystemOneProvider

model = SystemOneModel(
    'clm-latest',
    provider=SystemOneProvider(base_url='https://clm.example.com', api_key='your-api-key'),
)
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

You can also customize the [`SystemOneProvider`][pydantic_ai.providers.system_one.SystemOneProvider] with a custom `http_client`:

```python
from httpx2 import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.system_one import SystemOneProvider

custom_http_client = AsyncClient(timeout=30)
model = SystemOneModel(
    'clm-latest',
    provider=SystemOneProvider(base_url='http://127.0.0.1:8700', http_client=custom_http_client),
)
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

### Model settings

`temperature` is forwarded to the server. `clm-serve` divides the scores by it before turning them into probabilities, so above 1 flattens every distribution and below 1 sharpens it, and it moves every probability a [threshold](decision.md#confidence-and-thresholds) reads. `laya-serve` ignores it. `timeout`, `extra_headers` and `extra_body` are forwarded to the request, and the other generic settings, such as `top_p`, are ignored. [`SystemOneModelSettings`][pydantic_ai.models.system_one.SystemOneModelSettings] adds the two thresholds every decision model has, `decision_boolean_threshold` and `decision_route_threshold`.

```python
from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel

model = SystemOneModel('clm-latest')
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

A server refuses a request over its own limits, such as too many options in a pick-one, with an error response, which is raised as a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError], so a [`FallbackModel`](overview.md#fallback-model) can take over.

!!! note "Measure on your own data"
    Each model's confidence is its own, and a threshold tuned on one model or checkpoint does not carry over to
    another. Measure accuracy, the hand-off rate and any threshold on labelled examples of your own before relying on them.

## Serving a model

### Contrastive Language Models (CLM)

A CLM scores each option by how well it matches the text, with two small projection heads on top of a frozen [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) encoder, released under the Apache 2.0 license. Its server, `clm-serve` from the [`contrastive-lm`](https://pypi.org/project/contrastive-lm/) package, needs a GPU machine with [vLLM](https://docs.vllm.ai) for the encoder. Start the encoder, then `clm-serve` in front of it:

```bash
pip install contrastive-lm

vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --max-model-len 2048 --port 8090 &
clm-serve
```

On first start, `clm-serve` downloads the `Contrastive-LM/CLM-v0.1-8B` heads from Hugging Face into `~/.cache/clm/` and serves them as `clm-latest` on `http://127.0.0.1:8700`, so set `SYSTEM_ONE_BASE_URL` to that and use `system-one:clm-latest`. Where the weights come from and where they run is the server's to decide:

- `--ckpt PATH` serves a local checkpoint as `clm-latest` instead, such as heads you have [fine-tuned](https://github.com/Contrastive-LM/CLM/blob/main/docs/FINETUNING.md).
- `--model NAME=PATH` serves another checkpoint beside it under `NAME`, as `system-one:NAME`, and `--ckpt-dir DIR` every `.pt` file in `DIR` under its file name.
- `--device cpu` or `--device cuda` picks where the heads run; by default they use CUDA when it is available. The encoder's device and dtype are vLLM's, set with `CUDA_VISIBLE_DEVICES` and `--dtype`.
- With `CLM_API_KEY` set, the server only answers requests that carry that key, so set `SYSTEM_ONE_API_KEY` to the same value.

The heads only work with Qwen3-8B embeddings, so the encoder cannot be swapped for another model. `clm-latest` names whatever the server was started with, so a threshold you have [tuned](decision.md#tuning-a-threshold-on-your-own-data) against one checkpoint does not carry over to another: serve each checkpoint under its own `NAME` and tune against that.

Limits and what CLMs answer badly:

- **No limit on options.** A CLM scores every option of a question in one pass, so a pick-one question can have as many options as you like, and the [route question](decision.md#routes-which-thing-to-do) as many tools.
- **2048 tokens per text by default.** The server cuts off each text it embeds, the state together with one question, at `--max-tokens`, which has to match vLLM's `--max-model-len`. vLLM keeps the last tokens and drops the rest without an error, so a long conversation loses its start. Set the model's [`context_window`][pydantic_ai.profiles.ModelProfile.context_window] to the same number, and a processor that [compacts when the context window fills](../message-history.md#compact-when-the-context-window-fills) keeps the history under it:

    ```python {test="skip"}
    from pydantic_ai.models.system_one import SystemOneModel

    model = SystemOneModel('clm-latest', profile={'context_window': 2048})
    ```

- **`temperature` must be above 0 and at most 100.** The server answers anything else with an error, so an agent-wide `temperature` of `0` meant for a language model fails every CLM request.
- **Probabilities are relative to the options.** A CLM compares the options with each other, so it will always favour one of them, even when none fits. Give a pick-one an option for "none of these" where that is a real answer, and read a yes/no's probability as how much better yes fits than no.
- **Zero-shot is not the benchmark.** The published agentic benchmark results come from heads fine-tuned for the task, not from the `CLM-v0.1-8B` checkpoint zero-shot. Fine-tuning only trains the heads, so it is cheap.
- **English.** The reference checkpoint's model card lists English as its language.

### Laya

Laya is a family of small encoder models, ModernBERT-large or mmBERT-base with a decision head, released under the Apache 2.0 license, that run on a CPU or a single GPU. Its server, `laya-serve`, comes with the [`laya`](https://pypi.org/project/laya/) package:

```bash
pip install "laya[serve]"

LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve
```

`laya-serve` downloads the checkpoints from Hugging Face and listens on `http://0.0.0.0:8000`, so set `SYSTEM_ONE_BASE_URL` to `http://127.0.0.1:8000` on the same machine. It binds every interface and answers without a key unless it was started with `LAYA_API_KEY`, which you then set as `SYSTEM_ONE_API_KEY` too.

The model name picks a checkpoint: `english` ([`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)), `multilingual` or `typed-decisions`, as `system-one:english`. Any other name, such as `system-one:laya`, lets Laya's router pick one by the language of the text.

Limits and what Laya answers badly, from its [model card](https://huggingface.co/convaiinnovations/laya) and [HTTP API docs](https://github.com/NandhaKishorM/laya/blob/main/docs/http-api.md):

- **Few options.** All the options of a question share a fixed token budget, so accuracy falls off sharply past about 20 options, and the server refuses a pick-one over 100 options or a rubric over 32 levels. Split a large pick-one into a coarse question followed by a finer one.
- **Short texts.** The English checkpoint reads 512 tokens, about 320 of them the state; the other two read 1024. Set the model's `context_window` to match, as for CLM above.
- **Over-confident as shipped.** Laya's own docs recommend refitting its temperatures on your data before trusting any threshold, and its `confidence`, reported in the response's [`provider_details`](decision.md#confidence-and-thresholds), is an entropy-based number that is not comparable with Jev's.
- **Yes/no can follow the wording.** On the English checkpoint, a yes/no answer can lean on the option labels rather than the text.
