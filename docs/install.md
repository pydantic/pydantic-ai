# Installation

Pydantic AI is available on PyPI as [`pydantic-ai`](https://pypi.org/project/pydantic-ai/) so installation is as simple as:

```bash
pip/uv-add pydantic-ai
```

(Requires Python 3.10+)

This installs the `pydantic_ai` package, core dependencies, and libraries required to use the OpenAI, Anthropic, and Google models, plus the [CLI](cli.md), [MCP](mcp/client.md), [Evals](evals.md), [Web UI](ui/overview.md), [Retries](retries.md), and [Logfire](logfire.md) integrations.
To use any other models or integrations, add the relevant extras to your install command, e.g. `pydantic-ai[bedrock,temporal]`. Alternatively, you can install the [`pydantic-ai-slim`](#slim-install) package with only the extras you need.

## Use with Pydantic Logfire

Pydantic AI has an excellent (but completely optional) integration with [Pydantic Logfire](https://pydantic.dev/logfire) to help you view and understand agent runs.

Logfire comes included with `pydantic-ai` (but not the ["slim" version](#slim-install)), so you can typically start using it immediately by following the [Logfire setup docs](logfire.md#using-logfire).

## Running Examples

We distribute the [`pydantic_ai_examples`](https://github.com/pydantic/pydantic-ai/tree/main/examples/pydantic_ai_examples) directory as a separate PyPI package ([`pydantic-ai-examples`](https://pypi.org/project/pydantic-ai-examples/)) to make examples extremely easy to customize and run.

To install examples, use the `examples` optional group:

```bash
pip/uv-add "pydantic-ai[examples]"
```

To run the examples, follow instructions in the [examples docs](examples/setup.md).

## Slim Install

If you know which model you're going to use and want to avoid installing superfluous packages, you can use the [`pydantic-ai-slim`](https://pypi.org/project/pydantic-ai-slim/) package.
For example, if you're using just [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel], you would run:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

`pydantic-ai-slim` has the following optional groups:

* `logfire` — installs [Pydantic Logfire](logfire.md) dependency `logfire` [PyPI ↗](https://pypi.org/project/logfire){:target="_blank"}
* `evals` — installs [Pydantic Evals](evals.md) dependency `pydantic-evals` [PyPI ↗](https://pypi.org/project/pydantic-evals){:target="_blank"}
* `openai` — installs [OpenAI Model](models/openai.md) dependency `openai` [PyPI ↗](https://pypi.org/project/openai){:target="_blank"}
* `google` — installs [Google Model](models/google.md) dependency `google-genai` [PyPI ↗](https://pypi.org/project/google-genai){:target="_blank"}
* `anthropic` — installs [Anthropic Model](models/anthropic.md) dependency `anthropic` [PyPI ↗](https://pypi.org/project/anthropic){:target="_blank"}
* `groq` — installs [Groq Model](models/groq.md) dependency `groq` [PyPI ↗](https://pypi.org/project/groq){:target="_blank"}
* `mistral` — installs [Mistral Model](models/mistral.md) dependency `mistralai` [PyPI ↗](https://pypi.org/project/mistralai){:target="_blank"}
* `cohere` - installs [Cohere Model](models/cohere.md) dependency `cohere` [PyPI ↗](https://pypi.org/project/cohere){:target="_blank"}
* `bedrock` - installs [Bedrock Model](models/bedrock.md) dependency `boto3` [PyPI ↗](https://pypi.org/project/boto3){:target="_blank"}
* `xai` - installs [xAI Model](models/xai.md) dependency `xai-sdk` [PyPI ↗](https://pypi.org/project/xai-sdk){:target="_blank"}
* `openrouter` - installs the [OpenRouter](models/openrouter.md) dependency `openai` [PyPI ↗](https://pypi.org/project/openai){:target="_blank"}
* `huggingface` - installs [Hugging Face Model](models/huggingface.md) dependency `huggingface-hub` [PyPI ↗](https://pypi.org/project/huggingface-hub){:target="_blank"}
* `sentence-transformers` - installs [Sentence Transformers Embedding Model](embeddings.md#sentence-transformers-local) dependency `sentence-transformers` [PyPI ↗](https://pypi.org/project/sentence-transformers){:target="_blank"}
* `voyageai` - installs [VoyageAI Embedding Model](embeddings.md#voyageai) dependency `voyageai` [PyPI ↗](https://pypi.org/project/voyageai){:target="_blank"}
* `duckduckgo` - installs [DuckDuckGo Search Tool](common-tools.md#duckduckgo-search-tool) dependency `ddgs` [PyPI ↗](https://pypi.org/project/ddgs){:target="_blank"}
* `tavily` - installs [Tavily Search Tool](common-tools.md#tavily-search-tool) dependency `tavily-python` [PyPI ↗](https://pypi.org/project/tavily-python){:target="_blank"}
* `exa` - installs [Exa Search Tool](common-tools.md#exa-search-tool) dependency `exa-py` [PyPI ↗](https://pypi.org/project/exa-py){:target="_blank"}
* `scavio` - installs [Scavio Search Tool](common-tools.md#scavio-search-tool) dependency `scavio` [PyPI ↗](https://pypi.org/project/scavio){:target="_blank"}
* `web-fetch` - installs [Web Fetch Tool](common-tools.md#web-fetch-tool) dependency `markdownify` [PyPI ↗](https://pypi.org/project/markdownify){:target="_blank"}
* `cli` - installs [CLI](cli.md) dependencies `rich` [PyPI ↗](https://pypi.org/project/rich){:target="_blank"}, `prompt-toolkit` [PyPI ↗](https://pypi.org/project/prompt-toolkit){:target="_blank"}, and `argcomplete` [PyPI ↗](https://pypi.org/project/argcomplete){:target="_blank"}
* `mcp` - installs [MCP](mcp/client.md) dependency `fastmcp-slim[client]` [PyPI ↗](https://pypi.org/project/fastmcp-slim){:target="_blank"}
* `ui` - installs [UI Event Streams](ui/overview.md) dependency `starlette` [PyPI ↗](https://pypi.org/project/starlette){:target="_blank"}
* `web` - installs [Web UI](ui/overview.md) dependencies `starlette` [PyPI ↗](https://pypi.org/project/starlette){:target="_blank"}, `httpx` [PyPI ↗](https://pypi.org/project/httpx){:target="_blank"}, and `uvicorn` [PyPI ↗](https://pypi.org/project/uvicorn){:target="_blank"}
* `ag-ui` - installs [AG-UI Event Stream Protocol](ui/ag-ui.md) dependencies `ag-ui-protocol` [PyPI ↗](https://pypi.org/project/ag-ui-protocol){:target="_blank"} and `starlette` [PyPI ↗](https://pypi.org/project/starlette){:target="_blank"}
* `retries` - installs [HTTP Retries](retries.md) dependency `tenacity` [PyPI ↗](https://pypi.org/project/tenacity){:target="_blank"}
* `temporal` - installs [Temporal Durable Execution](durable_execution/temporal.md) dependency `temporalio` [PyPI ↗](https://pypi.org/project/temporalio){:target="_blank"}
* `dbos` - installs [DBOS Durable Execution](durable_execution/dbos.md) dependency `dbos` [PyPI ↗](https://pypi.org/project/dbos){:target="_blank"}
* `prefect` - installs [Prefect Durable Execution](durable_execution/prefect.md) dependency `prefect` [PyPI ↗](https://pypi.org/project/prefect){:target="_blank"}
* `spec` - installs [AgentSpec](agent-spec.md) dependencies `pyyaml` [PyPI ↗](https://pypi.org/project/PyYAML){:target="_blank"} and `pydantic-handlebars` [PyPI ↗](https://pypi.org/project/pydantic-handlebars){:target="_blank"}

You can also install dependencies for multiple models and use cases, for example:

```bash
pip/uv-add "pydantic-ai-slim[openai,google,logfire]"
```
