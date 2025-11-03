# Installation

Pydantic AI is available on PyPI as [`pydantic-ai`](https://pypi.org/project/pydantic-ai/) so installation is as simple as:

```bash
pip/uv-add pydantic-ai
```

(Requires Python 3.10+)

This installs the `pydantic_ai` package, core dependencies, and libraries required to use all the models included in Pydantic AI.
If you want to install only those dependencies required to use a specific model, you can install the ["slim"](#slim-install) version of Pydantic AI.

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
* `vertexai` — installs `GoogleVertexProvider` dependencies `google-auth` [PyPI ↗](https://pypi.org/project/google-auth){:target="_blank"} and `requests` [PyPI ↗](https://pypi.org/project/requests){:target="_blank"}
* `google` — installs [Google Model](models/google.md) dependency `google-genai` [PyPI ↗](https://pypi.org/project/google-genai){:target="_blank"}
* `anthropic` — installs [Anthropic Model](models/anthropic.md) dependency `anthropic` [PyPI ↗](https://pypi.org/project/anthropic){:target="_blank"}
* `groq` — installs [Groq Model](models/groq.md) dependency `groq` [PyPI ↗](https://pypi.org/project/groq){:target="_blank"}
* `mistral` — installs [Mistral Model](models/mistral.md) dependency `mistralai` [PyPI ↗](https://pypi.org/project/mistralai){:target="_blank"}
* `cohere` - installs [Cohere Model](models/cohere.md) dependency `cohere` [PyPI ↗](https://pypi.org/project/cohere){:target="_blank"}
* `bedrock` - installs [Bedrock Model](models/bedrock.md) dependency `boto3` [PyPI ↗](https://pypi.org/project/boto3){:target="_blank"}
* `huggingface` - installs [Hugging Face Model](models/huggingface.md) dependency `huggingface-hub[inference]` [PyPI ↗](https://pypi.org/project/huggingface-hub){:target="_blank"}
* `outlines-transformers` - installs [Outlines Model](models/outlines.md) dependency `outlines[transformers]` [PyPI ↗](https://pypi.org/project/outlines){:target="_blank"}
* `outlines-llamacpp` - installs [Outlines Model](models/outlines.md) dependency `outlines[llamacpp]` [PyPI ↗](https://pypi.org/project/outlines){:target="_blank"}
* `outlines-mlxlm` - installs [Outlines Model](models/outlines.md) dependency `outlines[mlxlm]` [PyPI ↗](https://pypi.org/project/outlines){:target="_blank"}
* `outlines-sglang` - installs [Outlines Model](models/outlines.md) dependency `outlines[sglang]` [PyPI ↗](https://pypi.org/project/outlines){:target="_blank"}
* `outlines-vllm-offline` - installs [Outlines Model](models/outlines.md) dependencies `outlines` [PyPI ↗](https://pypi.org/project/outlines){:target="_blank"} and `vllm` [PyPI ↗](https://pypi.org/project/vllm){:target="_blank"}
* `duckduckgo` - installs [DuckDuckGo Search Tool](common-tools.md#duckduckgo-search-tool) dependency `ddgs` [PyPI ↗](https://pypi.org/project/ddgs){:target="_blank"}
* `tavily` - installs [Tavily Search Tool](common-tools.md#tavily-search-tool) dependency `tavily-python` [PyPI ↗](https://pypi.org/project/tavily-python){:target="_blank"}
* `cli` - installs [CLI](cli.md) dependencies `rich` [PyPI ↗](https://pypi.org/project/rich){:target="_blank"}, `prompt-toolkit` [PyPI ↗](https://pypi.org/project/prompt-toolkit){:target="_blank"}, and `argcomplete` [PyPI ↗](https://pypi.org/project/argcomplete){:target="_blank"}
* `mcp` - installs [MCP](mcp/client.md) dependency `mcp` [PyPI ↗](https://pypi.org/project/mcp){:target="_blank"}
* `fastmcp` - installs [FastMCP](mcp/fastmcp-client.md) dependency `fastmcp` [PyPI ↗](https://pypi.org/project/fastmcp){:target="_blank"}
* `a2a` - installs [A2A](a2a.md) dependency `fasta2a` [PyPI ↗](https://pypi.org/project/fasta2a){:target="_blank"}
* `ui` - installs [UI Event Streams](ui/overview.md) dependency `starlette` [PyPI ↗](https://pypi.org/project/starlette){:target="_blank"}
* `ag-ui` - installs [AG-UI Event Stream Protocol](ui/ag-ui.md) dependencies `ag-ui-protocol` [PyPI ↗](https://pypi.org/project/ag-ui-protocol){:target="_blank"} and `starlette` [PyPI ↗](https://pypi.org/project/starlette){:target="_blank"}
* `dbos` - installs [DBOS Durable Execution](durable_execution/dbos.md) dependency `dbos` [PyPI ↗](https://pypi.org/project/dbos){:target="_blank"}
* `prefect` - installs [Prefect Durable Execution](durable_execution/prefect.md) dependency `prefect` [PyPI ↗](https://pypi.org/project/prefect){:target="_blank"}

You can also install dependencies for multiple models and use cases, for example:

```bash
pip/uv-add "pydantic-ai-slim[openai,google,logfire]"
```
