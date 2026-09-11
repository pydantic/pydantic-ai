# Pydantic AI vs Google ADK

Google ADK is the Gemini-native kit: `LlmAgent`, a `Runner`, Vertex, Search, A2A, a web UI. If you're
on Google Cloud, that adjacency is the product.

Pydantic AI isn't tied to a cloud. The first production gap is stop: there is no `cancel`, `stop`, or
`abort` on `Runner` or `LlmAgent`.

## Side by side

| | Google ADK 2.8.0 | Pydantic AI 2.42 |
|---|---|---|
| Models | Gemini first | Any provider |
| Stop | Cancel the asyncio task | `CancellationToken` → `RunCancelled` |
| Trusted state | App / user / invocation state | `deps_type` plus `RunContext` |
| Compose | `LoopAgent`, `ParallelAgent` | `async` / `gather` |
| Deploy | Vertex | Anywhere |
| Test offline | Subclass `BaseLlm` | `TestModel` / `FunctionModel` |

## One token, three runs

```python {title="one_token_many_runs.py"}
import asyncio

from pydantic_ai import Agent, CancellationToken, RunCancelled, RunContext

agent = Agent('openai:gpt-5.6-luna')


@agent.tool
async def wait_on_stock(ctx: RunContext, sku: str) -> str:
    await asyncio.sleep(3600)
    return 'never'


async def main():
    token = CancellationToken()
    tasks = [
        asyncio.create_task(
            agent.run('Look up warehouse stock for SKU-WAIT.', cancellation_token=token)
        )
        for _ in range(3)
    ]
    await asyncio.sleep(0.1)
    token.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f'runs cancelled by one token: {sum(isinstance(r, RunCancelled) for r in results)}/3')
    #> runs cancelled by one token: 3/3
```

ADK emits OpenTelemetry GenAI conventions too. We both do. Most of this field doesn't.

## FAQ

**Gemini?** Yes, including Vertex. This isn't about the model.

**Drop-in?** No. Sessions become history. `ParallelAgent` becomes `gather`.

---

*google-adk 2.8.0, Pydantic AI 2.42. No cancel/stop/abort on `Runner` or `LlmAgent`.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
