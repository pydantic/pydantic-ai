# Pydantic AI vs Google ADK

Google's Agent Development Kit is the Gemini-native way to build agents in Python. You define an
`LlmAgent`, hand it to a `Runner` with a session service, and the rest of Google's stack is close by:
a web dev UI, first-party Google Search and MCP tools, planners, evaluation, agent-to-agent messaging,
code executors, and deployment to Vertex AI. If your organisation is on Google Cloud, that adjacency
is worth a lot.

Version 2.8.0 is a big kit. Alongside `LlmAgent` there are `LoopAgent` and `ParallelAgent` for
composing runs, and the `Runner` now carries a resumability configuration and a `rewind_async` method
for stepping a session back.

Pydantic AI is smaller and not tied to a cloud. The difference that shows up first in production is
what happens when someone hits stop.

## Stopping a run

Agents get cancelled constantly in real products. A user closes the tab. A request times out. A daily
spend cap trips and everything in flight should wind down.

ADK has no API for this. There is no `cancel`, `stop`, or `abort` on `Runner` or on `LlmAgent`, so
ending a run early means cancelling the asyncio task running it. That works, but it's abrupt: your
tools get an ordinary task cancellation partway through whatever they were doing, and what you keep
afterwards is whatever the session service already wrote.

Pydantic AI treats stopping as a result rather than an accident. One `CancellationToken` can govern
several runs at once, it's safe to call from another thread, and each run ends by raising
`RunCancelled` carrying its own history:

```python {title="one_token_many_runs.py"}
"""One stop gesture, many runs: a CancellationToken governs every run it was
given to, and cancelling the token cancels all of them.

Google ADK's runner exposes no user cancellation API (grep-verified,
2026-09-10); here the same token stops three concurrent runs at once.
"""
import asyncio

from pydantic_ai import Agent, CancellationToken, RunCancelled
from pydantic_ai.models.function import FunctionModel

CONCURRENT = 3

async def hang(messages, info):
    await asyncio.sleep(3600)  # in-flight until cancelled

async def main():
    token = CancellationToken()
    agent = Agent(FunctionModel(hang))
    tasks = [asyncio.create_task(agent.run('r', cancellation_token=token)) for _ in range(CONCURRENT)]
    await asyncio.sleep(0.1)
    token.cancel()  # one gesture
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f'runs cancelled by one token: {sum(isinstance(r, RunCancelled) for r in results)}/{CONCURRENT}')
    #> runs cancelled by one token: 3/3
    #> runs cancelled by one token: 3/3
    assert all(isinstance(r, RunCancelled) for r in results)

asyncio.run(main())


```


Three runs, one token, one gesture — and each ends in `RunCancelled`, which carries that run's
history, rather than in a bare task cancellation. A tool can
also stop its own run by calling `ctx.cancel()` — useful when the tool is the thing that discovers the
budget is gone. Cancellation that comes from outside, like `asyncio.timeout()` or a task group
shutting down, still propagates as a normal `CancelledError` so your own timeouts behave the way
Python says they should.

## The other differences

**Trusted state.** ADK carries app state, user state, and an invocation context through the run. It
works, but it's the same material the model's conversation is built from. Pydantic AI keeps
dependencies in a separate typed argument: tools read it, the model never sees it and can't name it,
so a database handle or a customer ID is not something a prompt can talk its way into.

**Testing without a network.** The installed `google.adk.models` has `BaseLlm` to subclass but no test
model, so an offline test means writing your own stub — the same situation as most frameworks, and
perfectly workable. Pydantic AI ships `TestModel` and `FunctionModel`, plus a global
`ALLOW_MODEL_REQUESTS = False` that turns any accidental real API call into an error.

**Where it runs.** ADK's natural home is Vertex AI, and its session and telemetry story assumes
Google's services. Pydantic AI runs wherever Python runs, and for crash recovery you wrap the same
agent in whichever durable engine you already operate — Temporal, DBOS, Prefect, Restate, Kitaru, or
Airflow.

## Side by side

| | Google ADK 2.8.0 | Pydantic AI 2.42 |
|---|---|---|
| Models | Gemini first; others through LiteLLM | Any provider directly, with `FallbackModel` for failover |
| Stopping a run | No cancel API; cancel the task | `CancellationToken` across runs, `ctx.cancel()` inside a tool, `RunCancelled` with the history |
| Trusted state | App state, user state, invocation context | `deps_type`, a separate argument the model never sees |
| Composing runs | `LoopAgent`, `ParallelAgent`, `ManagedAgent` | Ordinary async Python, and `pydantic_graph` when you want a state machine |
| Crash recovery | Sessions plus resumability config and `rewind_async` | Six engines wrap the agent object, and the engine is your choice |
| Testing offline | Subclass `BaseLlm` yourself | `TestModel` and `FunctionModel` included; real calls blockable globally |
| Evals | An evaluation module tied to their tooling | `pydantic-evals` in your test suite using the agent's own types |
| Tracing | Google's telemetry and Vertex | OpenTelemetry to wherever you send everything else |
| Deployment | Vertex AI is the paved road | Anywhere; it's a library |

## Choose Google ADK when

- You're on Google Cloud and want Vertex deployment, Gemini features, and Google's tools without glue.
- Agent-to-agent messaging and their planners map onto what you're building.
- The web dev UI and evaluation module save you building those.

## Choose Pydantic AI when

- A stop button needs to work properly and leave you something you can resume.
- The same agent should run on Gemini today and something else next quarter.
- Credentials must sit where the model can't reach them.
- You'd rather not have your agent framework decide where you deploy.

## FAQ

**Can I use Gemini with Pydantic AI?**
Yes, directly, including through Vertex. This isn't a comparison about which model you use.

**Is Pydantic AI a drop-in replacement?**
No. Tools and instructions port easily; sessions become message history you own, and `LoopAgent` and
`ParallelAgent` become a loop and an `asyncio.gather`.

**What does ADK do better?**
Everything downstream of being Google's. If your deployment target is Vertex and your model is Gemini,
ADK removes work we can't remove for you.

---

*Checked against google-adk 2.8.0 and Pydantic AI 2.42 on 2026-09-10. The ADK facts come from reading
the installed package: the absence of any cancel, stop, or abort method on `Runner` and `LlmAgent`,
the `Runner` method list, and the contents of `google.adk.models`. Its runtime behaviour needs a live
model and was not run. The Pydantic AI example is executed by this repository's test suite.*
