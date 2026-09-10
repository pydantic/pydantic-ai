# Tradeoffs, translated

When a comparison page lands on us with a number — "10,000× faster agent creation", "four lines
to a working agent", "memory that just works" — there's usually a mechanism behind the difference,
and it's almost always a tradeoff. This page takes the claims that circulate about Pydantic AI,
explains what actually happens, and translates what it means for you. We're not here to argue we're
faster at everything: we're here to show you the switch behind the difference, so you can flip it
for yourself.

## "Pydantic AI is heavier when you create an agent"

**The claim:** an article comparing us and Agno said Agno creates agents "~10,000× faster" with
"~50× lower memory".

**What actually happens (we measured it, 2026-09-10, clean environments):**

| | Agno 3.0.x | Pydantic AI 2.42.0 |
|---|---|---|
| time to construct one agent | 14.5 µs | 764 µs (~50×) |
| traced memory during construction | 7 KiB | 417 KiB (~50-60×) |

So the "10,000×" doesn't reproduce — it's about **50×**, and only for construction. But don't stop
at the correction; the interesting part is *why*.

**The mechanism:** creating an agent here does real work. It resolves the model provider, checks
your configuration and credentials, builds the tool schemas, validates spec templates. That work is
done **once, up front, where a mistake costs you a second**.

**The tradeoff:** theirs is faster to construct because that work is deferred. Ours is slower
because it's done where mistakes are cheap. "Why would you want to do those things later?" — you
wouldn't. The deferred version doesn't save you the work; it moves it to the worst possible moment:
runtime, in front of a customer, where the same mistake costs money and trust.

```python {title="config_fails_here.py"}
"""Config errors fail at construction, not at 3 a.m. in production."""
from pydantic_ai import Agent

try:
    Agent('does-not-exist:gpt-4')  # a typo'd provider
except Exception as exc:
    print(f'{type(exc).__name__}: {str(exc)[:120]}')
```

```text
UserError: Unknown model: does-not-exist:gpt-4. Did you mean 'openai-chat:gpt-4'?
```

**What it means for you:** 0.75 ms, paid once, at startup — versus the same mistake showing up
later, when it's expensive. That's the whole tradeoff, and it's why we think our side is the right
one for software you ship.

## "It's only four lines to a working agent"

**The claim:** minimal frameworks (smolagents is "one step above plain LLM calls"; the OpenAI SDK
quotes a four-line agent).

**What the four lines don't include:** a deps boundary, budgets, cancellation, evals, durability.
Those are either absent or something you add yourself, later.

**The tradeoff:** we print the seams; a demo doesn't need them and a shipped agent does. Every
framework makes this tradeoff; ours just puts the missing half on the page as running code —
[the production checklist](production-agents.md) is the four lines' other half.

## "Stateful graph, checkpointed at every step"

**The claim:** LangGraph persists state at every step, and their platform is the durable story.

**What happens:** checkpoints are a deliberate architecture — and an independent review (Speakeasy,
2026-03) notes the costs: no token-budget management (context bloat silently degrades long runs),
and a full copy of the state per step, so checkpoints grow with your payloads.

**The tradeoff:** their durability comes with a shape you must adopt. Ours wraps the same loop you
already drive — [the run stays plain, the engines attach](production-agents.md#9-durability-is-attached-at-run-time-not-written-into-the-agent).
The difference is whose architecture the agent lives in.

## "Memory that just works"

**The claim:** Mastra's Observational Memory auto-compresses conversations ~5-40× with no
configuration.

**What happens:** the compression runs **background LLM calls** whose tokens don't appear in your
agent's usage — automatic, and billed off the books (same independent review).

**The tradeoff:** automatic is not free; it's hidden. Ours is deps and history processors, wired by
you, billed visibly. You choose what automatic means for your budget.

## The tradeoffs *we* make, stated the same way

- **Construction does real work up front** — see above. We believe fail-fast beats fail-later.
- **No managed agent server.** Your infra stays your infra; that's a decision, not an omission.
- **No TS/JS framework.** If your whole product is TypeScript, the honest reads are the
  [Vercel](vs-vercel-ai-sdk.md) and [Mastra](vs-mastra.md) pages.
- **Curated integrations, not a 1,000-item directory.** You wire the rare one; the seams make that
  a weekend, not a project.
- **`run_sync` can't nest inside async code**, and a worker-thread tool can't be force-stopped.
  We say it so you read it before you build around it.

## The bottom line

Are we giving you an inefficient piece of software? No. We're making tradeoffs — like all software,
including the frameworks that look light by comparison. The difference is that ours are on this
page, measured where we could measure them, and argued in the open. We believe they're the right
tradeoffs for a framework you ship with; read the arguments and decide if you agree.

*Versions: Agno 3.0.x, Pydantic AI 2.42.0 — measured 2026-09-10 in clean virtualenvs (20,000
iterations each, `time.perf_counter` + `tracemalloc`). Snippets re-executed by this repository's
tests.*