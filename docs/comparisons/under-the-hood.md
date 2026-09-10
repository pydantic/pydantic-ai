# The numbers people quote about us

Comparison articles like a number. "10,000 times faster to create an agent." "Four lines to a working
agent." "Memory that just works." Usually there's a real mechanism behind the number and a trade behind
the mechanism, and the number on its own tells you neither.

This page takes the claims that circulate about Pydantic AI, checks them where we can, and explains
what's actually being traded. We're not arguing we win every measurement. We're showing you what the
measurement is made of.

## "Creating an agent is much slower"

**The claim.** An article comparing us with Agno said Agno creates agents about 10,000 times faster,
using about 50 times less memory.

**What we measured**, on 2026-09-10, in clean environments:

| | Agno 3.0.x | Pydantic AI 2.42 |
|---|---|---|
| Time to construct one agent | 14.5 µs | 764 µs |
| Memory traced during construction | 7 KiB | 417 KiB |

So it's about 50 times, not 10,000, and only for construction. But the ratio isn't the interesting
part.

**Why ours is slower.** Creating an agent here does real work: it resolves the model provider, checks
your configuration and credentials, builds the tool schemas, and validates any prompt templates. That
happens once, at startup, where a mistake costs you a second.

```python {title="config_fails_here.py"}
"""A bad model name fails when you build the agent, not on the first request."""
from pydantic_ai import Agent

try:
    Agent('does-not-exist:gpt-4')  # a typo'd provider
except Exception as exc:
    print(f'{type(exc).__name__}: {str(exc)[:120]}')
    """
    UserError: Unknown model: does-not-exist:gpt-4. Did you mean 'openai-chat:gpt-4'?
    """
```


**The trade.** Deferring that work makes construction faster and moves the failure to the first real
request — in front of a customer, where the same typo costs money instead of a second. And the number
is the wrong one to choose on either way: construction happens once per process and takes under a
millisecond, while a single model call takes hundreds of milliseconds. If agent construction is your
bottleneck, something else has gone very right.

## "It's four lines to a working agent"

**The claim.** Minimal frameworks quote a four-line agent, and it's true — ours is about that long too.

**What the four lines don't include.** Somewhere to put credentials the model can't see. A ceiling on
what the run may spend. A stop button that leaves you something to resume. Evals. Crash recovery. In
every framework, those are things you add later or don't have.

**The trade.** Every framework makes this one. The difference is what's on the page: our
[production list](production-agents.md) is the other half of the four lines, written as code you can
run.

## "Checkpointed at every step"

**The claim.** LangGraph persists state at every step, so durability is handled.

**What happens.** It does, and that's what makes time travel and forking work — those are real
features we don't have. The cost is that a checkpoint holds a copy of the graph state, so checkpoint
size tracks the size of what you're carrying, and the durability comes with a shape: to get crash
recovery you express your control flow as a graph.

**The trade.** Ours goes the other way. The run is an ordinary coroutine, so
[durability is a capability you add](production-agents.md#9-crash-recovery-without-rewriting-the-agent)
and the engine is one your company already operates. That's less convenient if you run nothing, and
better if you run Temporal.

## "Memory that just works"

**The claim.** Some frameworks compress conversation history automatically, with no configuration.

**What happens.** Automatic compression generally means extra model calls to summarise, and those
tokens are spent whether or not they show up in the usage you're looking at. We haven't measured
anyone else's implementation, so treat that as a thing to check rather than a claim of ours.

**The trade.** Ours isn't automatic. History processors and dependencies are yours to wire, which is
more work and more visible. If memory is the centre of your product, Mastra and Agno have built more
of it than we have, and we say so on [their](vs-mastra.md) [pages](vs-agno.md).

## The trades we make, in the same format

- **Construction does real work.** Slower to build an agent, mistakes found at startup.
- **No hosted platform.** Your infrastructure stays yours, and there's no dashboard on day one.
- **No TypeScript.** If your whole product is TypeScript, read the [Vercel](vs-vercel-ai-sdk.md) and
  [Mastra](vs-mastra.md) pages instead — they're the honest answer.
- **A curated integration list, not a directory of a thousand.** You'll occasionally wire one
  yourself.
- **`run_sync` can't be nested inside async code**, and a tool running in a worker thread can't be
  force-stopped. Both are documented, and we'd rather you read that here than discover it.

## The bottom line

Every framework is a set of trades, including the ones that look light next to us. Ours are on this
page, measured where we could measure them. Read them and decide whether they're the ones you'd make.

---

*Measured on 2026-09-10 against Pydantic AI 2.42 and Agno 3.0.x in clean environments. The example on
this page is executed by this repository's test suite on every commit.*
