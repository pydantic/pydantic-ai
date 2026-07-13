---
title: Managed Prompt
description: Back a Pydantic AI agent's instructions with a Logfire-managed prompt so you can version, label, and roll it out without redeploying.
---

# Managed Prompt

`ManagedPrompt` backs an agent's instructions with a
[Logfire-managed prompt](https://logfire.pydantic.dev/docs/reference/advanced/prompt-management/),
so you can iterate on your system prompt from the Logfire UI -- versioned, labelled, and rolled
out -- without touching code or redeploying. It's a Pydantic AI [capability](index.md), so you
wire it in through the `capabilities=` parameter on `Agent`.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire/)

Install the `logfire` extra:

```bash
uv add "pydantic-ai-harness[logfire]"
```

!!! note "A first-party `Managed` capability is in flight"
    A broader, first-party `Managed` capability is being built in
    [pydantic-ai#5107](https://github.com/pydantic/pydantic-ai/pull/5107) and will eventually be
    importable as `pydantic_ai.managed.logfire.Managed` -- covering instructions, model settings,
    and whole-spec variables. Until then, `ManagedPrompt` is the supported path for backing an
    agent's instructions with a Logfire-managed prompt.

## The problem it solves

Prompts are critical to agent behavior, but iterating on them through the normal
edit -> review -> deploy loop is slow. You can't easily A/B test a change, and you can't roll it
back the moment it misbehaves in production without shipping a new build.

`ManagedPrompt` moves the prompt out of your codebase and into Logfire's managed-variable store.
It declares the backing managed variable for you and resolves it **once per run**, feeding the
resolved value into the agent's instructions. Resolution happens inside the run's
[`wrap_run`](/ai/api/pydantic-ai/capabilities/#pydantic_ai.capabilities.AbstractCapability.wrap_run)
hook, using the
[`ResolvedVariable`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/) as a
context manager that stays open for the whole run -- so the selected label and version are attached
as baggage to every child span of the agent run. You get a direct correlation between a run's
behavior and the exact prompt version that produced it, plus instant iteration and rollback from
the Logfire UI.

## Usage

Pass the prompt name and a default value. The name `support_agent` is declared as the managed
variable `prompt__support_agent` -- the naming Logfire's Prompt management uses (hyphens in a name
become underscores). The `default` keeps the agent working until a remote value is published, so
your code always runs even before you create the prompt in Logfire.

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedPrompt

logfire.configure()

agent = Agent(
    'openai:gpt-5',
    capabilities=[
        ManagedPrompt(
            'support_agent',
            default='You are a helpful customer support agent. Be friendly and concise.',
            label='production',
        )
    ],
)

result = agent.run_sync('My order never arrived.')
print(result.output)
```

Pinning `label='production'` is the recommended default: the resolved value only changes on a
deliberate prompt rollout, which keeps the provider prompt cache hot (see
[Prompt-cache trade-off](#prompt-cache-trade-off) below).

## Targeting

For deterministic A/B assignment (the same user always sees the same label), pass a
`targeting_key`. It can be a static string or a callable that derives the key from the
[`RunContext`](/ai/api/pydantic-ai/tools/#pydantic_ai.tools.RunContext) -- handy when the
key lives in your agent's `deps`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedPrompt


@dataclass
class Deps:
    user_id: str


agent = Agent(
    'openai:gpt-5',
    deps_type=Deps,
    capabilities=[
        ManagedPrompt(
            'support_agent',
            default='You are a helpful customer support agent.',
            targeting_key=lambda ctx: ctx.deps.user_id,
        ),
    ],
)
```

Pass `attributes` (a mapping, or a callable returning one) for condition-based targeting rules.
When `label` is omitted, the variable's rollout and targeting rules pick the label. When both
`targeting_key` and `attributes` are omitted, Logfire falls back to its own targeting context and
then to the active trace id.

For Logfire-side targeting that lives outside the agent (e.g. set once per request handler), use
Logfire's
[`targeting_context`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/) in
an outer scope; `ManagedPrompt` only needs `targeting_key` / `attributes` when the key comes from
the agent's `RunContext`.

## Templating with deps

By default the resolved prompt is used verbatim. Pass `render_template=True` to render it as a
Handlebars template against the agent's `deps` -- the same mechanism as
[`TemplateStr`](/ai/api/pydantic-ai/agent/) -- so `{{field}}` is filled
from `deps`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedPrompt


@dataclass
class Deps:
    customer_name: str


agent = Agent(
    'openai:gpt-5',
    deps_type=Deps,
    capabilities=[
        ManagedPrompt(
            'support_agent',
            default='You are helping {{customer_name}}. Be friendly and concise.',
            render_template=True,
        ),
    ],
)
```

Rendering requires `pydantic-handlebars` (install `pydantic-ai-slim[spec]`). It is off by default.

## Prompt-cache trade-off

The resolved value lands in the agent's **system instructions**. Provider prompt caches (Anthropic,
OpenAI, etc.) key strictly by prefix -- `tools -> system -> messages` -- so any change to the system
block invalidates the cached prefix for the affected runs.

| Mode | Cache impact |
| --- | --- |
| Pinned `label='production'`, no rollout split | **Cache-stable.** The value only changes on a deliberate prompt rollout, which is the same cost as a redeploy. |
| Percentage rollout across labels (no `label=`) | Different runs land on different labels -> splits the cache into one lane per label. |
| `targeting_key` per user/tenant with multiple labels in play | Cache lanes per assigned label; deterministic per key but still N lanes overall. |
| Mid-traffic label flip in the Logfire UI | One-shot cold-invalidation for everyone on that label. |

In short: pinning a `label` keeps the cache hot; using `ManagedPrompt` as an A/B platform is opt-in
cache cost. If you don't need rollouts, `label='production'` is the recommended default.

## Bringing your own variable

Declaring the same name more than once is fine -- each `ManagedPrompt` builds its own backing
variable, so sharing a prompt across several agents just works. Pass an existing
[`logfire.variables.Variable`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
as the first argument instead of a name when you want to declare the variable yourself -- for
example a template variable, or one registered for `variables_push`:

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedPrompt

logfire.configure()

support_prompt = logfire.var(
    name='prompt__support_agent',
    type=str,
    default='You are a helpful customer support agent. Be friendly and concise.',
)

agent = Agent('openai:gpt-5', capabilities=[ManagedPrompt(support_prompt, label='production')])
```

When `name` is a prompt name (not a `Variable`), pass `logfire_instance=` to declare the variable
on a specific Logfire instance instead of the module-level default. `default` is required when
`name` is a prompt name and is ignored when you pass a `Variable` (which already carries its own
default and instance).

## How it composes

- **Resolves once per run.** A label flip or rollout change that lands in Logfire mid-run is not
  picked up until the next run starts -- the trade-off for run-stable instructions and a single
  baggage scope across all child spans.
- **Runs outermost.** The capability wraps
  [`Instrumentation`](/ai/api/pydantic-ai/capabilities/#pydantic_ai.capabilities.Instrumentation)
  so the resolved variable's baggage covers the agent run span as well as its children. On recent
  Logfire versions both the selected label and the version are propagated as separate baggage
  attributes.
- **Concurrency-safe.** Resolution is isolated per run via a context variable, so a single
  capability instance is safe to share across concurrent runs.
- **Inspectable mid-run.** `ManagedPrompt.resolved` exposes the active run's `ResolvedVariable`
  (`value`, `label`, `version`, `reason`) for inspection -- e.g. from inside a tool. It is `None`
  outside a run.

## API reference

The resolved prompt is a `str`. Pass the bare prompt name (the `prompt__` prefix and
hyphen-to-underscore normalization are applied for you) and a `default`, then use `label`,
`targeting_key`, `attributes`, `render_template`, and `logfire_instance` to control resolution.

::: pydantic_ai_harness.ManagedPrompt
