---
title: Managed Prompt
description: Back a Pydantic AI agent's instructions with a Logfire-managed prompt so you can version, label, and roll it out without redeploying.
---

# Managed Prompt

`ManagedPrompt` takes an agent's instructions from a
[Logfire-managed prompt](https://logfire.pydantic.dev/docs/reference/advanced/prompt-management/). Edit,
version, label, and roll out the prompt in the Logfire UI; the next run picks it up without a redeploy. It's a
Pydantic AI [capability](index.md), so you wire it in through the `capabilities=` parameter on `Agent`.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/logfire/)

Install the `logfire` extra:

```bash
pip/uv-add "pydantic-ai-harness[logfire]"
```

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem it solves

Changing a prompt through the edit -> review -> deploy loop is slow, and an A/B test or a rollback needs a
new build. `ManagedPrompt` keeps the prompt in Logfire's managed-variable store instead, where you can roll
a change out to a share of traffic and roll it back from the UI.

## How it works

At the start of each run, `ManagedPrompt` resolves the variable `prompt__<name>` in the run's
[`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run] hook and uses its value as the instructions
for that run. The [`ResolvedVariable`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
stays open as a context manager for the whole run, so the selected label (and the version, on Logfire 4.34 and
later) is set as baggage, and every span in the run records which prompt produced it.

## Usage

Pass the prompt name and a default value. The name `support_agent` is declared as the managed
variable `prompt__support_agent`, the naming Logfire's Prompt management uses (hyphens in a name
become underscores). The `default` is used until a remote value is published.

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness import ManagedPrompt

logfire.configure()

agent = Agent(
    '{{ openai:medium }}',
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

Pinning `label='production'` keeps the resolved value fixed until a deliberate prompt rollout, which keeps the
provider prompt cache warm (see [Prompt-cache trade-off](#prompt-cache-trade-off) below).

## Targeting

For deterministic A/B assignment (the same user always sees the same label), pass a
`targeting_key`. It can be a static string or a callable that derives the key from the
[`RunContext`][pydantic_ai.tools.RunContext], for example from `deps`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness import ManagedPrompt


@dataclass
class Deps:
    user_id: str


agent = Agent(
    '{{ openai:medium }}',
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
Without `label`, the variable's rollout and targeting rules pick the label. Without `targeting_key`, Logfire
uses its [`targeting_context`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/), then the
trace id of the span enclosing the run. The prompt resolves before the agent run span opens, so with no
enclosing span the label is drawn at random per run.

## Templating with deps

By default the resolved prompt is used verbatim. Pass `render_template=True` to render it as a
Handlebars template against the agent's `deps`, the same mechanism as
[`TemplateStr`][pydantic_ai.template.TemplateStr], so `{{field}}` is filled from `deps`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness import ManagedPrompt


@dataclass
class Deps:
    customer_name: str


agent = Agent(
    '{{ openai:medium }}',
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

The template can reference any field of `deps`, so anyone who can edit the prompt in Logfire can place any
`deps` value in the system prompt, and from there in your traces. Keep secrets out of `deps`, or leave
rendering off.

The template is compiled once per run. If a remote prompt is not a valid template, the run uses the code
default, emits a `RuntimeWarning`, and logs a warning to Logfire; the run's baggage and `resolved` then report
the code default, with the template error in `resolved.exception`. An invalid code default raises.

## Prompt-cache trade-off

The resolved value lands in the agent's **system instructions**. Provider prompt caches (Anthropic,
OpenAI, etc.) key strictly by prefix (`tools -> system -> messages`), so each distinct value has
its own cached prefix.

Within one conversation (runs that continue from `message_history`), a new value doesn't rewrite
that prefix. The block keeps the value the conversation started with, and the new version is
[appended](../agent.md#appending-instruction-changes) to the history as a
replacement, so the cached conversation behind it stays reusable. That's also why the prompt is
sent inside a `<context id="capability:managed-prompt-support_agent">` element from the first
request. Appended changes reach the model as system messages, so keep the prompt, and the `deps`
fields `render_template=True` fills in, to text you control: see
[keep untrusted text out of instructions](../agent.md#untrusted-instruction-text).

Across conversations, the value a new conversation starts with decides its prefix:

| Mode | Cache impact |
| --- | --- |
| Pinned `label='production'`, no rollout split | **Cache-stable.** The value only changes on a deliberate prompt rollout, which is the same cost as a redeploy. |
| Pinned `label` that does not exist on the variable | Falls back to the rollout; same as no `label`. |
| Percentage rollout across labels (no `label=`) | Different runs land on different labels -> splits the cache into one lane per label. |
| `targeting_key` per user/tenant with multiple labels in play | Cache lanes per assigned label; deterministic per key but still N lanes overall. |
| Mid-traffic label flip in the Logfire UI | New conversations on that label start on a cold prefix; conversations already under way append the new value. |

If you don't need rollouts, pin `label='production'`.

## Bringing your own variable

Each `ManagedPrompt` builds its own unregistered variable, so several agents can use the same prompt name.
That variable is invisible to `variables_push()` and to `@{...}@` composition from other variables. To get
those, declare it with `logfire.var(...)` and pass it as the first argument. `TemplateVariable` is not
supported; use a plain `Variable` with `render_template=True`.

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness import ManagedPrompt

logfire.configure()

support_prompt = logfire.var(
    name='prompt__support_agent',
    type=str,
    default='You are a helpful customer support agent. Be friendly and concise.',
)

agent = Agent('{{ openai:medium }}', capabilities=[ManagedPrompt(support_prompt, label='production')])
```

When `name` is a prompt name (not a `Variable`), pass `logfire_instance=` to declare the variable
on a specific Logfire instance instead of the module-level default. `default` is required when
`name` is a prompt name. With a `Variable`, `default` and `logfire_instance` are ignored with a warning,
since the variable carries its own.

The capability's `id` is derived from the prompt: `ManagedPrompt('support_agent', ...)` and a
`Variable` named `prompt__support_agent` both get `managed-prompt-support_agent`. It names the
prompt's instruction block, so a new version can be appended to a conversation rather than
rewriting it. Two `ManagedPrompt`s with different names on one agent each get their own block, and
two with the same name raise a `UserError` when the agent is built, since they're one prompt stated
twice. Pass `id=` to override the derived one. Earlier releases left the `id` unset and accepted
two of one name, so an agent built that way now needs a distinct `id` on each.

## How it composes

- **Resolves once per run.** A label flip or rollout change that lands in Logfire mid-run is picked up when the
  next run starts.
- **Runs outermost.** The capability wraps
  [`Instrumentation`][pydantic_ai.capabilities.Instrumentation]
  so the resolved variable's baggage covers the agent run span as well as its children. The label is set as the
  baggage entry `logfire.variables.prompt__<name>`; Logfire 4.34 and later also set
  `logfire.variables.prompt__<name>.version`.
- **Concurrency-safe.** Resolution is isolated per run via a context variable, so a single
  capability instance is safe to share across concurrent runs.
- **Inspectable mid-run.** `ManagedPrompt.resolved` exposes the active run's `ResolvedVariable`
  (`value`, `label`, `version`, `reason`), for example from inside a tool. It is `None` outside a run.
- **Telemetry.** Besides the baggage, Logfire records a `Resolve variable prompt__<name>` span per run with the
  resolved prompt text, the targeting key, and the targeting attributes.
  `InstrumentationSettings(include_content=False)` does not apply to this span.
- **Resolves on the event loop.** Resolution is synchronous. With Logfire's remote variables provider, the
  first resolution in a process waits for the initial fetch of the variables config; set
  `block_before_first_resolve=False` in `logfire.VariablesOptions` to resolve against the code default instead.

## API reference

The resolved prompt is a `str`. Pass the bare prompt name (the `prompt__` prefix and
hyphen-to-underscore normalization are applied for you) and a `default`, then use `label`,
`targeting_key`, `attributes`, `render_template`, and `logfire_instance` to control resolution.

::: pydantic_ai_harness.ManagedPrompt
