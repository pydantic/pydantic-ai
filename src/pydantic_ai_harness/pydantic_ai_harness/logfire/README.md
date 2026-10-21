# Logfire-backed capabilities

Drive agent configuration from [Logfire managed variables](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/),
so you can iterate on it from the Logfire UI -- versioned, labelled, and rolled out -- without redeploying.

Install the extra:

uv:

```bash
uv add 'pydantic-ai-harness[logfire]'
```

pip:

```bash
pip install 'pydantic-ai-harness[logfire]'
```

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/logfire/)

## `ManagedPrompt`

Back an agent's instructions with a Logfire-managed
[Prompt](https://logfire.pydantic.dev/docs/reference/advanced/prompt-management/).

### The problem

Prompts are critical to agent behavior, but iterating on them through the normal
edit -> review -> deploy loop is slow, and you can't easily A/B test a change or roll it
back the moment it misbehaves in production.

### The solution

`ManagedPrompt` declares the backing managed variable for you and resolves it **once per
run**, feeding the value into the agent's instructions. The resolution happens inside the
run's `wrap_run` hook using the
[`ResolvedVariable`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
as a context manager that stays open for the whole run -- so the selected label and version
are attached as baggage to every child span of the agent run. You get a direct correlation
between a run's behavior and the exact prompt version that produced it, plus instant
iteration and rollback from the Logfire UI.

### Usage

Pass the prompt name and a default value. The name `support_agent` is declared as the managed
variable `prompt__support_agent` -- the naming Logfire's Prompt management uses (hyphens in a
name become underscores). The default keeps the agent working until a remote value is published.

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness import ManagedPrompt

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

### Targeting

For deterministic A/B assignment (the same user always sees the same label), pass a
`targeting_key`. It can be a static string or a callable that derives the key from the
[`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext) -- handy
when the key lives in your agent's `deps`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness import ManagedPrompt


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

Pass `attributes` (or a callable returning them) for condition-based targeting rules.
When `label` is omitted, the variable's rollout and targeting rules pick the label.
When `targeting_key` is omitted, Logfire falls back to its own targeting context and then to the
trace id of the span enclosing the run. The prompt resolves before the agent run span opens, so with
no enclosing span the label is drawn at random per run.

### Templating with deps

By default the resolved prompt is used verbatim. Pass `render_template=True` to render it as a
Handlebars template against the agent's `deps` -- the same mechanism as
[`TemplateStr`](https://pydantic.dev/docs/ai/api/pydantic-ai/template/#pydantic_ai.template.TemplateStr) -- so `{{field}}` is filled
from `deps`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness import ManagedPrompt


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

The template can reference any field of `deps`, so anyone who can edit the prompt in Logfire can place any
`deps` value in the system prompt, and from there in your traces. Keep secrets out of `deps`, or leave
rendering off.

The template is compiled once per run. If a remote prompt is not a valid template, the run uses the code
default, emits a `RuntimeWarning`, and logs a warning to Logfire; the run's baggage and `resolved` then report
the code default, with the template error in `resolved.exception`. An invalid code default raises.

### Prompt-cache trade-off

The resolved value lands in the agent's **system instructions**. Provider prompt caches (Anthropic,
OpenAI, etc.) key strictly by prefix -- `tools -> system -> messages` -- so any change to the system
block invalidates the cached prefix for the affected runs.

| Mode | Cache impact |
| --- | --- |
| Pinned `label='production'`, no rollout split | **Cache-stable.** The value only changes on a deliberate prompt rollout, which is the same cost as a redeploy. |
| Pinned `label` that does not exist on the variable | Falls back to the rollout; same as no `label`. |
| Percentage rollout across labels (no `label=`) | Different runs land on different labels -> splits the cache into one lane per label. |
| `targeting_key` per user/tenant with multiple labels in play | Cache lanes per assigned label; deterministic per key but still N lanes overall. |
| Mid-traffic label flip in the Logfire UI | One-shot cold-invalidation for everyone on that label. |

In short: pinning a `label` keeps the cache hot; using `ManagedPrompt` as an A/B platform is opt-in
cache cost. If you don't need rollouts, `label='production'` is the recommended default.

### Using your own variable

Declaring the same name more than once is fine -- each `ManagedPrompt` builds its own backing
variable, so sharing a prompt across several agents just works. Pass an existing
[`logfire.variables.Variable`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
as the first argument instead of a name when you want to declare the variable yourself --
for example one registered for `variables_push`. The variable a `ManagedPrompt` builds from a name is
not registered, so `variables_push()` and `@{...}@` composition from other variables don't see it.
`TemplateVariable` is not supported; use a plain `Variable` with `render_template=True`:

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

agent = Agent('openai:gpt-5', capabilities=[ManagedPrompt(support_prompt, label='production')])
```

When `name` is a prompt name, pass `logfire_instance=` to declare the variable on a specific
Logfire instance instead of the module-level default. With a `Variable`, `default` and `logfire_instance`
are ignored with a warning, since the variable carries its own.

### Notes

- The prompt resolves to a `str`. By default it's used verbatim; set `render_template=True`
  to render `{{...}}` against `deps` (see [Templating with deps](#templating-with-deps)).
- Resolution is isolated per run via a context variable, so a single capability instance
  is safe to share across concurrent runs.
- `ManagedPrompt.resolved` exposes the active run's `ResolvedVariable` (value, label, version,
  reason) for inspection -- e.g. from inside a tool.
- The capability runs outermost (wrapping `Instrumentation`) so the resolved variable's baggage
  covers the agent run span as well as its children. The selected label is propagated as baggage;
  Logfire 4.34 and later also propagate the version as a separate baggage attribute.
- Resolution happens **once per run**. A label flip or rollout change that lands in Logfire
  mid-run is not picked up until the next run starts -- the trade-off for run-stable
  instructions and a single baggage scope across all child spans.
- For Logfire-side targeting that lives outside the agent (e.g. set once per request handler),
  use Logfire's
  [`targeting_context`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
  in an outer scope; `ManagedPrompt` only needs `targeting_key`/`attributes` when the key
  comes from the agent's `RunContext`.
- Besides the baggage, Logfire records a `Resolve variable prompt__<name>` span per run with the resolved
  prompt text, the targeting key, and the targeting attributes. `InstrumentationSettings(include_content=False)`
  does not apply to this span.
- Resolution runs synchronously on the event loop. With Logfire's remote variables provider, the first
  resolution in a process waits for the initial fetch of the variables config; set
  `block_before_first_resolve=False` in `logfire.VariablesOptions` to resolve against the code default instead.
