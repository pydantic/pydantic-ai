# Logfire-backed capabilities

Drive agent configuration from [Logfire managed variables](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/),
so you can change it from the Logfire UI without redeploying.

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

`ManagedPrompt` takes an agent's instructions from a Logfire-managed
[prompt](https://logfire.pydantic.dev/docs/reference/advanced/prompt-management/). Edit, version, label, and
roll out the prompt in the Logfire UI; the next run picks it up without a redeploy.

At the start of each run, `ManagedPrompt` resolves the variable `prompt__<name>` and uses its value as the
instructions for that run. While the run is active, the selected label and version are set as baggage, so
every span in the run records which prompt version produced it.

### Usage

Pass the prompt name and a default value. The name `support_agent` is declared as the managed
variable `prompt__support_agent`, the naming Logfire's Prompt management uses (hyphens in a
name become underscores). The default is used until a remote value is published.

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
[`RunContext`](https://pydantic.dev/docs/ai/api/pydantic-ai/tools/#pydantic_ai.tools.RunContext), for example from `deps`:

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
Without `label`, the variable's rollout and targeting rules pick the label. Without `targeting_key`, Logfire
uses its [`targeting_context`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/), then
the trace id of the span enclosing the run. The prompt resolves before the agent run span opens, so with no
enclosing span the label is drawn at random per run.

### Templating with deps

By default the resolved prompt is used verbatim. Pass `render_template=True` to render it as a
Handlebars template against the agent's `deps`, the same mechanism as
[`TemplateStr`](https://pydantic.dev/docs/ai/api/pydantic-ai/template/#pydantic_ai.template.TemplateStr), so `{{field}}` is
filled from `deps`:

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

The template can reference any field of `deps`, so anyone who can edit the prompt in Logfire can place any
`deps` value in the system prompt, and from there in your traces. Keep secrets out of `deps`, or leave
rendering off.

The template is compiled once per run. If a remote prompt is not a valid template, the run uses the code
default and emits a `RuntimeWarning`. An invalid code default raises.

### Prompt-cache trade-off

The resolved value lands in the agent's **system instructions**. Provider prompt caches (Anthropic,
OpenAI, etc.) key strictly by prefix (`tools -> system -> messages`), so any change to the system
block invalidates the cached prefix for the affected runs.

| Mode | Cache impact |
| --- | --- |
| Pinned `label='production'`, no rollout split | **Cache-stable.** The value only changes on a deliberate prompt rollout, which is the same cost as a redeploy. |
| Pinned `label` that does not exist on the variable | Falls back to the rollout; same as no `label`. |
| Percentage rollout across labels (no `label=`) | Different runs land on different labels -> splits the cache into one lane per label. |
| `targeting_key` per user/tenant with multiple labels in play | Cache lanes per assigned label; deterministic per key but still N lanes overall. |
| Mid-traffic label flip in the Logfire UI | One-shot cold-invalidation for everyone on that label. |

If you don't need rollouts, pin `label='production'`.

### Using your own variable

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

agent = Agent('openai:gpt-5', capabilities=[ManagedPrompt(support_prompt, label='production')])
```

When `name` is a prompt name, pass `logfire_instance=` to declare the variable on a specific
Logfire instance instead of the module-level default. With a `Variable`, `default` and `logfire_instance`
are ignored with a warning, since the variable carries its own.

### Notes

- Resolution is isolated per run via a context variable, so a single capability instance
  is safe to share across concurrent runs.
- `ManagedPrompt.resolved` exposes the active run's `ResolvedVariable` (value, label, version,
  reason), for example from inside a tool. It is `None` outside a run.
- The capability runs outermost (wrapping `Instrumentation`), so the resolved variable's baggage
  covers the agent run span as well as its children. The label is set as the baggage entry
  `logfire.variables.prompt__<name>`; Logfire 4.34 and later also set `logfire.variables.prompt__<name>.version`.
- Resolution happens once per run. A label flip or rollout change that lands in Logfire
  mid-run is picked up when the next run starts.
- Telemetry: besides the baggage, Logfire records a `Resolve variable prompt__<name>` span per run with the
  resolved prompt text, the targeting key, and the targeting attributes. `InstrumentationSettings(include_content=False)`
  does not apply to this span.
- Resolution runs synchronously on the event loop. With Logfire's remote variables provider, the first
  resolution in a process waits for the initial fetch of the variables config; set
  `block_before_first_resolve=False` in `logfire.VariablesOptions` to resolve against the code default instead.
