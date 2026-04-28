# Managed Agents

Most agent configuration — the prompt, model settings, the model itself — lives in code. When you want to iterate on a prompt, run an A/B test, or give a non-engineer a way to tweak production behavior, code changes and redeploys get in the way.

Pydantic AI's [`pydantic_ai.managed`][pydantic_ai.managed] subpackage lets you externalise selected agent fields to a managed-configuration service. You declare at construction time which fields should come from the service, and the values are resolved fresh on every run.

Today, [Logfire managed variables](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/) are the supported source. Additional providers may follow.

## Managed variables with Logfire

The [`Managed`][pydantic_ai.managed.logfire.Managed] capability connects [Logfire managed variables](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/) to agent fields. Each run, the capability reads the current value of each configured variable and feeds it into the matching agent field. You change the value in the Logfire UI; the next run picks it up — no redeploy.

Two fields are currently supported:

- **[`instructions`][pydantic_ai.managed.logfire.Managed.instructions]** — a [`Variable[str]`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/) whose value is appended to the agent's static [instructions](agent.md#instructions) on every run.
- **[`model_settings`][pydantic_ai.managed.logfire.Managed.model_settings]** — a `Variable[dict]` whose value is merged on top of the agent's static [model settings](agent.md#model-run-settings).

Support for additional fields (including the model itself and per-run metadata) is planned.

### Installation

Managed variables ship in the Logfire SDK; install Pydantic AI with the `logfire` extra:

```bash
pip install "pydantic-ai-slim[logfire]"
```

Managed variables require `logfire>=4.24.0`. If you're upgrading an existing install that uses the `logfire` extra, run `pip install -U "logfire>=4.24.0"`. See the [installation guide](install.md) for the full list of extras.

### Example

```python {title="managed_agent.py" test="skip"}
import logfire

from pydantic_ai import Agent
from pydantic_ai.managed.logfire import Managed

logfire.configure()

prompt = logfire.var(
    'customer_support_prompt',
    default='You are a helpful customer-support assistant.',
)
settings = logfire.var(
    'customer_support_settings',
    type=dict,
    default={'temperature': 0.2},
)

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[Managed(instructions=prompt, model_settings=settings)],
)

result = agent.run_sync('My order is late.')
print(result.output)
```

The variables are declared once at module load. On every [`agent.run()`][pydantic_ai.agent.Agent.run] call, the capability resolves the current value from Logfire, layers it on top of any configuration already set on the agent, and proceeds with the run. If Logfire has no value for a variable, its `default` is used — making the code safe to run in tests and local development without a Logfire project configured.

### Merging with agent-level configuration

Values contributed by a `Managed` capability are **additive**:

- Managed instructions are concatenated with the agent's static instructions and any `instructions=` passed to [`agent.run()`][pydantic_ai.agent.Agent.run]; nothing is overridden.
- Managed model settings are dict-merged with the agent's static settings and any `model_settings=` passed to [`agent.run()`][pydantic_ai.agent.Agent.run]. For overlapping keys, call-site settings take precedence over managed settings, which take precedence over agent defaults.

### Layering multiple variables

To drive multiple fields from multiple variables, list several `Managed` capabilities:

```python {title="layered_managed.py" test="skip"}
import logfire

from pydantic_ai import Agent
from pydantic_ai.managed.logfire import Managed

logfire.configure()

base_prompt = logfire.var('base_prompt', default='You are helpful.')
tone_overlay = logfire.var('tone_overlay', default='Be concise.')

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        Managed(instructions=base_prompt),
        Managed(instructions=tone_overlay),
    ],
)
```

Capabilities follow middleware ordering — the first in the list is outermost. For additive fields like instructions, the order determines the concatenation order but otherwise does not matter.

### Observability

When a `Managed` capability runs, each resolved variable is entered as a context manager for the duration of the run. This sets Logfire [baggage](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/) tagging every downstream span and log with the variable's active label, so you can filter observability data by which prompt or config was live during a run.

## API reference

- [`pydantic_ai.managed`][pydantic_ai.managed]
- [`pydantic_ai.managed.logfire`][pydantic_ai.managed.logfire]
