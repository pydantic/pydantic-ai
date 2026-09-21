# GitHub Actions

GitHub is where an agent runs with nobody at the other end: it is started by a push, an issue, a
schedule or a button, and what it produces is a comment, a commit or a pull request rather than a
reply on a screen. Pydantic AI meets that in two shapes, and which one you want depends on who can
start the run.

| | [`pydantic/pydantic-ai/action`](#running-an-agent-in-a-step) | [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) |
|---|---|---|
| **What you write** | A step in a workflow you already have | A Markdown file that compiles to a whole workflow |
| **Where the agent runs** | On the runner, as the workflow user | In a container behind an egress firewall |
| **How it writes back** | Whatever your later steps do with the result | Safe outputs: comments, issues and pull requests gh-aw creates for it |
| **Credentials** | Reachable by the agent, from the step's `env:` | Held outside the agent's sandbox, brokered through a proxy |
| **Use it when** | You choose the trigger and the prompt | The trigger is an issue, a comment, or a pull request from a fork |

The second is the one to reach for whenever someone other than you can decide what the agent is
asked to do, and it has [a page of its own](https://pydantic.dev/docs/ai/harness/gh-aw/). The rest
of this page is the first.

## Running an agent in a step

The action installs Pydantic AI, runs one agent against one prompt, and gives you back what it
said. The provider credential goes in the step's `env:`, not in an input.

```yaml {test="skip" lint="skip"}
name: Summarize the pull request

on:
  pull_request:

permissions:
  contents: read

jobs:
  summarize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with:
          persist-credentials: false

      - id: agent
        uses: pydantic/pydantic-ai/action@main
        with:
          model: openai:gpt-5.6-sol
          prompt: Summarize what changed in this pull request.
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Use the result
        env:
          SUMMARY: ${{ steps.agent.outputs.result }}
        run: printf '%s\n' "$SUMMARY"
```

Pin the action to a commit SHA rather than a branch in anything you rely on.

The agent's output is printed to the log, written to the job summary, and exposed as the step's
`result` output. A multi-line answer survives all three.

Set exactly one of `prompt` and `prompt-file`. A prompt kept in the repository keeps long
instructions out of the workflow file, and is read relative to `working-directory`:

```yaml {test="skip" lint="skip"}
- uses: pydantic/pydantic-ai/action@main
  with:
    model: anthropic:claude-sonnet-4-5
    prompt-file: .github/prompts/review.md
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Choosing the agent

Without an `agent` input, the action runs the [Harness](https://pydantic.dev/docs/ai/harness/)'s
[`Coder`](https://pydantic.dev/docs/ai/harness/coder/), which reads and edits files and runs
commands — the same composition the gh-aw engine runs by default.

Point `agent` at a `module:variable` pair to run your own [`Agent`][pydantic_ai.Agent] instead,
and install the project that defines it with `pip-install`:

```yaml {test="skip" lint="skip"}
- uses: pydantic/pydantic-ai/action@main
  with:
    agent: my_project.agents:reviewer
    model: openai:gpt-5.6-sol
    prompt: Review the current checkout.
    pip-install: .
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

An `agent` ending in `.yml`, `.yaml` or `.json` is read as an [agent spec](agent-spec.md), so an
agent that is only instructions, tools and a model needs no Python package at all:

```yaml {test="skip" lint="skip"}
- uses: pydantic/pydantic-ai/action@main
  with:
    agent: .github/agents/reviewer.yml
    prompt: Review the current checkout.
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

That example passes no `model`, because a spec can name its own. The `model` input overrides
whatever model the agent carries, and is required only when it carries none.

The Harness is installed when `agent` names something inside it, and left out otherwise, so a
workflow running its own agent doesn't wait for it or resolve its dependencies. Set
`harness-version` to install it anyway, pinned to a release.

## Inputs

| Input | Default | Description |
|---|---|---|
| `prompt` | | The prompt to send. Set exactly one of `prompt` and `prompt-file`. |
| `prompt-file` | | A prompt file, read relative to `working-directory`. |
| `agent` | `pydantic_ai_harness.coder:coder_agent` | A `module:variable` target, or a `.yml`, `.yaml` or `.json` agent spec file. |
| `model` | | A [model](models/overview.md) identifier such as `openai:gpt-5.6-sol`. Overrides the model the agent carries. |
| `python-version` | `3.12` | The Python version the agent runs on. |
| `pip-install` | | Extra space-separated packages, such as the project your agent lives in. |
| `pydantic-ai-version` | `>=2.44.0` | The `pydantic-ai-slim` version specifier to install. |
| `harness-version` | | An exact `pydantic-ai-harness` version, installed whatever the agent is. |
| `working-directory` | `.` | The directory packages are installed from and the agent runs in. |

The action installs `pydantic-ai-slim` with the `openai`, `anthropic` and `spec` extras. Any other
provider is a `pip-install` away.

## Security

!!! warning "The agent runs on the runner, not in a sandbox"
    The agent runs as the workflow user with whatever the job can reach, and the default `Coder`
    can run commands. Anything in the job's environment — including `GITHUB_TOKEN`, if a step puts
    it there — is reachable by the model. Keep the workflow's `permissions:` at the minimum the
    job needs, don't pass `GITHUB_TOKEN` to this step, and check out with
    `persist-credentials: false` so the checkout doesn't leave a credential behind.

    Use this action when you decide what the agent is asked to do. When the prompt can come from
    an issue, a comment, or a pull request from a fork, use
    [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) instead, which runs the
    agent behind an egress firewall, keeps credentials outside its sandbox, and writes back through
    safe outputs.

Provider credentials belong in the step's `env:`. The action takes no API key input, so a key never
passes through an action input that a failure could echo. Exchanging a GitHub OIDC token for
short-lived provider credentials, the way
[Anthropic's](https://github.com/anthropics/claude-code-action) and
[OpenAI's](https://github.com/openai/codex-action) own actions can, is not implemented here yet.

## Observability

A run nobody watched needs to have left a record somewhere. Put a [Logfire](logfire.md) token in
the step's `env:`, and the action installs Logfire, configures it, and instruments the agent
before the run starts:

```yaml {test="skip" lint="skip"}
- uses: pydantic/pydantic-ai/action@main
  with:
    model: openai:gpt-5.6-sol
    prompt: Summarize what changed in this pull request.
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    LOGFIRE_TOKEN: ${{ secrets.LOGFIRE_TOKEN }}
```

`OTEL_EXPORTER_OTLP_ENDPOINT` does the same for any other OpenTelemetry backend. An agent that
configures Logfire itself keeps its own configuration: the action's happens before your agent is
imported, so whatever the agent sets up afterwards is what the run uses.
