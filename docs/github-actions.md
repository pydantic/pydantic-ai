---
description: "Run a Pydantic AI agent as a step in a GitHub Actions workflow with pydantic/pydantic-ai-action, or on issues and pull requests through GitHub Agentic Workflows."
---

# GitHub Actions

GitHub is where an agent runs with nobody at the other end: it is started by a push, an issue, a
schedule or a button, and what it produces is a comment, a commit or a pull request rather than a
reply on a screen. Pydantic AI meets that in two shapes, and which one you want depends on who can
start the run.

| | [`pydantic/pydantic-ai-action`](#running-an-agent-in-a-step) | [GitHub Agentic Workflows](harness/gh-aw.md) |
|---|---|---|
| **What you write** | A step in a workflow you already have | A Markdown file that compiles to a whole workflow |
| **Where the agent runs** | On the runner, as the workflow user | In a container behind an egress firewall |
| **How it writes back** | Whatever your later steps do with the result | Safe outputs: comments, issues and pull requests gh-aw creates for it |
| **Credentials** | Reachable by the agent, from the step's `env:` | Held outside the agent's sandbox, brokered through a proxy |
| **Use it when** | You trust the prompt, the checkout and the agent | Any of them can come from an issue, a comment, or a pull request from a fork |

The second is the one to reach for whenever someone other than you can decide what the agent is
asked to do or what it reads, and it has [a page of its own](harness/gh-aw.md). The rest
of this page is the first.

## Running an agent in a step

The [`pydantic/pydantic-ai-action`](https://github.com/pydantic/pydantic-ai-action) action
installs Pydantic AI, runs one agent against one prompt, and gives you back what it said. The
provider credential goes in the step's `env:`, not in an input.

```yaml {test="skip" lint="skip"}
name: Summarize the pull request

on:
  pull_request:

permissions:
  contents: read

jobs:
  summarize:
    # A pull request from a fork gets no secrets, so the agent would have nothing to authenticate
    # with, and its prompt and checkout would both be attacker-controlled. That case belongs in
    # gh-aw, not here.
    if: github.event.pull_request.head.repo.full_name == github.repository
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with:
          persist-credentials: false

      - id: agent
        uses: pydantic/pydantic-ai-action@v1
        with:
          model: {{ openai:medium }}
          prompt: Summarize what changed in this pull request.
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Use the result
        env:
          SUMMARY: ${{ steps.agent.outputs.result }}
        run: printf '%s\n' "$SUMMARY"
```

`@v1` moves with each `v1.x.y` release of
[`pydantic/pydantic-ai-action`](https://github.com/pydantic/pydantic-ai-action); pin a full commit
SHA instead in anything you rely on.

The agent's output is printed to the log, written to the job summary, and exposed as the step's
`result` output. A multi-line answer survives all three, and the runner reads none of it as
instructions: workflow commands are stopped for the whole run, so an `::add-mask::` or
`::stop-commands::` line in the agent's answer is printed rather than obeyed, and the `result` output is
delimited with a random token so an answer cannot forge a second output.

That covers the action publishing the answer, not what you do with it next: `result` is text the
model wrote. Pass it to a later step through `env:`, as the example does, or a file, never as
`${{ steps.agent.outputs.result }}` inside a `run:` script, where it would run as shell.

Set exactly one of `prompt` and `prompt-file`. A prompt kept in the repository keeps long
instructions out of the workflow file, and is read relative to `working-directory`:

```yaml {test="skip" lint="skip"}
- uses: pydantic/pydantic-ai-action@v1
  with:
    model: {{ anthropic:medium }}
    prompt-file: .github/prompts/review.md
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Choosing the agent

Without an `agent` input, the action runs the [Harness](harness/index.md)'s
[`Coder`](harness/coder.md), which reads and edits files and runs
commands. It is the exported `pydantic_ai_harness.coder:coder_agent`, whose commands get only `PATH`,
`HOME` and the locale variables, not the step's environment.

Point `agent` at a `module:variable` pair to run your own [`Agent`][pydantic_ai.Agent] instead,
and install the project that defines it with `pip-install`:

```yaml {test="skip" lint="skip"}
- uses: pydantic/pydantic-ai-action@v1
  with:
    agent: my_project.agents:reviewer
    model: {{ openai:medium }}
    prompt: Review the current checkout.
    pip-install: .
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

An `agent` ending in `.yml`, `.yaml` or `.json` is read as an [agent spec](agent-spec.md), so an
agent that is only instructions, tools and a model needs no Python package at all:

```yaml {test="skip" lint="skip"}
- uses: pydantic/pydantic-ai-action@v1
  with:
    agent: .github/agents/reviewer.yml
    prompt: Review the current checkout.
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

That example passes no `model`, because a spec can name its own. The `model` input overrides
whatever model the agent carries, and is required only when it carries none.

The Harness is installed when `agent` names something inside it, and left out otherwise, so a
workflow running its own agent doesn't wait for it or resolve its dependencies. An agent of your own
that uses the Harness installs it through `pip-install`.

Both packages are pinned to the exact versions each release of the action was tested with, so a
workflow that pins the action to a commit runs the same agent code every time. Set
`pydantic-ai-version` to pick another release; with the default agent, set `harness-version` to the
matching Harness release too (`0.67.0` goes with `2.67.0`).

## Inputs

| Input | Default | Description |
|---|---|---|
| `prompt` | | The prompt to send. Set exactly one of `prompt` and `prompt-file`. |
| `prompt-file` | | A prompt file, read relative to `working-directory`. |
| `agent` | `pydantic_ai_harness.coder:coder_agent` | A `module:variable` target, or a `.yml`, `.yaml` or `.json` agent spec file. |
| `model` | | A [model](models/overview.md) identifier such as `{{ openai:medium }}`. Overrides the model the agent carries. |
| `python-version` | `3.12` | The Python the agent runs on, 3.11 or later. |
| `pip-install` | | Extra space-separated packages, such as the project your agent lives in. |
| `pydantic-ai-version` | `2.67.0` | The exact `pydantic-ai-slim` version to install. |
| `harness-version` | `0.67.0` | The exact `pydantic-ai-harness` version, installed when `agent` names the Harness. |
| `working-directory` | `.` | The directory packages are installed from and the agent runs in. |

The action installs `pydantic-ai-slim` with the `openai`, `anthropic` and `spec` extras. Any other
provider is a `pip-install` away.

## Security

!!! warning "The agent runs on the runner, not in a sandbox"
    The agent runs as the workflow user with whatever the job can reach, and the default `Coder`
    can run commands. Anything in the job's environment — including `GITHUB_TOKEN`, if a step puts
    it there — is reachable by the model.

    Use this action only when everything the agent is given is trusted: the prompt, the checkout,
    the agent or spec it runs, and the packages `pip-install` installs, which run their own build
    code with the provider credential already in the environment. A branch counts as untrusted if
    anyone you wouldn't hand the credential to can push to it. When any of these can come from
    someone else, use [GitHub Agentic Workflows](harness/gh-aw.md) instead, which runs the agent
    behind an egress firewall, keeps credentials outside its sandbox, and writes back through safe
    outputs.

In a workflow that runs the action, keep `permissions:` at the minimum the job needs, don't pass
`GITHUB_TOKEN` to the step, and check out with `persist-credentials: false` so the checkout
doesn't leave a credential behind. Anything that needs write access or a deploy credential belongs
in a separate job that runs after the agent's and reads `result` from its output. To have a person
approve each run, put the agent's job in an
[environment](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)
with required reviewers.

The action starts the agent's process without the runner's command files and tokens —
`GITHUB_ENV`, `GITHUB_PATH`, `GITHUB_STATE`, the OIDC token request variables and the cache and
runtime tokens — so no command the agent runs can find them, not even in a process's starting
environment under `/proc`, and none can plant a variable or a binary for later steps or mint an
OIDC token. The output and job summary paths stay, because the action writes the answer to them.
That is not a sandbox: the provider credential is in the agent's process, where a command can
read it, and a command can still go looking for the runner's files on disk, which is why
privileged steps belong in another job.

Provider credentials belong in the step's `env:`. The action takes no API key input, so a key never
passes through an action input that a failure could echo. Exchanging a GitHub OIDC token for
short-lived provider credentials, the way
[Anthropic's](https://github.com/anthropics/claude-code-action) and
[OpenAI's](https://github.com/openai/codex-action) own actions can, is not implemented here yet; it
is tracked in [#8411](https://github.com/pydantic/pydantic-ai/issues/8411).

## Observability

A run nobody watched needs to have left a record somewhere. Put a [Logfire](logfire.md) token in
the step's `env:`, and the action installs Logfire, configures it, and instruments the agent
before the run starts:

```yaml {test="skip" lint="skip"}
- uses: pydantic/pydantic-ai-action@v1
  with:
    model: {{ openai:medium }}
    prompt: Summarize what changed in this pull request.
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    LOGFIRE_TOKEN: ${{ secrets.LOGFIRE_TOKEN }}
```

`OTEL_EXPORTER_OTLP_ENDPOINT` does the same for any other OpenTelemetry backend. An agent that
configures Logfire itself keeps its own configuration: the action's happens before your agent is
imported, so whatever the agent sets up afterwards is what the run uses.
