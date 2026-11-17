---
title: Require Verification
description: "Send a Pydantic AI coding agent back to run its tests, lint, type checks, or build when it tries to finish after editing code without a passing check."
---

# Require Verification

Send a coding agent back to verify its work when it tries to finish after editing code without a passing check since its last edit. Use it on agents that edit code, where a claim of finished, verified work should rest on a check that ran after the final edit.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

```bash
pip/uv-add pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import FileSystem, RequireVerification, Shell

agent = Agent('{{ anthropic:large }}', capabilities=[FileSystem(), Shell(), RequireVerification()])
```

`RequireVerification` records what the agent's tool calls do during a run:

- An **edit** is a successful [`FileSystem`](filesystem.md) `write_file` or `edit_file` call, recognized by its tool kind whatever it is named, or a shell command matching `edit_commands`, such as `ruff format .`, `sed -i`, or a `>` redirect. Edits to files matching `exempt_paths` don't count.
- A **check** is a shell command matching `verification_commands`, such as `pytest`, `uv run pyright`, `pnpm test`, or `cargo build`. The [`Shell`](shell.md) tool's result tells whether it passed.

Each edit makes every earlier check result stale. When the run would end after an edit, and checks have run since then, the latest run of each must have passed. Otherwise a system prompt carrying `instructions` is added to the conversation and the model answers again, with its tools available. After `max_attempts` of these redirects in one run, the run ends anyway and reports it with a `CompletionUnverifiedEvent`.

| Field | Default | Meaning |
|---|---|---|
| `max_attempts` | `2` | How many times per run the agent is sent back to verify |
| `instructions` | `DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS` | What the agent is told each time it is sent back |
| `verification_commands` | `DEFAULT_VERIFICATION_COMMANDS` | Regular expressions for the commands that count as checks |
| `edit_commands` | `DEFAULT_EDIT_COMMANDS` | Regular expressions for the commands that count as edits |
| `exempt_paths` | `('*.md', '*.mdx', '*.rst')` | Glob patterns for the files whose edits need no check |

```python
from pydantic_ai_harness import RequireVerification

RequireVerification(
    max_attempts=1,
    instructions='Run `just ci` and report its result before you finish.',
    verification_commands=[r'just\s+ci\b'],
)
```

## Recognizing commands

- A command is read from the argument a tool's metadata names with `code_arg_name`, on tools whose metadata sets `code_arg_language='shell'`, as `Shell`'s `run_command` and `shell` tools do.
- A command counts as a check when one of its `&&`-joined steps starts with a match for a `verification_commands` pattern, after any leading `NAME=value` assignments. A command with `;`, `|`, `||`, a background `&`, or a command substitution does not count, because the shell's exit status can then hide a failed check. File-descriptor duplications such as `2>&1`, and text in single quotes, are ignored for this.
- A command counts as an edit when an `edit_commands` pattern matches anywhere in it outside quoted strings. The default `>` pattern skips `->`, `>&`, and redirections to `/dev/null`. An edit command is not also a check, so `ruff check --fix` makes earlier checks stale and leaves the run with none.
- A check passed when `run_command` reports no `[exit code: N]`, timeout, or file-size limit notice, or when the `shell` tool reports an `exit_code` of `0`. A command started with `start_command` or the `shell` tool's background mode, or still running when the `shell` tool returns, never counts as a check, even after it finishes. Run checks in the foreground.
- Commands outside `edit_commands` are not edits. Extend it with the commands your agent uses to change files, such as a code generator.

## Ending unverified

- A run that ends after every redirect is used emits a [`CompletionUnverifiedEvent`](#api-reference) with the number of `attempts` and the `failed_commands`, the checks whose latest run since the last edit failed. Listen for it with [`Hooks`](../hooks.md) to fail a CI job or flag the result:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai_harness import FileSystem, RequireVerification, Shell
from pydantic_ai_harness.require_verification import CompletionUnverifiedEvent

unverified: list[CompletionUnverifiedEvent] = []
hooks = Hooks[None]()


@hooks.on.event(CompletionUnverifiedEvent)
async def record(ctx: RunContext[None], event: CompletionUnverifiedEvent) -> None:
    unverified.append(event)


agent = Agent('{{ anthropic:large }}', capabilities=[FileSystem(), Shell(), RequireVerification(), hooks])
```

- The event is dispatched immediately, so listeners get it before the run returns. It is not recorded in the message history.
- With `max_attempts=0` the agent is never sent back, and an unverified run only emits the event.

## Redirects

- A redirect costs at least one more model request, more when the model runs checks during it, and counts toward [`UsageLimits`][pydantic_ai.usage.UsageLimits] like any other request.
- The instruction is a [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] whose `metadata` holds `'origin': 'pydantic-ai-harness.require_verification.v1'` and the `attempt` number. It keeps the default conversation scope, so it stays in the message history.
- Edits, check results, and attempts live in the instance `for_run` creates for each run, so each run starts with none and compaction does not reset them.

Three kinds of run end without a redirect or an event:

- A run that pauses for [deferred tools](../deferred-tools.md) and returns [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests], or ends while the provider is still processing a model request and returns [`PendingModelResponse`](../batch.md#ending-a-run-while-it-waits).
- A run whose output did not come from a model request, recognized by a response with no `model_name`.
- [`Agent.run_stream()`][pydantic_ai.agent.Agent.run_stream], which ends at the first output it streams. Use [`Agent.run()`][pydantic_ai.agent.Agent.run] with an `event_stream_handler`, or [`Agent.iter()`][pydantic_ai.agent.Agent.iter].

## Durable execution

Under [durable execution](../durable_execution/overview.md), the tool hooks that record edits and checks and the hook that redirects the run (`after_node_run`) run in workflow code. When Temporal, DBOS, or Prefect replays a run, they run again in the same order over the journaled tool results and model responses, and send the run back the same way.

## Telemetry

Each redirect records a zero-duration `verification required` span with `require_verification.attempt`. A run that ends unverified records a `completion unverified` span with `require_verification.attempts`. Neither carries command text.

## Composition

- Any edit makes every earlier check stale, including checks of code the edit cannot affect. Telling them apart would need each check's full dependency closure (fixtures, configuration, imports), and a wrong guess would let unverified work through, so the capability prefers an extra check.
- Edits made through other tools, such as a custom file tool, are not recorded.
- [Refine Output](refine-output.md) asks the agent to review its answer. `RequireVerification` asks it to run checks, and only after it edited code. With both on one agent, each sends the run back for its own reason.
- [Permissions](permissions.md) may ask a person before a check runs. A denied check is not a check result.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/require_verification/).

## API reference

::: pydantic_ai_harness.RequireVerification

::: pydantic_ai_harness.require_verification.CompletionUnverifiedEvent

::: pydantic_ai_harness.require_verification.DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS

::: pydantic_ai_harness.require_verification.DEFAULT_VERIFICATION_COMMANDS

::: pydantic_ai_harness.require_verification.DEFAULT_EDIT_COMMANDS
