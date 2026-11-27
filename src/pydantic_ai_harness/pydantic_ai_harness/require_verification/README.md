# Require Verification

Send a coding agent back to verify its work when it tries to finish after editing code without a passing check since its last edit. Use it on agents that edit code, where a claim of finished, verified work should rest on a check that ran after the final edit.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/require_verification/)

## Usage

uv:

```bash
uv add pydantic-ai-harness
```

pip:

```bash
pip install pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import FileSystem, RequireVerification, Shell

agent = Agent('anthropic:claude-opus-5-5', capabilities=[FileSystem(), Shell(), RequireVerification()])
```

`RequireVerification` records what the agent's tool calls do during a run:

- An **edit** is a successful [`FileSystem`](../file_system/) `write_file` or `edit_file` call, recognized by its tool kind whatever it is named, or a shell command matching `edit_commands`, such as `ruff format .` (but not `ruff format --check .`), `sed -i`, `git checkout`, `mv`, `rm`, an `sh -c`, `python -c`, or `node -e` script, or a `>` redirect. Edits that `write_file` or `edit_file` make to files matching `exempt_paths` don't count. Shell edit commands always count, since the file they touch isn't known.
- A **check** is a shell command matching `verification_commands`, such as `pytest`, `uv run --no-sync pyright`, `.venv/bin/pytest`, `ruff format --check .`, `pnpm -r test`, or `cargo build`. The [`Shell`](../shell/) tool's result tells whether it passed.

Each edit makes every earlier check result stale. When the run would end after an edit, it is sent back unless at least one check has run since then and the latest run of each passed: an instruction carrying `instructions` is added to the conversation and the model answers again, with its tools available. After `max_attempts` of these redirects in one run, the run ends anyway and reports it with a `CompletionUnverifiedEvent`.

| Field | Default | Meaning |
|---|---|---|
| `max_attempts` | `2` | How many times per run the agent is sent back to verify |
| `instructions` | `DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS` | What the agent is told each time it is sent back, including how to run a check so it counts |
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
- A command's exit status is that of its last `;`- or newline-separated statement, so a command counts as a check when one of the `&&`-joined steps of that statement starts with a match for a `verification_commands` pattern, after any leading `NAME=value` assignments. `cd pkg; pytest` counts, and `pytest; echo done` does not. A probe such as `pytest --version`, `--help`, or `--collect-only` is not a check. A statement with `|`, `||`, a background `&`, or a command substitution does not count, and neither does a command with `exit`, `exec`, or `return`, because the exit status can then hide a failed check. File-descriptor duplications such as `2>&1`, and separators and pipes inside quotes, are ignored for this.
- A command counts as an edit when an `edit_commands` pattern matches anywhere in it outside quoted strings. The default `>` pattern skips `->`, `>&`, and redirections to `/dev/null`. An edit command is not also a check, so `ruff check --fix` makes earlier checks stale and leaves the run with none.
- A check passed when `run_command` reports no `[exit code: N]`, timeout, or file-size limit notice, or when the `shell` tool reports an `exit_code` of `0`. Results are read in the formats of `Shell`'s tools and of ACP's `run_command`, which reports `[exited with code N]` or `[terminated by signal S]`. Only the `shell` tool's result, recognized by its `PID: ... (supervisor; ...)` handles, is read for a JSON status, so a check whose own output ends in JSON still counts. A custom shell tool must end a failed command's result with one of these lines, or its failures count as passes. A command started with `start_command` or the `shell` tool's background mode, or still running when the `shell` tool returns, never counts as a check, even after it finishes, and clears the result of an earlier run of the same command. Run checks in the foreground.
- Results are kept per command text, not per working directory. With `Shell(persist_cwd=True)`, a later run of the same command in another directory replaces the earlier result. Use a command that names its directory, such as `cd pkg && pytest`, when the directory matters.
- Results are read before [Tool Output Limits](../tool_output_limits/) truncates, spills, or summarizes them, wherever it is listed.
- Calls from one model response run in parallel. A check that started before an edit in another call finished records no result, as it may have checked the code from before the edit.
- Commands outside `edit_commands` are not edits. Extend it with the commands your agent uses to change files, such as a code generator.

## Ending unverified

- The first time the run would end unverified after every redirect is used, it emits a `CompletionUnverifiedEvent` with the number of `attempts` and the `failed_commands`, the checks whose latest run since the last edit failed. Listen for it with [`Hooks`](https://pydantic.dev/docs/ai/core-concepts/hooks/) to fail a CI job or flag the result:

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


agent = Agent('anthropic:claude-opus-5-5', capabilities=[FileSystem(), Shell(), RequireVerification(), hooks])
```

- The event is dispatched immediately, so listeners get it before the run returns. It is emitted at most once per run, even when another capability, such as `RefineOutput`, continues the run after it. It is not recorded in the message history.
- With `max_attempts=0` the agent is never sent back, and an unverified run only emits the event.

## Redirects

- A redirect costs at least one more model request, more when the model runs checks during it, and counts toward [`UsageLimits`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits) like any other request.
- The instruction is a `VerificationRequiredPart`, an [`InstructionCapabilityPart`](https://pydantic.dev/docs/ai/api/pydantic-ai/messages/#pydantic_ai.messages.InstructionCapabilityPart) kind (a [capability part](https://pydantic.dev/docs/ai/capabilities/custom/#capability-parts) in the application's voice) whose `attempt` field holds the redirect number and never reaches the model. It keeps the default conversation scope, so it stays in the message history. Find redirects by `isinstance(part, VerificationRequiredPart)` (from `pydantic_ai_harness.require_verification`), not by their text.
- Edits, check results, and attempts live in the instance `for_run` creates for each run, so each run starts with none and compaction does not reset them.
- A run resumed after a pause for [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/), such as an approval, still knows about the successful `write_file` and `edit_file` edits made earlier in the same turn (not denied or failed ones): they are read from the message history. So are shell edit commands, for tools offered in the resumed run. Check results from before the pause are not, so the resumed run is sent back if it edited code earlier in the turn and has not run a check since resuming.
- A model that answers a redirect without calling a tool, such as by explaining a blocker, is not sent back again: the run ends and emits the `CompletionUnverifiedEvent`.

Three kinds of run end without a redirect or an event:

- A run that pauses for [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) and returns `DeferredToolRequests`, or ends while the provider is still processing a model request and returns [`PendingModelResponse`](https://pydantic.dev/docs/ai/core-concepts/batch/#ending-a-run-while-it-waits).
- A run whose output did not come from a model request, recognized by a response with no `model_name`.
- `Agent.run_stream()`, which ends at the first output it streams. Use `Agent.run()` with an `event_stream_handler`, or `Agent.iter()`.

## Durable execution

Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), the tool hooks that record edits and checks and the hook that redirects the run (`after_node_run`) run in workflow code. When Temporal, DBOS, or Prefect replays a run, they run again in the same order over the journaled tool results and model responses, and send the run back the same way.

## Telemetry

Each redirect records a zero-duration `verification required` span with `require_verification.attempt`. A run that ends unverified records a `completion unverified` span with `require_verification.attempts`. Neither carries command text.

## Composition

- Any edit makes every earlier check stale, including checks of code the edit cannot affect. Telling them apart would need each check's full dependency closure (fixtures, configuration, imports), and a wrong guess would let unverified work through, so the capability prefers an extra check.
- Edits made through other tools, such as a custom file tool, are not recorded.
- [Refine Output](../refine_output/) asks the agent to review its answer. `RequireVerification` asks it to run checks, and only after it edited code. With both on one agent, each sends the run back for its own reason.
- [Permissions](../permissions/) may ask a person before a check runs. A denied check is not a check result.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/require_verification/).
