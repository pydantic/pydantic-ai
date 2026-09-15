---
title: Coder
description: Autonomous coding with six tools and context management.
---

# Coder

`Coder` gives a Pydantic AI agent tools and guidance for investigating, editing, and testing a local codebase.
It is a regular combined capability: Coder-specific tools and argument repair compose with repository context and context management.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Benchmarking

See the [Terminal-Bench 2.1 playbook](https://github.com/pydantic/pydantic-ai-harness/blob/main/pydantic_ai_harness/coder/TERMINAL_BENCH.md)
for running Coder inside Harbor, pinning the adapter and harness, and inspecting trial results.

## Usage

Install the Coder extra to include ripgrep (`rg`) for file listing and search:

```bash
pip/uv-add "pydantic-ai-harness[coder]"
```

The extra installs `ripgrep==14.1.0` except on Android, where `rg` must be supplied separately on `PATH`.
Add a provider extra such as `[coder,anthropic]` when needed.
Commands run on the host without an allowlist;
use an OS-level sandbox or container for untrusted work. Path restrictions on file tools are not a shell sandbox.

<!-- Keep this blown-out example in sync across docs/coder.md, docs/index.md, README.md, pydantic_ai_harness/coder/README.md, and examples/coding_agent.py. -->

```python
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder

agent = Agent(
    'anthropic:claude-fable-5',
    name='coder',
    capabilities=[Coder('.')],
)
```

```python
result = agent.run_sync('Investigate the failing parser test, fix the cause, and run focused checks.')
print(result.output)
```

The exported `pydantic_ai_harness.coder:coder_agent` is the same composition, model-less and named `coder`.
Use it with the Pydantic AI CLI:

```bash
uvx --with "pydantic-ai-harness[coder]" clai -a pydantic_ai_harness.coder:coder_agent -m anthropic:claude-fable-5
```

## Six tools

| Tool | Behavior |
| --- | --- |
| `read_file(path, offset=0, limit=None)` | Zero-based line offset, one-based displayed line numbers, up to 2,000 lines. No hash header. Reads stream up to 60,000 content characters; lines above 65,536 bytes require shell inspection. |
| `write_file(path, content)` | Replace a file or create it in an existing directory. No expected hash. |
| `edit_file(path, old_text=..., new_text=...)` | Replace exactly one occurrence of a non-empty string. |
| `list_files(path='.', glob=None, limit=200)` | `rg --files`, respecting ignore rules. Returns at most 1,000 lines. |
| `grep(pattern, ...)` | Ripgrep search with `path`, `glob`, `file_type`, `ignore_case`, `literal`, `context` (0-20), and `limit` (1-1,000). |
| `shell(command, mode='foreground', timeout=270)` | Unrestricted foreground or background commands rooted at the workspace. |

Edits also accept `replacements=[{'old_text': 'before', 'new_text': 'after'}, ...]` instead of the single pair.
Each replacement must match exactly once in the result of the preceding replacement. All replacements are
validated in memory before writing; a failed batch leaves the file unchanged. Do not mix the two forms.
Read existing files first. File writes retain the standalone filesystem's protected-path rules
(`.git`, `.env`, keys, and secrets); shell can bypass these rules.

Search output is bounded by returned lines (including context) and a 64,000-character cap.
A truncation marker means more output was omitted; narrow the search rather than assuming it was complete.
Use `shell` for `mkdir`, `find`, process inspection, and `kill`. Coder does not include planning,
delegation, directory-creation, file-info, process-check, or process-start/stop tools.

## Long-running commands

Foreground waits at most 270 seconds (or a smaller positive `timeout`) and then returns handles for the
same running process. Background returns those handles immediately. Both report a supervisor/session PID,
an absolute output log path, and an absolute JSON status path. The status contains the child PID and
`exit_code` (`null` while running); it may not exist until the supervisor has started. Output combines
stdout and stderr; foreground returns the final 16,000 bytes available when it returns.

A detached supervisor publishes status and reaps the command. A daemon thread reaps the supervisor while
the calling Python process is alive. Cancelled foreground calls terminate their process group because they cannot return handles.
Successfully returned commands outlive individual agent runs and event loops, allowing
servers to keep running. There is no agent scheduler, completion notification, or automatic wake-up after
a final response. Commands and logs are host-local, not replay-safe durable workflow activities.

Use `shell` to read the returned absolute log/status paths (they are outside the file tools' workspace).
On POSIX, `kill -- -PID` targets the process group. The caller owns stopping servers and deleting their
log directories when no longer needed. Logs are not rotated: bound verbose long-running commands yourself.
A process killed externally before its supervisor publishes completion may leave a running status; inspect
its PID as well. Windows callers should use the platform's process-tree termination command instead.

Default guidance tells the agent to finish required work before giving a final response: do other useful
work, then `sleep 60` and inspect status/output repeatedly until completion or a genuine blocker.
Servers may remain running after startup and readiness are verified. Common LLM API-key environment
variables are filtered from command environments; other host credentials and files remain accessible.

## Instructions and composition

The default instructions emphasize autonomous investigation, focused edits, tests, DRY, YAGNI, SOLID,
and pragmatic simplicity. The 600-line suggestion applies to new files, not a mandate to split existing
large files. `Coder(instructions='...')` appends project-specific guidance rather than replacing defaults.
The instructions adapt the software-work and autonomy guidance in Code Puppy's `agent_code_puppy.py`
and `cli_runner.py`, without its identity or tone.

The composition, in order, is:

1. Private malformed-JSON repair before normal tool validation.
2. A regular `Capability` with default instructions and the six Coder tools.
3. `RepoContext(workspace_dir=..., expose_inventory_tool=False)` for repository instructions and structure.
4. `ClearToolResults(max_fraction=0.7)` and `WarnNearLimits(max_context_fraction=0.9)`.
5. Private `ToolOutputLimits` specialization using `Band(over=64000, action=Truncate(max_chars=64000))`,
   without the spill-retrieval tool. It adds no model calls.

Use `Coder` for this exact composition, including its private tool implementations and JSON repair.
Standalone `FileSystem`, `Shell`, `Planning`, and `SubAgents` remain available with their existing APIs
for consumers assembling different agents. To migrate, remove Coder's `allowed_commands` and `subagents`
constructor arguments and the `DEFAULT_ALLOWED_COMMANDS` import. Add standalone capabilities explicitly
when their additional tools are wanted. Replace old `run_command`/`start_process` calls with `shell`.

## Tool argument repair

`Coder` uses `json-repair` for malformed JSON before Pydantic AI validates the tool schema.
Valid JSON and already-parsed arguments pass through unchanged. Missing fields and invalid types still
follow normal validation and retry behavior. Repair applies to tools added alongside Coder too.
If the repair parser raises a value or recursion error, original arguments go through normal validation.

Repair is heuristic: malformed input can be ambiguous, and inferred strings may differ from the model's
intent. It does not supply a schema to the repair library or bypass exact edit matching.
Each attempt emits a `coder.repair_tool_arguments` span through `ctx.tracer`, without arguments or file
contents. Other Coder operations rely on core tool spans. Writes and edits emit filesystem change-request and completion events. Bounded reads do not compute whole-file hashes or emit hash-bearing read events.

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/coder/).

## API reference

::: pydantic_ai_harness.coder.Coder
