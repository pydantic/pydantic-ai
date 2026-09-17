# Coder

`Coder` gives a Pydantic AI agent tools and guidance for investigating, editing, and testing a local codebase.
It is a regular combined capability made from [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/), [`Shell`](https://pydantic.dev/docs/ai/harness/shell/), [`RepoContext`](https://pydantic.dev/docs/ai/harness/repo-context/), and the [context management](https://pydantic.dev/docs/ai/harness/compaction/) capabilities, so you can use it whole or take it apart.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Usage

Install the Coder extra to include ripgrep (`rg`), which backs the `list_files` and `grep` tools:

uv:

```bash
uv add "pydantic-ai-harness[coder]"
```

pip:

```bash
pip install "pydantic-ai-harness[coder]"
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

## Composition

`Coder(workspace)` is these capabilities, in this order:

1. A private hook that repairs malformed JSON tool arguments before normal validation (see below).
2. A `Capability` carrying the default instructions, plus any `instructions=` you pass.
3. [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/)`(root_dir=workspace, content_hashes=False, max_read_chars=60000, tools=FILE_TOOL_NAMES)`, where
   `FILE_TOOL_NAMES` is `read_file`, `write_file`, `edit_file`, `list_files`, and `grep`.
4. [`Shell`](https://pydantic.dev/docs/ai/harness/shell/)`(cwd=workspace, denied_commands=[], allow_interactive=True, default_timeout=270, denied_env_patterns=LLM_API_KEY_ENV_PATTERNS, tools=['shell'])`.
5. [`RepoContext`](https://pydantic.dev/docs/ai/harness/repo-context/)`(workspace_dir=workspace, expose_inventory_tool=False)` for repository instructions and structure.
   Pass `repo_context=False` to leave it out when the agent already binds its own `RepoContext`, so the
   instruction files are not loaded twice.
6. [`ClearToolResults`](https://pydantic.dev/docs/ai/harness/compaction/)`(max_fraction=0.7)` and [`WarnNearLimits`](https://pydantic.dev/docs/ai/harness/compaction/)`(max_context_fraction=0.9)`.
7. A private [`ToolOutputLimits`](https://pydantic.dev/docs/ai/harness/tool-output-limits/) specialization that truncates any tool result over 64,000 characters
   without adding a spill-retrieval tool.

Every tool comes from `FileSystem` or `Shell`; those pages document each one in full. Build the same
agent from the pieces to change any setting, for example to keep content hashes, add `list_directory`,
or allowlist commands.

## Six tools

| Tool | Behavior |
| --- | --- |
| `read_file(path, offset=0, limit=None)` | Zero-based line offset, one-based displayed line numbers, up to 2,000 lines or 60,000 characters of complete lines; the continuation hint names the exact next offset, and a line too long for the window is named and skippable. No hash header. |
| `write_file(path, content)` | Create a file in an existing directory, or replace one. No `expected_hash`. |
| `edit_file(path, old_text, new_text)` or `edit_file(path, replacements=[...])` | Exact replacements, each matching once; a batch is checked in memory and written only if every replacement matches. |
| `list_files(path='.', glob=None)` | `rg --files`, sorted by path, respecting ignore files and skipping hidden files. |
| `grep(pattern, ...)` | Ripgrep search with `path`, `glob`, `file_type`, `ignore_case`, `literal`, and `context` (0 to 20). |
| `shell(command, mode='foreground', timeout=270)` | Unrestricted commands rooted at the workspace that outlive the run. |

Results are bounded by `FileSystem`'s caps (2,000 lines or 60,000 characters per `read_file`, 1,000 lines or files per search or listing) and Coder's 64,000-character
tool-output limit; a truncation marker means more output was omitted, so narrow the search rather than
assuming it was complete. A `read_file` window stays under the output limit, so paging by `offset` never skips lines. Use `shell` for `mkdir`, `find`, process inspection, and `kill`. File writes
keep the standalone filesystem's protected-path rules (`.git`, `.env`, keys, and secrets); shell can bypass
these rules. Coder does not include planning, delegation, or the run-scoped `run_command` family.

## Filesystem scope

File tools are workspace-scoped by default. For trusted local use,
`Coder(unrestricted_filesystem=True)` sets `FileSystem(root_dir=<workspace drive root>, cwd=workspace,
protected_patterns=[])`: relative paths still resolve from the workspace, and absolute paths anywhere on
the drive are accepted. On POSIX this permits paths such as `/tmp/example.py`; on Windows this covers the
workspace drive, not other drives. OS permissions and file-change event listeners still apply. This permits
modifying secrets and repository metadata: use it only when you trust the agent and its inputs. Shell commands
were already unrestricted.

## Long-running commands

`shell` is the [`Shell`](https://pydantic.dev/docs/ai/harness/shell/) capability's persistent tool. Foreground waits at most 270 seconds
(or a smaller positive `timeout`) and then returns handles for the same running process; background returns
them immediately. Both end with a PID, an absolute output log path, and an absolute JSON status path whose
`exit_code` is `null` while the command runs; foreground puts the last 16,000 bytes of output before them. Commands outlive the agent run, so servers keep running; there
is no completion notification or automatic wake-up after a final response. The Shell page covers the
supervisor, cleanup, and the `CommandStartedEvent`, `CommandOutputEvent`, and `CommandFinishedEvent` progress
events a UI can subscribe to.

The default instructions tell the agent to finish required work before giving a final response: do other
useful work, then `sleep 60` and inspect status and output repeatedly until completion or a genuine blocker.
Servers may remain running after startup and readiness are verified. Common LLM API-key environment
variables are filtered from command environments; other host credentials and files remain accessible.

## Instructions

The default instructions emphasize autonomous investigation, focused edits, tests, DRY, YAGNI, SOLID,
and pragmatic simplicity. The 600-line suggestion applies to new files, not a mandate to split existing
large files. `Coder(instructions='...')` appends project-specific guidance rather than replacing defaults.
The instructions adapt the software-work and autonomy guidance in Code Puppy's `agent_code_puppy.py`
and `cli_runner.py`, without its identity or tone.

## Tool argument repair

`Coder` uses `json-repair` for malformed JSON before Pydantic AI validates the tool schema.
Valid JSON and already-parsed arguments pass through unchanged. Missing fields and invalid types still
follow normal validation and retry behavior. Repair applies to tools added alongside Coder too.
If the repair parser raises a value or recursion error, original arguments go through normal validation.

Repair is heuristic: malformed input can be ambiguous, and inferred strings may differ from the model's
intent. It does not supply a schema to the repair library or bypass exact edit matching.
Each attempt emits a `coder.repair_tool_arguments` span through `ctx.tracer`, without arguments or file
contents. Other Coder operations rely on core tool spans and on the events its `FileSystem` and `Shell`
capabilities emit.

## Benchmarking

See the [Terminal-Bench 2.1 playbook](TERMINAL_BENCH.md) for running Coder
inside Harbor, pinning the adapter and harness, and inspecting trial results.

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/coder/).
