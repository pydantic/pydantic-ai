# Coder

`Coder` gives a Pydantic AI agent tools and guidance for investigating, editing, and testing a codebase.
It works in the run's workspace, on this machine or in a sandbox. It is a regular combined capability made from [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/), [`Shell`](https://pydantic.dev/docs/ai/harness/shell/), [`RepoContext`](https://pydantic.dev/docs/ai/harness/repo-context/), [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/), and the [context management](https://pydantic.dev/docs/ai/harness/compaction/) capabilities, so you can use it whole or take it apart.

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
In a workspace without `rg`, such as a sandbox image that lacks it, `list_files` and `grep` use a single POSIX git/grep/find command instead.

For remote workspaces, use file tools (`grep`, `find_files`, `read_file`) instead of sending whole files through shell `cat`. Install `rg` and `git` in the sandbox image for fast search (the Coder extra installs `rg` on the agent host, not in a remote image). For large or generated trees, run `rg -n 'pattern' path` or `rg --files` through Shell and cap its output. Without `rg`, command-capable POSIX workspaces use one in-sandbox git/grep/find command for `grep`, `list_files`, and `search_files` (after an initial `rg` probe). Backends that are filesystem-only use bounded file walks. The POSIX fallback honors nested `.gitignore` in repositories and search-root `.ignore` with git available; nested `.ignore` rules are not applied by the POSIX fallback, and rg-specific regex features require `rg`. Without git, the POSIX fallback cannot apply ignore files. Searches report output and result caps rather than presenting partial results as complete.
Add a provider extra such as `[coder,anthropic]` when needed.
`Coder` works in the run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/). Here that is the current directory on your machine:

<!-- Keep this blown-out example in sync across docs/coder.md, docs/index.md, README.md, pydantic_ai_harness/coder/README.md, and examples/coding_agent.py. -->

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness.coder import Coder

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[LocalWorkspace('.'), Coder()],
)
result = agent.run_sync('Find out why tests/test_parser.py fails and fix the bug it caught.')
print(result.output)
```

File paths resolve from the workspace's working directory, and commands start there. To work in an isolated cloud machine instead, swap `LocalWorkspace` for a sandbox capability (Modal, E2B, or Sprites); the rest of the code stays the same. Commands run without an allowlist, and the file tools' path limits don't apply to them.

With [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/), for example:

```python
from pydantic_ai_harness.modal_sandbox import ModalSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[ModalSandbox(), Coder()])
```

With `LocalWorkspace`, [`agent.to_cli_sync()`](https://pydantic.dev/docs/ai/cli/) and [`agent.to_web()`](https://pydantic.dev/docs/ai/web/) work in the same directory. With a sandbox, the CLI keeps one sandbox for the session, but `to_web()` starts a new one for each message because the web protocol does not carry the workspace ref, so use `LocalWorkspace` when files must persist between web messages.

The exported `pydantic_ai_harness.coder:coder_agent` is the same agent, model-less and named `coder`, working in the directory that is current when it is imported.
Use it with the Pydantic AI CLI:

```bash
uvx --with "pydantic-ai-harness[coder]" clai -a pydantic_ai_harness.coder:coder_agent -m anthropic:claude-opus-5-5
```

### The command environment

Commands in a `LocalWorkspace` get your `PATH`, `HOME`, `LANG`, `LC_ALL` and `LC_CTYPE`, so they find your tools and their configuration and use your locale, and nothing else from your environment. `python`, `pytest` and other tools resolve through that `PATH`, so when you start the agent with `uv run` they come from the agent project's virtualenv, not necessarily the workspace's; pass `LocalWorkspace('.', env={'PATH': ...})` to point commands at the workspace's own interpreter. Pass only what they need with `env=`:

```python {names="defined"}
import os

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness.coder import Coder

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[LocalWorkspace('.', env={'GH_TOKEN': os.environ['GH_TOKEN']}), Coder()],
)
```

Don't pass `os.environ`: that hands the model's commands every secret in the process, LLM API keys included.

## Sharing a workspace

A workspace outlives the run that used it. To continue the conversation in the same files, pass its messages:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-opus-5-5', capabilities=[LocalWorkspace('.'), Coder()])
result = agent.run_sync('Add a --verbose flag to the CLI.')
result = agent.run_sync('Document the new flag in the README.', message_history=result.all_messages())
```

Message history can't move a `LocalWorkspace` to another directory.

To hand the work to another agent, or start a fresh conversation in the same files, pass the workspace itself:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.workspaces import ReadOnlyWorkspace
from pydantic_ai_harness.coder import Coder

coder = Agent('anthropic:claude-opus-5-5', capabilities=[LocalWorkspace('.'), Coder()])
reviewer = Agent('anthropic:claude-opus-5-5', capabilities=[Coder(instructions='Review the code for bugs.')])

result = coder.run_sync('Add a --verbose flag to the CLI.')
review = reviewer.run_sync('Review the new --verbose flag.', workspace=ReadOnlyWorkspace(result.workspace))
```

The reviewer works in the workspace you pass. [`ReadOnlyWorkspace`](https://pydantic.dev/docs/ai/core-concepts/workspace/#hand-the-workspace-to-another-agent) refuses commands and file changes, so the reviewer gets `read_file`, `list_files`, and `grep` but no shell or editing tools; pass `result.workspace` itself to let it run commands and edit files. With a sandbox, this is how several agents share one isolated machine. [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/) needs nothing extra: each delegate runs in the parent's workspace.

## Composition

`Coder()` is these capabilities, in this order:

1. A `Capability` carrying the default instructions, plus any `instructions=` you pass.
2. [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/)`(content_hashes=False, max_read_chars=50000, tools=FILE_TOOL_NAMES, max_retries=5)`, where
   `FILE_TOOL_NAMES` is `read_file`, `write_file`, `edit_file`, `list_files`, and `grep`. Its `root_dir` is the workspace's working directory.
   Each file tool allows five consecutive retries (for a denied path or a stale edit) rather than the agent's default one,
   so a repeated correctable mistake does not end a long run.
3. [`Shell`](https://pydantic.dev/docs/ai/harness/shell/)`(denied_commands=[], allow_interactive=True, default_timeout=270, tools=['shell'])`.
4. [`RepoContext`](https://pydantic.dev/docs/ai/harness/repo-context/)`(expose_inventory_tool=False)` for repository instructions and structure.
   Pass `repo_context=False` to leave it out when the agent already binds its own `RepoContext`, so the
   instruction files are not loaded twice.
5. [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/)`(include_self=True, agent_folders=None)`, so the agent can hand a self-contained
   sub-task to a fresh run of itself (see below). Pass `sub_agents=False` to leave it out.

Then the plumbing, which the agent never calls directly:

6. [`ClearToolResults`](https://pydantic.dev/docs/ai/harness/compaction/)`(max_fraction=0.7)` and [`WarnNearLimits`](https://pydantic.dev/docs/ai/harness/compaction/)`(max_context_fraction=0.9)`.
7. A private [`ToolOutputLimits`](https://pydantic.dev/docs/ai/harness/tool-output-limits/) specialization that truncates any tool result over 64,000 characters
   without adding a spill-retrieval tool. Its stable ID, `coder_tool_output_limits`, lets durability
   capabilities bind its inherited operations without colliding with a separately configured `ToolOutputLimits`.
8. [`RepairToolArguments`](../repair_tool_arguments/) repairs malformed JSON tool arguments before normal validation (see below).

Every tool comes from `FileSystem`, `Shell`, or `SubAgents`; those pages document each one in full. Build the same
agent from the pieces to change any setting, for example to keep content hashes, add `list_directory`,
or allowlist commands.

## Tools

| Tool | Behavior |
| --- | --- |
| `read_file(path, offset=0, limit=None)` | Zero-based line offset, one-based displayed line numbers, 2,000 lines unless `limit` says otherwise, and at most 50,000 characters of complete lines; the continuation hint names the exact next offset, and a line too long for the window is named and skippable. No hash header. |
| `write_file(path, content)` | Create a file in an existing directory, or replace one. No `expected_hash`. |
| `edit_file(path, old_text, new_text)` or `edit_file(path, replacements=[...])` | Exact replacements, each matching once; a batch is checked in memory and written only if every replacement matches. |
| `list_files(path='.', glob=None)` | `rg --files`, sorted by path, respecting ignore files and skipping hidden files. |
| `grep(pattern, ...)` | Ripgrep search with `path`, `glob`, `file_type`, `ignore_case`, `literal`, and `context` (0 to 20). |
| `shell(command, mode='foreground', timeout=270)` | Unrestricted commands rooted at the workspace that outlive the run. |
| `delegate_task(agent_name, task)` | Hand a self-contained sub-task to `self`, a fresh run of this agent. Present unless `sub_agents=False`. |

Results are bounded by `FileSystem`'s caps (2,000 lines by default and 50,000 characters per `read_file`, 1,000 lines or files per search or listing) and Coder's 64,000-character
tool-output limit; a truncation marker means more output was omitted, so narrow the search rather than
assuming it was complete. A `read_file` window stays under the output limit, so paging by `offset` never skips lines. Use `shell` for `mkdir`, `find`, process inspection, and `kill`. File writes
keep the standalone filesystem's read-only path rules (`.git`, `.env`, keys, and secrets); shell can bypass
these rules. Coder does not include planning or the `run_command` family.

## Sub-agents

With `sub_agents=True` (the default), `Coder` adds [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/)`(include_self=True, agent_folders=None)`,
which gives the agent `delegate_task` and one delegate, `self`: a fresh run of the same agent `Coder` is bound
to. The delegate starts without this conversation, so the agent passes it everything it needs, and it has
everything the agent has -- the same model, workspace, instructions, and capabilities, including an approval
gate, guardrail, or audit hook bound next to `Coder`, so those see the commands a delegate runs too. A
delegate can delegate in turn, up to three levels counting the top-level run (`SubAgents.max_depth`).

Only what is bound to the `Agent` carries over to a delegate; capabilities, toolsets, instructions, and model
settings passed to `run()` do not. So bind `Coder` with `Agent(capabilities=[...])`: passing it to `run()`
raises a `UserError` unless you also pass `sub_agents=False`. Disk agent definitions are not loaded. Pass
`sub_agents=False` to drop `delegate_task`, or compose [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/) yourself for a different roster,
per-delegate budgets, or a model menu.

## Filesystem scope

File tools are scoped to the workspace's working directory by default. For trusted local use,
`Coder(unrestricted_filesystem=True)` sets `FileSystem(root_dir='/',
read_only_patterns=[])`: relative paths still resolve from the working directory, and absolute paths anywhere in
the run's workspace are accepted, such as `/tmp/example.py`. OS permissions and file-change event listeners still apply. This permits
modifying secrets and repository metadata: use it only when you trust the agent and its inputs. Shell commands
were already unrestricted.

## Long-running commands

`shell` is the [`Shell`](https://pydantic.dev/docs/ai/harness/shell/) capability's persistent tool. Foreground waits at most 270 seconds
(or a smaller positive `timeout`) and then returns handles for the same running process; background returns
them immediately. Both end with a PID, an absolute output log path, and an absolute JSON status path (inside the workspace) whose
`exit_code` is `null` while the command runs; foreground puts the last 16,000 bytes of output before them. Commands outlive the agent run, so servers keep running; there
is no completion notification or automatic wake-up after a final response. The Shell page covers the
supervisor, cleanup, and the `CommandStartedEvent`, `CommandOutputEvent`, and `CommandFinishedEvent` progress
events a UI can subscribe to.

The default instructions tell the agent to finish required work before giving a final response: do other
useful work, then poll status and output until completion or a genuine blocker.
Servers may remain running after startup and readiness are verified. Commands get only the environment
the workspace passes (see [The command environment](#the-command-environment)); host files remain
accessible to commands in a local workspace.

## Instructions

The default instructions keep engineering guidance brief: autonomous investigation and completion,
focused changes and verification, and pragmatic DRY, YAGNI, SOLID, and the Zen of Python.
They also ask the agent to leave only the requested change in the project: check behavior with inline
shell scripts rather than new files, add tests only where the project already has them, and delete
scratch files before finishing.
Tool descriptions supply tool usage; `RepoContext` supplies repository instructions and structure.
The instructions also name the workspace's working directory as the project, where shell commands start and the file tools work.
`Coder(instructions='...')` appends project-specific guidance rather than replacing defaults.
Use it for additional policy, such as file-size limits or a preferred verification workflow.

## Tool argument repair

`Coder` composes [`RepairToolArguments`](../repair_tool_arguments/), which uses `json-repair` for malformed JSON before Pydantic AI validates the tool schema.
Valid JSON and already-parsed arguments pass through unchanged. Missing fields and invalid types still
follow normal validation and retry behavior. Repair applies to tools added alongside Coder too.
If the repair parser raises a value or recursion error, original arguments go through normal validation.

Repair is heuristic: malformed input can be ambiguous, and inferred strings may differ from the model's
intent. It does not supply a schema to the repair library or bypass exact edit matching.
Each attempt emits a `repair_tool_arguments` span through `ctx.tracer`, without arguments or file
contents. Other Coder operations rely on core tool spans and on the events its `FileSystem` and `Shell`
capabilities emit.

## Durable execution

`Coder` works under Temporal, DBOS, and Prefect. [Durable execution](https://pydantic.dev/docs/ai/harness/durable-execution/) shows an example for each engine and what works on each.

## Upgrading

This release makes the workspace the single place that decides where an agent works. Removed arguments are still accepted, emit a `HarnessDeprecationWarning` naming the fix, and are ignored.

- **Attach a workspace.** `Coder`, `FileSystem`, `Shell`, `RepoContext`, and `Macroscope` fail at run start without one, as do `Skills`, `PydanticAIDocs` (with a local checkout), and `ToolOutputLimits` (when it can spill) unless given their own `workspace=` or store. Add `LocalWorkspace('.')` to the agent's capabilities, as in [Usage](#usage).
- **Set the directory on the workspace.** `Coder('dir')`, `Shell(cwd=)`, `FileSystem(cwd=)`, `Macroscope(cwd=)`, and `RepoContext(workspace_dir=)` are ignored; use `LocalWorkspace('./dir')`.
- **Pass the command environment.** Commands used to inherit your whole environment (`Coder` removed LLM API keys from it). Now they get only your `PATH`, `HOME`, `LANG`, `LC_ALL` and `LC_CTYPE`, plus the workspace's `env` and `Shell(env=)`. Pass what they need, such as an SSH agent socket, a `gh` token or proxy settings, with `LocalWorkspace('.', env={...})` (see [The command environment](#the-command-environment)).
- **`denied_env_patterns` does not filter `LocalWorkspace(env=)`.** It only drops names from `Shell(env=)`, so don't pass all of `os.environ` to the workspace.
- **`FileSystem(root_dir=)`** defaults to the working directory and resolves relative values from it. It must contain the working directory, symlinks that lead outside it are refused, and `root_dir='/'` turns the checks off.
- **Harness files moved into the working directory.** Tool-output spills and Shell background-job files are under `.pydantic-ai-harness/` (git-ignored) instead of `$TMPDIR`. Spills in `.pydantic-ai-harness/tool-output/` are never pruned; delete the directory when you no longer need them. `ToolOutputLimits(store=LocalFileStore())` keeps spills in a temp directory on this machine, without a workspace.
- **Background commands outlive the run.** `start_command` jobs are no longer killed when the run ends, so a later run in the conversation can check them; stop them with `stop_command` or your own cleanup.
- **Local file and shell tools need a POSIX host.** `LocalWorkspace` does not run on Windows; use a sandbox capability there, or WSL.
- **Skills** are read from the workspace at run start and loaded as deferred capabilities. [`Skills(workspace=LocalWorkspaceBackend('/app'))`](https://pydantic.dev/docs/ai/harness/skills/) reads them from somewhere else.
- **[`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/)** spills oversized tool results to the workspace instead of `$TMPDIR`, so it needs one too. A web-only agent can keep the old behaviour with `Researcher(store=LocalFileStore())`.
- **Sub-agent definitions** are read from the workspace at run start, and `~/.agents/agents/` is no longer read. [`SubAgents(workspace=LocalWorkspaceBackend('/app'))`](https://pydantic.dev/docs/ai/harness/subagents/) reads them from somewhere else.
- **[Memory's `FileStore`](https://pydantic.dev/docs/ai/harness/memory/)** keeps its files in the workspace, and receipts in `.memory-operations.json` replace its SQLite journal. `FileStore('.', workspace=LocalWorkspaceBackend('/path'))` keeps them on this machine.
- **Capability Creation** runs only when the workspace is a writable `LocalWorkspace`.
- **Durable runs in flight.** Adding a workspace changes what a durable run records at its start, so Temporal and DBOS runs started before the change no longer replay. Let them finish or version the deployment first; see [deploying changes](https://pydantic.dev/docs/ai/harness/durable-execution/#engine-notes) and the workspace guide's [Durable execution](https://pydantic.dev/docs/ai/core-concepts/workspace/#durable-execution) section.

When retaining `result.workspace` after a run with a provider backend that exposes `aclose()`, finish using it and call `await result.workspace.backend.aclose()` to release its client session. This closes the client, not necessarily the sandbox; follow that provider's deletion API for owned sandboxes.

Sandbox refs identify existing environments; provider-specific cleanup should use an ID-only delete API for refs your application owns (where that provider offers one). Do not create or attach a backend merely to delete a sandbox. Directory upload and preview URLs depend on the provider SDK.

With a remote sandbox such as `ModalSandbox(working_dir='/workspace')`, `Coder` loads repo instructions at run start, which creates the sandbox before the model's first tool call. Use `Coder(repo_context=False)` if the sandbox should be created lazily; the instructions then do not name the working directory. A new sandbox creates its `working_dir` for you.

## Benchmarking

See the [Terminal-Bench 2.1 playbook](TERMINAL_BENCH.md) for running Coder
inside Harbor, pinning the adapter and harness, and inspecting trial results.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/coder/).
