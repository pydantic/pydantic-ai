# Coding Agents and Workspaces

Harness capabilities that touch files or run commands (`Coder`, `FileSystem`, `Shell`, `RepoContext`,
`Macroscope`) act in the run's **workspace**, never on a path they pick themselves. Attach the
workspace as its own capability: `LocalWorkspace` for this machine, or `ModalSandbox` / `E2BSandbox` /
`SpritesSandbox` for an isolated cloud machine. Core workspace semantics (refs, `workspace=`
precedence, `ReadOnlyWorkspace`, continuing from message history) are in the
`building-pydantic-ai-agents` skill's WORKSPACES.md reference; this file covers the harness side.

## Which one do I want?

| Need | Use |
| --- | --- |
| General coding agent: investigate, edit, test, delegate | `Coder()` + a workspace |
| Narrower agent: file tools and/or commands with policy (allowlists, patterns, read-only) | `FileSystem()` and/or `Shell()` + a workspace |
| Untrusted model or repo, or no access to this machine | Swap `LocalWorkspace` for `ModalSandbox()` / `E2BSandbox()` / `SpritesSandbox()`; tool capabilities stay the same |
| Only your own tools use `ctx.workspace` | A workspace capability alone |

Workspace capabilities (`LocalWorkspace` and the sandboxes) register **no tools**. `allowed_commands`,
`root_dir`, and patterns are guardrails against accidents, not isolation.

## Attach a workspace, or the run fails

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError

from pydantic_ai_harness import Shell

agent = Agent('test', capabilities=[Shell()])  # no workspace attached
try:
    agent.run_sync('List the files.')
except UserError as e:
    print(str(e).split('.')[0])
    #> `Shell` needs a workspace, but none is attached to this run
```

- Add `LocalWorkspace('.')` (from `pydantic_ai.capabilities`; `- LocalWorkspace: .` in an agent spec)
  or a sandbox capability, or pass `workspace=` to `run()`.
- Set the directory **on the workspace**: `LocalWorkspace('./repo')`. `Coder('dir')`, `Shell(cwd=)`,
  `FileSystem(cwd=)`, `Macroscope(cwd=)`, `RepoContext(workspace_dir=)` warn and are **ignored**.
- `LocalWorkspace` needs a POSIX host (use WSL or a sandbox on Windows) and is not a sandbox.
  Commands inherit only `PATH`, `HOME`, `LANG`, `LC_ALL`, `LC_CTYPE`; pass the rest with
  `LocalWorkspace('.', env={'GH_TOKEN': ...})`. Never pass `os.environ` (leaks LLM API keys to the
  model's commands). Under `uv run`, `python`/`pytest` resolve from the agent's venv via `PATH`.
- **Harness files** (Shell job logs, sticky-`cd` state, tool-output spills) go in
  `.pydantic-ai-harness/` under the working directory, with a `.gitignore` of `*`. Spills are never
  pruned. `FileSystem` keeps that directory read-only.

## Coder

Combined capability: default instructions + `FileSystem` + `Shell` + `RepoContext` + `SubAgents` +
`ClearToolResults(max_fraction=0.7)`, `WarnNearLimits(max_context_fraction=0.9)`, a `ToolOutputLimits`
that truncates any tool result to 64,000 chars (no `read_tool_result`), `RepairToolArguments`.

```bash
uv add "pydantic-ai-harness[coder]"   # installs ripgrep (rg) on the agent host
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

from pydantic_ai_harness.coder import Coder

agent = Agent(
    'test',  # e.g. 'anthropic:claude-opus-5-5'
    capabilities=[
        LocalWorkspace('.'),
        Coder(instructions='Run `make test` before you finish.'),
    ],
)
```

Tools (no content hashes, 50,000 chars per read; `shell` is unrestricted, waits at most 270 s, returns
the last 16,000 bytes of output after `[... output truncated, N earlier bytes omitted]` plus the full
log's path (`.pydantic-ai-harness/shell/<id>/output.log`), and its commands outlive the run;
`delegate_task` has one delegate, `self`):

- `read_file(path, offset=0, limit=None)`, `write_file(path, content)`,
  `edit_file(path, old_text=None, new_text=None, replacements=None)`
- `list_files(path='.', glob=None)`,
  `grep(pattern, path='.', glob=None, file_type=None, ignore_case=False, literal=False, context=0)`
- `shell(command, mode='foreground', timeout=None)` (`mode='background'` too), `delegate_task(agent_name, task)`

Keyword-only options: `instructions=None` (appended to the default guidance),
`unrestricted_filesystem=False` (`True` sets `FileSystem(root_dir='/', read_only_patterns=[])`),
`repo_context=True`, `sub_agents=True`.

- **Bind `Coder` on the `Agent`**: passing it to `run(capabilities=...)` raises `UserError` unless
  `sub_agents=False`. Delegates re-run the bound agent (same model, workspace, and neighbouring
  capabilities such as approval gates), up to 3 levels deep.
- **Lazy sandbox creation:** `RepoContext` reads the workspace at run start, which creates a sandbox
  before any tool call. `Coder(repo_context=False)` defers it (the instructions then stop naming the
  working directory). Also use it when you bind your own `RepoContext`, or files load twice.
- File tools stay in the working directory with `.git`, `.env*`, keys, and `secrets*` read-only; `shell`
  bypasses all of that. To change a bundled setting (hashes, allowlist, `list_directory`), compose
  `FileSystem`/`Shell`/`RepoContext`/`SubAgents` yourself.
- Your own `ClearToolResults` / `TieredCompaction` / `WarnNearLimits` next to `Coder` run as well (no id
  clash), each at its own trigger. List your `ToolOutputLimits` **after** `Coder` so it sees raw returns;
  listed before, it sees Coder's 64,000-char result. For `shell`, bands must sit under ~16,000 chars.

`pydantic_ai_harness.coder:coder_agent` is a model-less `Agent(name='coder')` with
`LocalWorkspace('.')` (cwd at import) and `Coder()`:

```bash
uvx --with "pydantic-ai-harness[coder]" clai -a pydantic_ai_harness.coder:coder_agent -m anthropic:claude-opus-5-5
```

### Sharing a workspace across runs and agents

```python {test="skip" lint="skip"}
# Same conversation, same files: the history carries the workspace ref.
result = coder.run_sync('Document it.', message_history=result.all_messages())
# Another agent (no workspace capability of its own), same files.
# ReadOnlyWorkspace (pydantic_ai.workspaces) leaves out shell and edit tools.
review = reviewer.run_sync('Review it.', workspace=ReadOnlyWorkspace(result.workspace))
```

History cannot move a `LocalWorkspace` to another directory. With a sandbox, `agent.to_web()` starts a
new sandbox per message (no ref in the web protocol); the CLI keeps one per session.

## FileSystem

Path-scoped file tools. `tools=` defaults to `read_file`, `write_file`, `edit_file`, `list_directory`,
`search_files`, `find_files`, `create_directory`, `file_info`. Opt-in ripgrep tools `list_files` and
`grep` must be named in `tools` (a sandbox image needs its own `rg`, else a POSIX git/grep/find
fallback runs).

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

from pydantic_ai_harness import FileSystem

agent = Agent(
    'test',
    capabilities=[
        LocalWorkspace('.'),
        FileSystem(
            tools=['read_file', 'edit_file', 'list_files', 'grep'],
            allowed_patterns=['*.py', '*.toml'],
            denied_patterns=['**/node_modules/*'],
        ),
    ],
)
```

Tool arguments (scripted tests need them; `expected_hash` exists only with `content_hashes=True`, the default):
`read_file(path, offset=0, limit=None)`, `write_file(path, content, expected_hash=None)`,
`edit_file(path, old_text=None, new_text=None, replacements=None, expected_hash=None)`,
`list_directory(path='.')`, `search_files(pattern, path='.', include_glob=None)`,
`find_files(pattern, path='.')`, `create_directory(path)`, `file_info(path)`; `list_files` and `grep`
as under Coder.

Parameters: `root_dir` (default the working directory; must be the working directory or an **ancestor**
of it, so `FileSystem(root_dir='data')` under `LocalWorkspace('.')` raises `UserError`: attach
`LocalWorkspace('./data')` instead; `'/'` disables containment), `allowed_patterns` /
`denied_patterns` / `read_only_patterns` (`fnmatch`: `*` spans `/`; a directory pattern covers
descendants), `read_only=False` (`True` registers only read tools), `content_hashes=True`
(`expected_hash` concurrency check), `max_read_lines=2000`, `max_read_chars=50_000`, `max_*_results=1000`.

- Passing `read_only_patterns` **replaces** the defaults (`**/.git/*`, `**/.env`, `**/.env.*`, `*.pem`,
  `*.key`, `**/secrets*`, `**/.pydantic-ai-harness/**`).
- Walkers skip dotfiles and dot-directories unless a path or pattern names one explicitly, and filter entries, not the root, by
  `allowed_patterns`. A read-only workspace gets only read tools.
- Veto writes with `@agent.on_event(FileChangeRequestEvent)` + `event.cancel(reason)` (from
  `pydantic_ai_harness.filesystem`); the event carries a unified `diff`.

## Shell

Default tools `run_command`, `start_command`, `check_command`, `stop_command`; `tools=['shell']`
registers only the persistent `shell` tool instead.

```python
from pydantic_ai_harness import LLM_API_KEY_ENV_PATTERNS, Shell

Shell(allowed_commands=['ls', 'cat', 'rg'])  # allowlist; default denylist is dropped
Shell(denied_commands=[])  # no command-name filtering
Shell(
    env={'PYTHONUNBUFFERED': '1', 'ANTHROPIC_API_KEY': 'sk-...'},
    denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,  # filters Shell(env=) only
)
Shell(tools=['shell'], default_timeout=270)  # persistent commands
```

Tool arguments: `run_command(command, timeout_seconds=None)`, `start_command(command)`,
`check_command(command_id)`, `stop_command(command_id)`; `shell(command, mode='foreground', timeout=None)`.

Parameters: `denied_commands` (default `rm`, `rmdir`, `mkfs`, `dd`, `format`, `shutdown`, `reboot`,
`halt`, `poweroff`, `init`), `denied_operators`, `default_timeout=30.0`, `max_output_chars=50_000`
(keeps the **tail**; `shell` already returns at most the last 16,000 bytes), `max_file_bytes=None`
(per-file `ulimit -f`), `persist_cwd=False` (sticky `cd`), `allow_interactive=False`, `env`,
`denied_env_patterns`.

- `ValueError` for: both `allowed_commands` and non-empty `denied_commands`; `max_file_bytes` with
  `persist_cwd=True` or `tools=['shell']`; `tools=['shell']` with `default_timeout` outside (0, 270].
- Only the first token is checked: `python`, `git`, `make`, `uv` can spawn anything.
- `denied_env_patterns` does **not** filter `LocalWorkspace(env=)`.
- Background commands **outlive the run**; a later run on the same workspace can check or stop them.
  Clean up with `stop_command` or your own code. They need `mv` and `base64` in the workspace, and
  `setsid` for process-group stops.
- `persist_cwd` does not apply to `shell`. A read-only workspace gets no Shell tools.

## Sandbox workspaces

| | `ModalSandbox` | `E2BSandbox` | `SpritesSandbox` |
| --- | --- | --- | --- |
| Extra | `[modal]` | `[e2b]` | `[sprites]` |
| Credentials | `modal token new`, or `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` | `E2B_API_KEY` | `SPRITE_TOKEN` |
| Lifetime | `sandbox_timeout=86_400` (10 to 86,400), optional `idle_timeout`; then terminated | `sandbox_timeout=3_600` (Pro: up to 86,400); then **paused**, resumed on attach | Unlimited: sleeps when idle, persists until deleted |
| Default cwd / user | `/root`, `root` | `/home/user`, `user` | `/home/sprite`, `sprite` |
| `rg` preinstalled | Yes | No | No |
| New-sandbox options | `image`, `app_name`, `create_app_if_missing` | `template`, `allow_internet_access` | `runtime`, `client` |

All take `working_dir` (absolute; created on a new sandbox, must exist on an attached one) and `env`
(nothing from your machine is passed through). All are asyncio-only and reject `defer_loading=True`.

One sandbox per conversation in a web service: store the ref and the history, reattach each turn,
destroy at the end.

```python {test="skip"}
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.run import AgentRunResult
from pydantic_ai.workspaces import WorkspaceRef, WorkspaceUnavailableError

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox

REF = TypeAdapter(WorkspaceRef)  # {"provider": ..., "id": ...}; no credentials


@dataclass
class Conversation:  # your DB row
    sandbox_ref: bytes | None = None
    history: bytes | None = None  # result.all_messages_json()


hooks = Hooks()  # a failed run returns no result, so record its ref here


@hooks.on.run_error
async def keep_ref(
    ctx: RunContext[Conversation], *, error: BaseException
) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        ctx.deps.sandbox_ref = REF.dump_json(ctx.workspace.ref)
    raise error


sandbox = ModalSandbox(working_dir='/root/project', idle_timeout=600)
agent = Agent(
    'anthropic:claude-opus-5-5',
    deps_type=Conversation,
    capabilities=[sandbox, Coder(), hooks],
)


async def turn(conv: Conversation, prompt: str) -> str:
    ref = REF.validate_json(conv.sandbox_ref) if conv.sandbox_ref else None
    history = ModelMessagesTypeAdapter.validate_json(conv.history) if conv.history else None
    try:
        result = await agent.run(prompt, deps=conv, message_history=history, workspace=ref)
    except WorkspaceUnavailableError:  # expired or destroyed: start a new sandbox
        result = await agent.run(prompt, deps=conv, message_history=history, workspace='new')
    if result.workspace.ref is not None:
        conv.sandbox_ref = REF.dump_json(result.workspace.ref)
    conv.history = result.all_messages_json()
    return result.output


async def end_conversation(conv: Conversation) -> None:  # or on a TTL sweep
    if conv.sandbox_ref is not None:
        await ModalSandbox().destroy(REF.validate_json(conv.sandbox_ref))
```

- **Pydantic AI never terminates the sandbox**: it bills until `await <Sandbox>().destroy(ref)` or its
  lifetime ends. `destroy` acts by ID (any instance works; Sprites uses the instance's `client=`), does
  not wake the sandbox, returns quietly if it is gone, and raises `ValueError` for another provider's
  ref. `after_run` skips failed runs; one-shot jobs destroy `ctx.workspace.ref` in `run_error`.
- `workspace=` wins over the ref in `message_history`, and UI adapters strip refs from client-sent
  history, so keep the ref server-side. Attaching to a gone sandbox raises `WorkspaceUnavailableError`
  (so do rejected credentials); no empty replacement is created unless you pass `workspace='new'`.
- Default images lack project dependencies (even `pytest`). To prepare one up front, build
  `ModalSandboxBackend(working_dir=...)` (or E2B/Sprites), use `Workspace(backend)` (`write_text`,
  `run([...], timeout=300)`), pass `workspace=backend`, destroy `backend.ref` in `finally`; a Sprites
  backend you build needs `await backend.aclose()`.
- Idempotent per-turn setup goes in a `@hooks.on.before_run` hook: `await ctx.workspace.run('test -d
  repo/.git || git clone <url> repo', shell=True, timeout=300)`. Clone into a subdirectory: `shell`,
  background commands, and output spills create `.pydantic-ai-harness/` in the working directory, so
  `git clone <url> .` fails on later turns (or use `git init` + `git fetch`).
- For an existing native sandbox pass `workspace=ModalSandboxBackend(sandbox=...)` (same for E2B and
  Sprites); the capability's settings don't apply to it.
- On E2B and Sprites, a background child that inherits stdout/stderr keeps `run()` waiting: redirect
  its output (`start_command` already does).
- `ModalSandbox` warns `ModalSandboxNoToolsWarning` when the run has no `Shell`/`FileSystem` tools;
  pass `warn_if_no_tools=False` if only your own tools use it.

## Durable execution

Put the workspace, harness capabilities, and `TemporalDurability()` / `DBOSDurability()` /
`PrefectDurability()` on the agent at construction, with a `name`. The run records its workspace ref;
every activity reattaches to that environment.

| Feature | Temporal | DBOS | Prefect |
| --- | --- | --- | --- |
| File tools, background jobs, `persist_cwd`, `FileChangeRequestEvent` | Yes | Yes | Yes |
| `run_command`, `shell` | Yes, within the activity timeout | Yes | Yes |
| Live tool events | File-change request and write events only | Yes | Yes |
| `delegate_task` | To `self` only | Yes | Yes |

- Temporal activities get 60 s; `run_command` and `shell` get 300 s. Longer `run_command` timeouts need
  `SetToolMetadata(tools=['run_command'], temporal=ActivityConfig(start_to_close_timeout=...))`; for
  longer work use `start_command` or `shell(mode='background')`. Approval listeners re-run on replay.
- A worker dying between sandbox creation and ref recording leaves an orphan sandbox. Adding or
  removing capabilities changes replay history: drain workflows or version workers first.

## RepoContext

Loads `CLAUDE.md`/`AGENTS.md` into static instructions at run start and exposes
`inventory_agent_context()` (locates `.claude`/`.agents`/`.codex`/`.grok` skills, agents, hooks).

```python
from pydantic_ai_harness import RepoContext

RepoContext(home_dir='..', nested_traversal=True)  # walk up one level; follow traversal
```

`home_dir=None` (default) scans only the working directory; set it to walk up to that ancestor. It is
a workspace path (on E2B, `/home/user`, not `Path.home()`); `~` raises `UserError`.
`nested_traversal=True` surfaces a directory's instruction file when `FileSystem` reads or lists it
(`nested_inject='pointer'` or `'contents'`), in the message tail so the cached prefix stays stable.

## Macroscope

`run_macroscope_review(base=None)` runs `macroscope codereview --raw` in the workspace and returns a
`MacroscopeReview` of `MacroscopeIssue` findings; the agent fixes them with its other tools. No extra,
but the `macroscope` CLI must be installed and signed in **inside the workspace**. Options: `base`,
`command='macroscope'`, `timeout=600.0`, `guidance` (`None` default text, `''` none). Pair with `Coder`
or `FileSystem`/`Shell`: `Agent(..., capabilities=[LocalWorkspace('.'), Coder(), Macroscope(base='main')])`.

## LocalStack

AWS CLI against emulated AWS. It does **not** use the run workspace: `aws` and `docker` run on the
agent's host. No extra. Tools: `aws_cli` (command without `aws` or `--endpoint-url`; argv, no shell)
and `localstack_health`.

`LocalStack(allowed_services=['s3', 'dynamodb'])` connects to an instance you started;
`LocalStack(manage_container=True)` starts a fresh Docker container per run. `allowed_services`/`denied_services` are mutually exclusive. The default image needs
`LOCALSTACK_AUTH_TOKEN` (forwarded automatically); concurrent managed runs need distinct ports. The AWS
CLI can read and write host files (`file://`, `s3 cp`).

## CLAI 2

`pydantic-clai2` is a separate terminal client (`clai2`) whose default agent is
`Coder(unrestricted_filesystem=True)` in the launch directory, with no sandbox or approval layer.
`clai2 --worktree NAME` starts in a new git worktree (not a sandbox). To chat with your own agent:
`asyncio.run(chat(agent, deps=None))` with `from pydantic_clai2 import chat`.

## See also

- https://pydantic.dev/docs/ai/harness/coder/
- https://pydantic.dev/docs/ai/harness/filesystem/
- https://pydantic.dev/docs/ai/harness/shell/
- https://pydantic.dev/docs/ai/harness/modal-sandbox/
- https://pydantic.dev/docs/ai/harness/e2b-sandbox/
- https://pydantic.dev/docs/ai/harness/sprites-sandbox/
- https://pydantic.dev/docs/ai/harness/durable-execution/
- https://pydantic.dev/docs/ai/harness/repo-context/
- https://pydantic.dev/docs/ai/harness/macroscope/
- https://pydantic.dev/docs/ai/harness/localstack/
- https://pydantic.dev/docs/ai/harness/clai2/
- https://pydantic.dev/docs/ai/core-concepts/workspace/
