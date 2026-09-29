---
title: FileSystem
description: "Give a Pydantic AI agent tools to read, write, edit, list, and search files in the run's workspace, bounded by one directory with allow, deny, and read-only glob patterns."
---

# FileSystem

`FileSystem` gives an agent a fixed set of file tools -- read, write, edit, list,
search, find, create, and inspect -- all scoped to a single `root_dir` in the
run's workspace. Every path is resolved, symlinks included, and
containment-checked before any I/O, and access is filtered through allow, deny, and read-only glob patterns.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/filesystem/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

Letting an agent touch the filesystem directly is risky: path traversal
(`../../etc/passwd`), clobbering `.git`, or leaking `.env` secrets. Hand-rolling the guards around every tool call is
repetitive and easy to get subtly wrong.

`FileSystem` centralizes those guards. It exposes one bounded toolset so you
configure the boundary once and reuse it across agents.

## Usage

Add `FileSystem` to your agent's `capabilities`, together with a workspace for
the files to live in:

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import FileSystem

Path('./workspace').mkdir(exist_ok=True)
agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[LocalWorkspace('./workspace'), FileSystem()],
)

result = agent.run_sync('Read config.toml and tell me the package name.')
print(result.output)
```

`root_dir` defaults to the workspace's working directory, so the agent above can
reach `./workspace` and nothing outside it through the file tools.

## Where files live

`FileSystem` reads and writes files in the run's
[workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/): your machine with `LocalWorkspace`, or
a sandbox. A run without a workspace fails at its start.

With a read-only workspace (`LocalWorkspace(..., read_only=True)`), only the
read tools are offered. `list_files` and `grep` run `rg` inside the workspace;
where it can't run commands, they walk its files instead, as `find_files` and
`search_files` do.

## Tools

`FileSystem` contributes eight tools by default, plus two opt-in ripgrep tools, all path-scoped to `root_dir`:

| Tool | Purpose |
|---|---|
| `read_file` | Read a text file with line numbers and a content hash. Binary files are detected and not dumped. Supports `offset`/`limit` paging; with `max_read_chars`, the whole result (header and hint included) fits the cap unless the cap is smaller than the header and hint themselves, the window ends on the last complete line that fits, and the continuation hint names the first line not shown. |
| `write_file` | Create or overwrite a file. Optional `expected_hash` rejects stale writes (optimistic concurrency). |
| `edit_file` | Exact-string replacement: one `old_text`/`new_text` pair, or a `replacements` batch applied in order. Each `old_text` must match exactly once; a batch is checked in memory and written only if every replacement matches. Optional `expected_hash`. |
| `list_directory` | List a directory's entries with type indicators and sizes. |
| `search_files` | Regex search over file contents, optionally narrowed by an `include_glob`; skips files over 10 MiB or unreadable files and reports skipped paths. |
| `find_files` | Glob search over file names (e.g. `*.py`, `**/*.json`). The pattern is relative to `path`; absolute patterns are rejected. |
| `create_directory` | Create a directory and any missing parents. |
| `file_info` | Metadata for a file or directory: size, type, line count, hash, and symlink target, where the workspace provides them. |
| `list_files` | Opt-in, ripgrep-backed: files under a directory, recursively, sorted by path, with an optional `glob`. |
| `grep` | Opt-in, ripgrep-backed: content search with `glob`, `file_type`, `ignore_case`, `literal`, and `context` (0 to 20) options; a `path` may name a file or a directory. |

An explicitly unavailable run workspace fails at run start with its configured reason, rather than advice to attach a local workspace.

Missing paths (including directories passed to `read_file`) return `Path not found: <path>` as a tool result, so repeated lookups do not exhaust the model's tool retry budget. Invalid arguments still request a retry.

For remote workspaces, use the file tools (`grep`, `find_files`, `read_file`) rather than `cat` through a shell. Install `rg` and `git` in the sandbox image for fast searches; installing `rg` on the agent host does not install it in a remote workspace. For large or generated trees, use Shell with `rg -n 'pattern' path` and cap its output. Without `rg`, command-capable POSIX workspaces use one in-sandbox git/grep/find command for `grep`, `list_files`, and `search_files` (after an initial `rg` probe). Filesystem-only backends use slower, bounded file walks. The POSIX fallback honors nested `.gitignore` in repositories and search-root `.ignore` with git available; nested `.ignore` rules are not applied by the POSIX fallback, and rg-specific regex features require `rg`. Without git, the POSIX fallback cannot apply ignore files. With git, it also searches files git tracks even though `.gitignore` lists them, which `rg` skips. The first search that falls back in a workspace logs one debug message saying so. An explicitly named file is searched even when ignored. Searches report output and result caps rather than presenting partial results as complete. A file a search cannot read (with `rg`, a directory too) is skipped, and a note at the end of the result names it and the reason.

Recursive file walks visit each real directory once, so aliases to a directory do not duplicate its contents.

### Tool selection and the ripgrep tools

`tools` names the tools to register, from `FILE_SYSTEM_TOOL_NAMES`. The default,
`DEFAULT_TOOL_NAMES`, is the eight tools that need only the workspace's
filesystem. `list_files` and `grep` run the `rg` executable inside the
workspace when it is on its `PATH`, so they are opt-in by name. The
`coder` extra installs `rg` for a local workspace. Without `rg`, both use an in-workspace POSIX command. The fallback lacks ripgrep `file_type` support and some ignore-file rules. On a read-only or filesystem-only workspace, both walk the files instead, without ignore files or `file_type`.

```python
from pydantic_ai_harness import FileSystem

FileSystem(tools=['read_file', 'edit_file', 'list_files', 'grep'])
```

Both respect ripgrep's defaults: `.gitignore` inside a git repository and
`.ignore` files anywhere. As in ripgrep, an explicit `glob` takes precedence
over those ignore files. Hidden files can be selected by an explicit dotfile glob, and hidden directories by naming them as the search path. `list_directory` and walker-backed `find_files`/`search_files` report hidden entries they saw but omitted; pruned hidden directories count as one entry, not their unseen contents. Command-backed searches do not count hidden omissions. Output is sorted by path, so a capped
result is a deterministic prefix rather than a random subset. `grep` reports
matches as `path:line:text` and context lines as `path-line-text`, paths relative
to the working directory; a pattern uses ripgrep's regex syntax unless `literal` is set. A
pattern ripgrep rejects comes back to the model as a retry, so it can correct
the call. Every path
ripgrep prints goes through the same containment and pattern checks as the other
walkers before it is shown. Ripgrep output over 8 MiB is cut, and the search
is reported as truncated. `read_only=True` keeps only the tools in
`READ_ONLY_TOOL_NAMES` from whatever `tools` selects.

### Content hashes

`content_hashes=False` drops the hash from `read_file` headers and from
`write_file`/`edit_file` results, and removes the `expected_hash` parameter from
those two tools. The hashes give a model optimistic concurrency control over a
workspace that something else may also be editing; for a single-writer coding
agent they only add tokens to every read and write. Events still carry
`content_hash` either way.

### Working directory and root

Relative paths resolve from the workspace's working directory. Set `root_dir`
higher, such as a parent holding sibling projects, to let the model reach
beyond the project directory without spelling out absolute paths. `root_dir`
must contain the working directory. To work in a subdirectory, set it on the
workspace instead (`LocalWorkspace('./repo')`).

`list_directory`, `find_files`, `search_files`, `list_files`, and `grep` return
paths relative to the working directory, even when searching a subdirectory.
These paths can be passed directly to read/write tools. Files outside the
working directory but inside `root_dir` use `..` components. Containment,
access patterns, and event paths retain their `root_dir` basis, as does
`search_files`'s `include_glob` filter.

## Events

`FileSystem` emits typed capability events in the `file_system` namespace so a
host can show what the agent did to the workspace, or veto a change before it
lands, without parsing tool arguments:

| Event | Dispatch | Operation | Payload |
|---|---|---|---|
| `FileChangeRequestEvent` | immediate | `write_file`, `edit_file`, `create_directory` | `path`, `root_dir`, `operation`, `diff`, `truncated`; `cancel(reason)` |
| `FileReadEvent` | stream | `read_file` | `path`, `root_dir`, `content_hash` |
| `DirectoryListedEvent` | stream | `list_directory` | `path`, `root_dir`, `entry_count` |
| `FileWrittenEvent` | stream | `write_file` | `path`, `root_dir`, `content_hash` |
| `FileEditedEvent` | stream | `edit_file` | a `FileWrittenEvent` plus `diff`, `truncated` |
| `DirectoryCreatedEvent` | stream | `create_directory` | `path`, `root_dir` |
| `FilesSearchedEvent` | stream | `search_files`, `find_files`, `list_files`, `grep` | `path`, `root_dir`, `pattern`, `search` (`grep` or `find`), `match_count`, `truncated` |

`FileChangeRequestEvent` is a decision. It fires after the path has passed the
access checks and, for `write_file` and `edit_file`, after the conflict check,
so a listener only sees changes that would otherwise go ahead: a denied path,
a missing parent for `write_file`, a parent that is not a directory, a stale
`expected_hash` for a file that exists, or a directory that collides with a
file emits no request, so a listener cannot approve what the policy or the
filesystem refuses. A
listener may take a while (a human approving the diff, say), so once the
request returns the path is resolved and checked again, and a write or edit
re-reads the file just before writing and checks that it still holds what the
listener was shown: a file changed in the meantime fails after it was
announced instead of being overwritten, and an edit does not recreate a file
deleted in the meantime. This holds the window between the check and the write
(see [Security model](#security-model)) to what it is without a listener for
readable targets. For a target the workspace cannot read, an approved write has
no content guard; passing `expected_hash` instead refuses the write before
announcement. A listener that calls `cancel(reason)`
stops the change before it touches the disk, and the model gets the reason as the tool
result. A listener that raises instead aborts the run, as any raising event
listener does, and the change is not applied. `diff` is the unified diff from
the current content to the proposed content: a new file diffs from empty, a
file the workspace cannot read is announced with the file headers alone and
`truncated` set, since what it holds cannot be shown, and a `create_directory`
has no diff. A `create_directory` on a directory that already exists changes
nothing and emits nothing. The other events are notifications.

`FileEditedEvent` subclasses `FileWrittenEvent`, so a listener for writes
receives edits too and can read the `diff` when it has one. Before this
release `edit_file` emitted a plain `FileWrittenEvent`, so a serialized edit
had the kind `file_system.file_written`; it is now `file_system.file_edited`.
A listener registered for `FileWrittenEvent` still receives it; code that
matches on the serialized kind needs to accept both. Diffs are cut at
`MAX_EVENT_DIFF_CHARS` (8192) with a `truncated` flag, so a persisted or
forwarded event stream cannot be flooded by one large write, and a change
whose text is longer than `MAX_DIFF_SOURCE_CHARS` (32768) on either side is
not diffed at all: the `diff` is the two file headers and `truncated` is set.
The same fallback applies when `(old.count("\n") + 1) * (new.count("\n") + 1)`
exceeds 65536, bounding line-matching work before calling the differ.
A final line without a newline is marked the way `git diff` marks it, so a
change to the final newline alone is visible. A `FilesSearchedEvent` counts
the matches the model received; `truncated` says the search stopped at
`max_search_results` or `max_find_results`.

`path` is the normalized location relative to `root_dir`, never an absolute
path, so it is safe to echo to the model or a UI. `root_dir` is the emitting
filesystem's root as an absolute POSIX path inside the run's workspace, so a
subscriber rooted elsewhere can locate the file as
`posixpath.join(root_dir, path)` in that workspace instead of assuming it
shares the emitter's root.

Every event path has passed the containment check and the denied patterns. A
`DirectoryListedEvent` or `FilesSearchedEvent` names the walk root, which is
not gated by `allowed_patterns` (see [Security model](#security-model)); only
its entries are. A denied or failed operation emits no event, including a
`read_file` whose `offset` is past the end of the file.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import FileSystem
from pydantic_ai_harness.filesystem import FileChangeRequestEvent

agent = Agent('anthropic:claude-opus-5-5', capabilities=[LocalWorkspace('.'), FileSystem()])

@agent.on_event(FileChangeRequestEvent)
async def hold_migrations(ctx, event):
    if event.path.startswith('migrations/'):
        event.cancel('migrations need a human')
```

Other capabilities subscribe with `@on_event` on a method, the way
`RepoContext` follows `FileReadEvent` and `DirectoryListedEvent`. A host with
its own file tools can emit the same event types by importing them from
`pydantic_ai_harness.filesystem`, which lets subscribers react without
depending on tool names or raw model arguments.

`FileSystem` emits no OpenTelemetry spans of its own: the core tool-call span
already records each operation and its result, and the events above carry the
diff a trace would not.

Tool errors the model can correct -- a missing file, a denied path, a stale
edit, a directory that collides with an existing file, an invalid glob pattern,
a path name rejected by Windows, a path name the filesystem cannot encode, an
over-long path name, a symlink loop -- are surfaced as
[`ModelRetry`](../agent.md#reflection-and-self-correction),
so the agent gets the error message back and can adjust rather than aborting
the run. Failures the model can do nothing about, such as a full disk or a
workspace that is gone, still abort.

When an OS error supplies a filename, `FileSystem` reports it relative to
`root_dir`; paths outside `root_dir` become `<outside-workspace>`. `file_info`
applies the same rule to absolute symlink targets.

## Security model

- **Containment.** Every tool call resolves its path, symlinks included, and
  rejects one that leads outside `root_dir`, whether through `..`, an absolute
  path, or a symlink. Listings may name a link that leads outside, but reading or
  writing it is rejected, and the directory walkers don't descend into it.
  Patterns match the path relative to `root_dir`; `read_only_patterns` and
  `denied_patterns` also match a symlink's target, so a link to `.env` is
  read-only like `.env` itself. `root_dir='/'` turns containment off; the
  patterns still apply, and with no patterns set as well the boundary check is
  off entirely.
- **Symlinks need the workspace's `realpath`.** Local and command-running
  workspaces resolve symlinks for these checks. A filesystem-only backend without
  `SupportsRealpath` gives only the path text: `..` and absolute paths outside
  `root_dir` are still rejected, but a symlink leading outside is not detected,
  and a link to `.env` does not match the patterns. That is exact for storage
  without symlinks, such as an object store.
- **A guardrail, not isolation.** The checks run before each operation, so a
  symlink swapped in between the check and the use is not caught, and `Shell`
  commands ignore `root_dir` entirely. Use a sandbox workspace when the agent or
  the tree is untrusted.
- **Bounded walks.** `search_files` and `find_files` stop after 10,000
  directories or 100,000 entries and end their result with a
  `[... walk cut short ...]` line.
- **Binary detection.** `read_file` returns a placeholder instead of dumping
  binary bytes into the model context.
- **Optimistic concurrency.** `write_file`/`edit_file` accept an
  `expected_hash` so an agent operating on a stale read is told to re-read
  rather than silently overwriting newer content.
- **Write targets.** `write_file` won't overwrite a directory. With
  `LocalWorkspace`, reading or writing a FIFO blocks.

### Custom storage

To keep files somewhere else, write a `WorkspaceBackend` that implements
`SupportsFilesystem` (and `SupportsCommands`, for ripgrep searches and
`file_info` symlink targets) and attach it to the run. Containment, patterns,
events, and hashes apply unchanged. This replaces the removed
`FileSystemToolset.open_read` and `open_write` hooks.

To call a tool method yourself, outside a run, pass the workspace:

```python
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.filesystem import FileSystem, FileSystemToolset


async def main() -> None:
    toolset = FileSystem().get_toolset()
    assert isinstance(toolset, FileSystemToolset)
    print(await toolset.read_file('README.md', workspace=LocalWorkspaceBackend('.')))
```

## Pattern filtering

Three independent glob lists control access. Patterns are matched with
`fnmatch`, whose `*` spans `/`, so `*.py` matches `src/main.py` and you rarely
need `**`.

| Field | Effect |
|---|---|
| `allowed_patterns` | If non-empty, only matching paths are accessible (allowlist). |
| `denied_patterns` | Matching paths are rejected (denylist), even when `allowed_patterns` matches them. |
| `read_only_patterns` | Matching paths are read-only: reads succeed, writes are rejected. |

A directory pattern also applies to descendants: `denied_patterns=['private']` denies `private/notes.txt` as well as `private`. Read-only directory patterns similarly protect writes below them.

`read_only_patterns` defaults to `**/.git/*`, `**/.env`, `**/.env.*` (at any depth), `*.pem`, `*.key`,
and `**/secrets*`, and `**/.pydantic-ai-harness/**`, where harness capabilities keep
their own files (spilled tool output, background job status). Pass an empty list to make every path writable.
`protected_patterns` is its deprecated name and still works, with a warning.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import FileSystem

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[
        LocalWorkspace('.'),
        FileSystem(
            allowed_patterns=['*.py', '*.toml'],
            denied_patterns=['**/node_modules/*'],
        ),
    ],
)
```

### Direct access vs. walkers

The three rules apply at two different granularities:

- **Direct access** (`read_file`, `write_file`, `edit_file`, `file_info`,
  `create_directory`) gates the operation's target path. You must name a path
  that the patterns permit.
- **Walkers** (`list_directory`, `search_files`, `find_files`, `list_files`, `grep`) gate their root
  by denied patterns, but **not** by `allowed_patterns` -- a directory root
  like `.` never matches a file pattern such as `src/*.py`, so requiring it to
  would make every listing fail. Instead, the root is walked and each
  **entry** is filtered with read-level access against `allowed_patterns` and
  `denied_patterns`. A directory listing cannot surface a path the agent
  couldn't otherwise read.

So with `allowed_patterns=['*.py']`, `list_directory('.')` succeeds and shows
only the `.py` entries; `read_file('notes.md')` is rejected.

Matching `read_only_patterns` alone does not hide an entry. Read-only paths
that pass the allowed, denied, and dotfile filters remain visible to the
walkers and directly readable via `read_file`/`file_info`; write operations
reject them.

!!! note
    Dotfiles and dot-directories (`.git`, `.env`, `.github`, ...) are skipped by
    every walker -- `list_directory`, `search_files`, `find_files`, `list_files`, and `grep` --
    regardless of patterns.

## Configuration

```python {names="defined"}
from pydantic_ai_harness.filesystem import DEFAULT_TOOL_NAMES, FileSystem

FileSystem(
    root_dir=None,                 # str | Path -- containment boundary (None = the working directory; '/' = no checks)
    allowed_patterns=[],           # allowlist globs (empty = allow all)
    denied_patterns=[],            # denylist globs
    read_only_patterns=[...],      # read-only globs (defaults to secrets/.git)
    max_read_lines=2000,           # cap for a single read_file
    max_read_chars=50_000,         # cap on a whole read_file result, ending on a complete line
    max_list_results=1000,         # cap for list_directory
    max_search_results=1000,       # cap for search_files and grep
    max_find_results=1000,         # cap for find_files and list_files
    read_only=False,               # keep only READ_ONLY_TOOL_NAMES
    content_hashes=True,           # report hashes and accept expected_hash
    tools=DEFAULT_TOOL_NAMES,      # which tools to register (add 'list_files', 'grep')
    max_retries=None,              # consecutive retries per tool before the run fails (None = the agent's budget)
)
```

The integer limits must be positive; they are validated at construction and
raise `ValueError` otherwise. A walker that hits its cap ends its output with a
`[... truncated at N ...]` marker, and only when a further entry was actually
dropped.

## Agent spec (YAML/JSON)

`FileSystem` works with Pydantic AI's
[agent spec](../agent-spec.md):

```yaml
model: anthropic:claude-opus-5-5
capabilities:
  - FileSystem:
      allowed_patterns: ['*.py', '*.toml']
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import FileSystem

agent = Agent.from_file('agent.yaml', custom_capability_types=[FileSystem])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate
`FileSystem`, and attach a workspace to the run (`workspace=` on the run method,
or a workspace capability in Python).

## Durable execution

`FileSystem` works under Temporal, DBOS, and Prefect. [Durable execution](durable-execution.md) shows an example for each engine and which file events Temporal delivers live.

## Further reading

- [Pydantic AI capabilities](../capabilities/overview.md)
- [Toolsets](../toolsets.md)
- [the capabilities overview](index.md)

## API reference

::: pydantic_ai_harness.FileSystem
