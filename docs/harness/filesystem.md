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
    '{{ anthropic:large }}',
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
| `read_file` | Numbered lines and a content hash, paged by `offset`/`limit`, at most `max_read_lines` per call. Images (PNG, JPEG, GIF, WebP) and PDFs up to 5 MB are returned as files; other binary files as size, media type, and a 64-byte hex preview. A file over 64 MiB is described, not read. With `max_read_chars`, the result ends on the last complete line that fits and the continuation hint names the first line not shown. |
| `write_file` | Create or overwrite a file. Optional `expected_hash` rejects stale writes (optimistic concurrency). |
| `edit_file` | Exact-string replacement: one `old_text`/`new_text` pair, or a `replacements` batch applied in order. Each `old_text` must match exactly once; a batch is checked in memory and written only if every replacement matches. Optional `expected_hash`. |
| `list_directory` | List a directory's entries with type indicators and sizes. |
| `search_files` | Regex search over file contents, optionally narrowed by an `include_glob` matched with `fnmatch` against root-relative paths; skips files over 10 MiB or unreadable files and reports skipped paths. |
| `find_files` | Glob search over file names (e.g. `*.py`, `**/*.json`). The pattern is relative to `path`; absolute patterns are rejected. |
| `create_directory` | Create a directory and any missing parents. |
| `file_info` | Metadata for a file or directory: size, type, whether a file is binary, a text file's line count and hash, and symlink target, where the workspace provides them. A file over 64 MiB reports its size only. |
| `list_files` | Opt-in, ripgrep-backed: files under a directory, recursively, sorted by path, with an optional `glob`. |
| `grep` | Opt-in, ripgrep-backed: content search with `glob`, `file_type`, `ignore_case`, `literal`, and `context` (0 to 20) options; a `path` may name a file or a directory. |

An explicitly unavailable run workspace fails at run start with its configured reason, rather than advice to attach a local workspace.

Missing paths (including directories passed to `read_file`) return `Path not found: <path>` as a tool result, so repeated lookups do not exhaust the model's tool retry budget. Invalid arguments still request a retry.

For remote workspaces, use the file tools (`grep`, `find_files`, `read_file`) rather than `cat` through a shell. How a search runs depends on the workspace:

- In a remote workspace, install `rg` and `git` in the image for fast searches. The agent host's `rg` is not used there.
- Without `rg`, a command-capable POSIX workspace runs one git/grep/find command for `grep`, `list_files`, and `search_files`, after an initial `rg` probe, and logs one debug message the first time. With git it reads nested `.gitignore` files but only the search root's `.ignore`, and also searches files git tracks even though `.gitignore` lists them, which `rg` skips. Without git it applies no ignore files. Regex features specific to `rg` need `rg`.
- A filesystem-only or read-only workspace walks its files in Python, bounded. Python's regex engine has no time limit, so a regex that repeats a repeating group, such as `(a+)+`, is refused: that shape can take exponential time. The check is a heuristic for that shape, not a time limit.
- An explicitly named file is always searched, even when ignored. A file a search cannot read (with `rg`, a directory too) is skipped, and a note at the end of the result names it and the reason. Output and result caps are reported, so a partial result never reads as complete.

For large or generated trees, use Shell with `rg -n 'pattern' path` and cap its output.

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

### Tool kinds

Each tool declares a [tool kind](../tools-advanced.md#typed-tool-parts) in the `file_system`
namespace, named after the tool: `read_file` has the kind
`file_system.read_file`, exported as `READ_FILE_KIND`, and likewise
`WRITE_FILE_KIND`, `EDIT_FILE_KIND`, `LIST_DIRECTORY_KIND`, `SEARCH_FILES_KIND`,
`FIND_FILES_KIND`, `CREATE_DIRECTORY_KIND`, `FILE_INFO_KIND`, `LIST_FILES_KIND`,
and `GREP_KIND`. The kind is part of the tool's definition rather than its name,
so another capability or a history processor can recognize these tools by kind
instead of comparing tool names. Each kind declares the shape of its tool's
arguments (`ReadFileArgs`, `WriteFileArgs`, and so on), and a call whose
arguments fit it is promoted to a typed part (`ReadFileCallPart`,
`WriteFileCallPart`, and so on) in a run's messages and in a loaded history,
with the validated arguments as `typed_args`:

```python
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.filesystem import ReadFileCallPart


def paths_read(messages: list[ModelMessage]) -> list[str]:
    return [
        args['path']
        for message in messages
        for part in message.parts
        if isinstance(part, ReadFileCallPart) and (args := part.typed_args) is not None
    ]
```

`read_file` also has a typed return part, `ReadFileReturnPart`, for a return
whose outcome is `'success'`, which includes a `Path not found: <path>` result.
Its `content` is the text with its header, or `[header, BinaryContent]` for an
image or PDF. The kind declares no content shape, since `ToolKind` takes a class
and not a union, so `typed_content` is untyped. The other tools return text
written for the model rather than a format to parse, so their returns stay plain
`ToolReturnPart`s that carry the kind as `tool_kind`; the [events](#events)
carry their outcomes as structured data.

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

`FileChangeRequestEvent` is a decision; the other events are notifications. It
fires only for a change that passed the access and conflict checks, so a
listener cannot approve what the policy or the filesystem refuses. Call
`event.cancel(reason)` to refuse the change; the model gets `reason` as the
tool result. A listener that raises aborts the run, and the change is not
applied.

A listener may take a while (a human approving the diff, say), so after it
returns the path is checked again and the file re-read: a file that changed
meanwhile fails instead of being overwritten, and an edit does not recreate a
file deleted meanwhile. A target the workspace cannot read has no content to
guard; pass `expected_hash` to refuse such a write before it is announced.

`diff` is the unified diff from the current content to the proposed content. A
new file diffs from empty; an unreadable file is announced with the file
headers alone and `truncated` set; `create_directory` has no diff, and on a
directory that already exists it emits nothing.

`FileEditedEvent` subclasses `FileWrittenEvent`, so a listener for writes
receives edits too; its serialized kind is `file_system.file_edited`. Diffs are cut at
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

agent = Agent('{{ anthropic:large }}', capabilities=[LocalWorkspace('.'), FileSystem()])

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
over-long path name, a symlink loop, a FIFO or other special file -- are surfaced as
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
  read-only like `.env` itself. Denied and read-only patterns ignore case, so
  `.GIT/config` cannot get past `**/.git` on a case-insensitive filesystem.
  `root_dir='/'` turns containment off; the
  patterns still apply, and with no patterns set as well the boundary check is
  off entirely.
- **Symlinks need the workspace's `realpath`.** Local and command-running
  workspaces resolve symlinks for these checks. A filesystem-only backend without
  `SupportsRealpath` gives only the path text: `..` and absolute paths outside
  `root_dir` are still rejected, but a symlink leading outside is not detected,
  and a link to `.env` does not match the patterns. That is exact for storage
  without symlinks, such as an object store.
- **A guardrail, not isolation.** Containment and allow, deny, and read-only
  patterns are functional guardrails. `LocalWorkspace` does not establish a
  security isolation boundary, even with command execution disabled. Bypasses
  within the workspace process's existing OS authority are correctness bugs.
  The checks run before each operation, so a symlink swapped between the check
  and the use is not caught, and `Shell` commands ignore `root_dir` entirely.
  For isolation, use a sandbox workspace and configure its boundary to protect
  the required resources. Files inside that boundary are not automatically
  confidential from the agent.
- **Bounded walks.** `search_files` and `find_files` stop after 10,000
  directories or 100,000 entries and end their result with a
  `[... walk cut short ...]` line.
- **Binary detection.** `read_file` does not decode binary bytes into the model
  context. It returns an image (PNG, JPEG, GIF, WebP) or PDF as `BinaryContent`,
  recognized by its leading bytes rather than its extension, which Pydantic AI
  sends [inside the tool result or on the user channel](../tools-advanced.md#tool-return-file-provenance)
  depending on the model. If the model can't take that kind of file (an image on
  a model without vision, or a PDF on Groq), Pydantic AI raises when it builds
  the next request, which ends the run
  ([input support by model](../input.md)). A file over 5 MB, the strictest
  mainstream per-image limit, is described instead, as is any other binary file.
- **Optimistic concurrency.** `write_file`/`edit_file` accept an
  `expected_hash` so an agent operating on a stale read is told to re-read
  rather than silently overwriting newer content.
- **Write targets.** `write_file` won't overwrite a directory. A FIFO, socket,
  or device is refused as not a regular file.
- **Bounded reads.** `read_file` and `file_info` read a file whole, for its line
  count and hash, so a file over 64 MiB is described by its size instead.

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
| `denied_patterns` | Matching paths are rejected (denylist), even when `allowed_patterns` matches them. Matching ignores case. |
| `read_only_patterns` | Matching paths are read-only: reads succeed, writes are rejected. Matching ignores case. |

A directory pattern also applies to descendants: `denied_patterns=['private']` denies `private/notes.txt` as well as `private`. Read-only directory patterns similarly protect writes below them.

`read_only_patterns` defaults to `**/.git`, `**/.env`, `**/.env.*`, `*.pem`, `*.key`,
`**/secrets*`, and `**/.pydantic-ai-harness/**`, which match at any depth. For example,
`apps/api/.env` and `sub/.git/config` are protected along with root-level files, and so is a
`.git` file, which a git worktree or submodule has in place of the directory. Protected paths stay
readable; add a path to `denied_patterns` to make it unreadable. Harness capabilities keep their own files,
such as spilled tool output and background job status, under `.pydantic-ai-harness/`. Pass an empty list to
make every path writable.
`protected_patterns` is its deprecated name and still works, with a warning.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import FileSystem

agent = Agent(
    '{{ anthropic:large }}',
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
    Walkers (`list_directory`, `search_files`, `find_files`, `list_files`, and `grep`) skip
    dotfiles and dot-directories unless the glob names one (`.github/*`) or the search path is one.

## Configuration

```python {names="defined"}
from pydantic_ai_harness.filesystem import DEFAULT_TOOL_NAMES, FileSystem

FileSystem(
    root_dir=None,                 # str | Path -- containment boundary (None = the working directory; '/' = no checks)
    allowed_patterns=[],           # allowlist globs (empty = allow all)
    denied_patterns=[],            # denylist globs
    read_only_patterns=[...],      # read-only globs (defaults to secrets/.git)
    max_read_lines=2000,           # most lines one read_file returns, and its default limit
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
model: {{ anthropic:large }}
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

`FileSystem` works under Temporal, DBOS, and Prefect; the [workspace guide](../workspace.md#durable-execution) shows what works on each engine. On Temporal only file-change requests and write events reach listeners live; see [live events on Temporal](../durable_execution/temporal.md#harness-workspace-capabilities).

## Further reading

- [Pydantic AI capabilities](../capabilities/overview.md)
- [Toolsets](../toolsets.md)
- [the capabilities overview](index.md)

## API reference

::: pydantic_ai_harness.FileSystem
