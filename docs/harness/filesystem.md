---
title: FileSystem
description: Give a Pydantic AI agent sandboxed, glob-filtered file access scoped to a single directory tree, with symlink-safe containment checks.
---

# FileSystem

`FileSystem` gives an agent a fixed set of file tools -- read, write, edit, list,
search, find, create, and inspect -- all scoped to a single `root_dir`. Every path is
resolved and containment-checked (symlinks included) before any I/O, and access
is filtered through allow / deny / protected glob patterns.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/filesystem/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

Letting an agent touch the filesystem directly is risky: path traversal
(`../../etc/passwd`), symlinks that escape the project, clobbering `.git`, or
leaking `.env` secrets. Hand-rolling the guards around every tool call is
repetitive and easy to get subtly wrong.

`FileSystem` centralizes those guards. It exposes one bounded, sandboxed
toolset so you configure the boundary once and reuse it across agents.

## Usage

Add `FileSystem` to your agent's `capabilities` with a `root_dir`. Everything
the agent reads or writes is confined to that directory.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import FileSystem

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[FileSystem(root_dir='./workspace')],
)

result = agent.run_sync('Read config.toml and tell me the package name.')
print(result.output)
```

`root_dir` defaults to the current directory (`.`), but passing an explicit
workspace path is the recommended practice -- the sandbox is only as tight as
the root you give it.

## Tools

`FileSystem` contributes eight tools by default, plus two opt-in ripgrep tools, all path-scoped to `root_dir`:

| Tool | Purpose |
|---|---|
| `read_file` | Read a text file with line numbers and a content hash. Binary files are detected and not dumped. Supports `offset`/`limit` paging; with `max_read_chars`, the whole result (header and hint included) fits the cap, the window ends on the last complete line that fits, and the continuation hint names the first line not shown. |
| `write_file` | Create or overwrite a file. Optional `expected_hash` rejects stale writes (optimistic concurrency). |
| `edit_file` | Exact-string replacement: one `old_text`/`new_text` pair, or a `replacements` batch applied in order. Each `old_text` must match exactly once; a batch is checked in memory and written only if every replacement matches. Optional `expected_hash`. |
| `list_directory` | List a directory's entries with type indicators and sizes. |
| `search_files` | Regex search over file contents, optionally narrowed by an `include_glob`. |
| `find_files` | Glob search over file names (e.g. `*.py`, `**/*.json`). The pattern is relative to `path`; absolute patterns are rejected. |
| `create_directory` | Create a directory and any missing parents. |
| `file_info` | Metadata for a file or directory (size, type, line count, hash, symlink target). |
| `list_files` | Opt-in, ripgrep-backed: files under a directory, recursively, sorted by path, with an optional `glob`. |
| `grep` | Opt-in, ripgrep-backed: content search with `glob`, `file_type`, `ignore_case`, `literal`, and `context` (0 to 20) options; a `path` may name a file or a directory. |

### Tool selection and the ripgrep tools

`tools` names the tools to register, from `FILE_SYSTEM_TOOL_NAMES`. The default,
`DEFAULT_TOOL_NAMES`, is the eight pure-Python tools. `list_files` and `grep` run
the `rg` executable, which must be on `PATH` (the `coder` extra installs it), so
they are opt-in by name:

```python
from pydantic_ai_harness import FileSystem

FileSystem(root_dir='./workspace', tools=['read_file', 'edit_file', 'list_files', 'grep'])
```

Both respect ripgrep's defaults: `.gitignore` inside a git repository and
`.ignore` files anywhere. As in ripgrep, an explicit `glob` takes precedence
over those ignore files; unlike ripgrep, dotfiles and dot-directories stay
hidden even then, as with the other walkers. Output is sorted by path, so a capped
result is a deterministic prefix rather than a random subset. `grep` reports
matches as `path:line:text` and context lines as `path-line-text`, paths relative
to `cwd`; a pattern uses ripgrep's regex syntax unless `literal` is set. A
missing `rg` or a pattern ripgrep rejects comes back to the model as a retry, so
it can correct the call or use `search_files`/`find_files` instead. Every path
ripgrep prints goes through the same containment and pattern checks as the other
walkers before it is shown. `read_only=True` keeps only the tools in
`READ_ONLY_TOOL_NAMES` from whatever `tools` selects.

### Content hashes

`content_hashes=False` drops the hash from `read_file` headers and from
`write_file`/`edit_file` results, and removes the `expected_hash` parameter from
those two tools. The hashes give a model optimistic concurrency control over a
workspace that something else may also be editing; for a single-writer coding
agent they only add tokens to every read and write. Events still carry
`content_hash` either way.

### Working directory

`cwd` is the directory relative paths resolve from; it defaults to `root_dir`
and must lie inside it. Set it to hand the model a project directory while
`root_dir` grants access to more, such as a parent directory or the filesystem
root, without the model spelling out absolute paths.

`list_directory`, `find_files`, `search_files`, `list_files`, and `grep` return
paths relative to `cwd`, even when searching a subdirectory. These paths can be
passed directly to read/write tools. Files outside `cwd` but inside `root_dir`
use `..` components. Containment, access patterns, and event paths retain their
`root_dir` basis, as does `search_files`'s `include_glob` filter.

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
checks under its open descriptor that the file still holds what the listener
was shown: a path or file replaced in the meantime fails after it was
announced instead of being redirected or overwritten, and an edit does not
recreate a file deleted in the meantime. This holds the window between the
containment check and the I/O (see [Security model](#security-model)) to what
it is without a listener for readable targets. For a target the process cannot
read, an approved write has no content guard; passing `expected_hash` instead
refuses the write before announcement. A listener that calls `cancel(reason)`
stops the change before it touches the disk, and the model gets the reason as the tool
result. A listener that raises instead aborts the run, as any raising event
listener does, and the change is not applied. `diff` is the unified diff from
the current content to the proposed content: a new file diffs from empty, a
file the process cannot read is announced with the file headers alone and
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

`path` is the normalized, symlink-resolved location relative to `root_dir`,
never an absolute host path, so it is safe to echo to the model or a UI.
`root_dir` is the emitting filesystem's resolved root, so a subscriber rooted
elsewhere can locate the file as `Path(root_dir) / path` instead of assuming
it shares the emitter's root.

Every event path has passed the containment check and the denied patterns. A
`DirectoryListedEvent` or `FilesSearchedEvent` names the walk root, which is
not gated by `allowed_patterns` (see [Security model](#security-model)); only
its entries are. A denied or failed operation emits no event, including a
`read_file` whose `offset` is past the end of the file.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import FileSystem
from pydantic_ai_harness.filesystem import FileChangeRequestEvent

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[FileSystem()])

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
[`ModelRetry`](/ai/core-concepts/agent/#reflection-and-self-correction),
so the agent gets the error message back and can adjust rather than aborting
the run. Failures the model can do nothing about, such as a full or read-only
disk, still abort.

When an OS error supplies a filename, `FileSystem` reports it relative to
`root_dir`; paths outside `root_dir` become `<outside-workspace>`. `file_info`
applies the same rule to absolute symlink targets.

## Security model

- **Containment.** Relative paths resolve from `cwd`; anything resolving
  outside `root_dir` -- via `..`, an absolute path, or a symlink -- is rejected. Symlinks
  are resolved with `os.path.realpath` *before* the containment check, and I/O
  then uses the resolved path. Directory walks (`list_directory`,
  `search_files`, `find_files`, `list_files`, `grep`) resolve each entry the same way and match the
  patterns against that resolved target, so a symlink cannot name a file
  outside the tree or present a denied file under a permitted name. These
  checks are pathname-based: if another process mutates the tree between
  resolution and I/O, the path read can differ from the path checked.
- **Binary detection.** `read_file` returns a placeholder instead of dumping
  binary bytes into the model context.
- **Optimistic concurrency.** `write_file`/`edit_file` accept an
  `expected_hash` so an agent operating on a stale read is told to re-read
  rather than silently overwriting newer content.
- **Regular write targets.** `write_file` rejects an existing target that is
  not a regular file. On POSIX, it opens the final target descriptor in
  non-blocking mode and checks that descriptor's type before truncating, so a
  FIFO at the final component cannot stall the tool even if it is swapped into
  place during the write.

## Pattern filtering

Three independent glob lists control access. Patterns are matched with
`fnmatch`, whose `*` spans `/`, so `*.py` matches `src/main.py` and you rarely
need `**`.

| Field | Effect |
|---|---|
| `allowed_patterns` | If non-empty, only matching paths are accessible (allowlist). |
| `denied_patterns` | Matching paths are always rejected (denylist). |
| `protected_patterns` | Matching paths are read-only -- reads succeed, writes are rejected. |

`protected_patterns` defaults to `.git/*`, `.env`, `.env.*`, `*.pem`, `*.key`,
and `**/secrets*`. Pass an empty list to disable protection.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import FileSystem

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        FileSystem(
            root_dir='./workspace',
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

Matching `protected_patterns` alone does not hide an entry. Protected paths
that pass the allowed, denied, and dotfile filters remain visible to the
walkers and directly readable via `read_file`/`file_info`; write operations
reject them.

!!! note
    Dotfiles and dot-directories (`.git`, `.env`, `.github`, ...) are skipped by
    every walker -- `list_directory`, `search_files`, `find_files`, `list_files`, and `grep` --
    regardless of patterns.

## Configuration

```python
from pydantic_ai_harness import FileSystem

FileSystem(
    root_dir='.',                  # str | Path -- sandbox root
    cwd=None,                      # where relative paths resolve from (defaults to root_dir)
    allowed_patterns=[],           # allowlist globs (empty = allow all)
    denied_patterns=[],            # denylist globs
    protected_patterns=[...],      # read-only globs (defaults to secrets/.git)
    max_read_lines=2000,           # cap for a single read_file
    max_read_chars=None,           # optional cap on a whole read_file result, ending on a complete line
    max_list_results=1000,         # cap for list_directory
    max_search_results=1000,       # cap for search_files and grep
    max_find_results=1000,         # cap for find_files and list_files
    read_only=False,               # keep only READ_ONLY_TOOL_NAMES
    content_hashes=True,           # report hashes and accept expected_hash
    tools=DEFAULT_TOOL_NAMES,      # which tools to register (add 'list_files', 'grep')
)
```

The integer limits must be positive; they are validated at construction and
raise `ValueError` otherwise. A walker that hits its cap ends its output with a
`[... truncated at N ...]` marker, and only when a further entry was actually
dropped.

## Agent spec (YAML/JSON)

`FileSystem` works with Pydantic AI's
[agent spec](/ai/core-concepts/agent-spec/):

```yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - FileSystem:
      root_dir: ./workspace
      allowed_patterns: ['*.py', '*.toml']
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import FileSystem

agent = Agent.from_file('agent.yaml', custom_capability_types=[FileSystem])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate
`FileSystem`.

## Further reading

- [Pydantic AI capabilities](/ai/capabilities/overview/)
- [Toolsets](/ai/tools-toolsets/toolsets/)
- [the capabilities overview](index.md)

## API reference

::: pydantic_ai_harness.FileSystem
