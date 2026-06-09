# FileSystem

Give an agent sandboxed, pattern-filtered access to a directory tree.

## The problem

Letting an agent touch the filesystem directly is risky: path traversal
(`../../etc/passwd`), symlinks that escape the project, clobbering `.git`, or
leaking `.env` secrets. Hand-rolling the guards around every tool call is
repetitive and easy to get subtly wrong.

## The solution

`FileSystem` exposes a fixed set of file tools, all scoped to a single
`root_dir`. Every path is resolved and containment-checked (symlinks included)
before any I/O, and access is filtered through allow / deny / protected glob
patterns.

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

## Tools

| Tool | Purpose |
|---|---|
| `read_file` | Read a text file with line numbers and a content hash. Binary files are detected and not dumped. |
| `write_file` | Create or overwrite a file. Optional `expected_hash` rejects stale writes (optimistic concurrency). |
| `edit_file` | Exact-string replacement; `old_text` must match exactly once. Optional `expected_hash`. |
| `list_directory` | List a directory's entries with type indicators and sizes. |
| `search_files` | Regex search over file contents, optionally narrowed by an `include_glob`. |
| `find_files` | Glob search over file names (e.g. `*.py`, `**/*.json`). |
| `create_directory` | Create a directory and any missing parents. |
| `file_info` | Metadata for a file or directory (size, type, line count, hash, symlink target). |

## Security model

- **Containment.** Paths resolve relative to `root_dir`; anything resolving
  outside -- via `..`, an absolute path, or a symlink -- is rejected. Symlinks
  are resolved with `os.path.realpath` *before* the containment check, closing
  the TOCTTOU window.
- **Binary detection.** `read_file` returns a placeholder instead of dumping
  binary bytes into the model context.
- **Optimistic concurrency.** `write_file`/`edit_file` accept an
  `expected_hash` so an agent operating on a stale read is told to re-read
  rather than silently overwriting newer content.

## Pattern filtering

Three independent glob lists control access. Patterns are matched with
`fnmatch`, whose `*` spans `/`, so `*.py` matches `src/main.py` and you rarely
need `**`.

| Field | Effect |
|---|---|
| `allowed_patterns` | If non-empty, only matching paths are accessible (allowlist). |
| `denied_patterns` | Matching paths are always rejected (denylist). |
| `protected_patterns` | Matching paths are read-only -- reads succeed, writes are rejected. |

`protected_patterns` defaults to `.git/`, `.env`/`.env.*`, `*.pem`, `*.key`,
and `**/secrets*`. Pass an empty list to disable protection.

### Direct access vs. walkers

The three rules apply at two different granularities:

- **Direct access** (`read_file`, `write_file`, `edit_file`, `file_info`,
  `create_directory`) gates the operation's target path. You must name a path
  that the patterns permit.
- **Walkers** (`list_directory`, `search_files`, `find_files`) gate their root
  by deny/protected patterns, but **not** by `allowed_patterns` -- a directory
  root like `.` never matches a file pattern such as `src/*.py`, so requiring
  it to would make every listing fail. Instead, the root is always walked and
  each **entry** is filtered against all three lists. A directory listing can
  never surface a path the agent couldn't otherwise read or write.

So with `allowed_patterns=['*.py']`, `list_directory('.')` succeeds and shows
only the `.py` entries; `read_file('notes.md')` is rejected.

> Dotfiles and dot-directories (`.git`, `.env`, `.github`, …) are skipped by
> all three walkers -- `list_directory`, `search_files`, and `find_files` --
> regardless of patterns.

## Configuration

```python
FileSystem(
    root_dir='.',                  # str | Path -- sandbox root
    allowed_patterns=[],           # allowlist globs (empty = allow all)
    denied_patterns=[],            # denylist globs
    protected_patterns=[...],      # read-only globs (defaults to secrets/.git)
    max_read_lines=2000,           # cap for a single read_file
    max_search_results=1000,       # cap for search_files
    max_find_results=1000,         # cap for find_files
)
```

The integer limits must be positive; they are validated at construction.

## Agent spec (YAML/JSON)

`FileSystem` works with Pydantic AI's
[agent spec](https://ai.pydantic.dev/agent-spec/):

```yaml
# agent.yaml
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

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Toolsets](https://ai.pydantic.dev/toolsets/)
