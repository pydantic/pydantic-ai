# Context

> [!NOTE]
> Import this capability from its submodule -- there is no top-level `pydantic_ai_harness` re-export:
>
> ```python
> from pydantic_ai_harness.context import RepoContext
> ```
>
> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

Discover and load a repo's accumulated coding-assistant context engineering (CE).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/context/)

## The problem

A repo accumulates CE for whatever coding assistant worked in it: instruction
files (`CLAUDE.md`/`AGENTS.md`) scattered across the tree, and assets under
`.claude`/`.agents`/`.codex`/`.grok` (skills, sub-agents, hooks). An agent that
loads only the top-level instruction file misses the ancestor context and has no
idea the rest of the setup exists, so it can neither honor it nor translate it.

## The solution

`RepoContext` bundles three strategies, each independently toggleable.

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness.context import RepoContext

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[RepoContext(workspace_dir=Path('.'), home_dir=Path.home())],
)
```

### 1. Walk-up instruction autoload (on by default)

Loads `CLAUDE.md`/`AGENTS.md` from `workspace_dir` and every ancestor up to
`home_dir` (inclusive). Precedence is ancestor-first, workspace-last: broadest
context first, most specific last. Files are deduped by resolved real path and by
content hash, so a symlinked `AGENTS.md -> CLAUDE.md` or two ancestors sharing
identical content load once.

When `home_dir` is `None` (the default), only `workspace_dir` is scanned -- no
walk-up. Pass `home_dir=Path.home()` to walk up to your home directory.

### 2. Asset inventory (on by default)

Exposes one tool, `inventory_agent_context()`, that reports where the repo's CE
assets live -- the `.claude`/`.agents`/`.codex`/`.grok` roots and, within each,
the `skills/` (SKILL.md), `agents/` (`.md`), and `settings.json` (hooks) it
contains. It returns a structured `AgentContextInventory`; it locates assets and
does not parse them, leaving translation to the orchestrator.

Rename the tool with `inventory_tool_name`, or scope which roots it scans with
`asset_roots`.

### 3. Nested-on-traversal (off by default)

When the model lists or reads a directory, surface that directory's
`CLAUDE.md`/`AGENTS.md`. This couples to the host's list/read tools, so it is
opt-in and configurable:

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness import FileSystem
from pydantic_ai_harness.context import RepoContext

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        FileSystem(root_dir='.'),
        RepoContext(
            workspace_dir=Path('.'),
            nested_traversal=True,
            traversal_tool_names=frozenset({'list_directory', 'read_file'}),  # the FileSystem tool names to hook
            traversal_path_arg='path',                                   # the path arg key
            nested_inject='pointer',                                     # or 'contents'
        )
    ],
)
```

`nested_inject='pointer'` (default) appends a one-line note pointing at the file;
`'contents'` inlines the file body. Each directory is surfaced at most once per
run.

## Cache cost

Injecting file contents into the system prompt costs prompt-cache stability: a
changed prefix re-bills the whole cached region. `RepoContext` keeps the two
cache-relevant paths separate:

- Strategy 1 reads its files **once at run start** and injects them as static
  system instructions, so the cached prefix stays byte-identical across turns.
- Strategy 3 is volatile (it depends on which directory was just touched), so its
  note is appended to the **tool result** in the message tail -- never to the
  system prompt -- and cannot invalidate the cached prefix.

## Configuration

```python
RepoContext(
    workspace_dir,                  # Path -- the deepest dir the agent works in (required)
    home_dir=None,                  # Path | None -- shallowest dir to stop walk-up at, inclusive
    filenames=('CLAUDE.md', 'AGENTS.md'),
    autoload_instructions=True,     # Strategy 1
    expose_inventory_tool=True,     # Strategy 2
    inventory_tool_name='inventory_agent_context',
    nested_traversal=False,         # Strategy 3
    nested_inject='pointer',        # 'pointer' | 'contents'
    traversal_tool_names=frozenset({'list_directory', 'read_file'}),
    traversal_path_arg='path',
    asset_roots=('.claude', '.agents', '.codex', '.grok'),
)
```

## Scope

`RepoContext` locates and loads CE; it does not parse skill/sub-agent frontmatter
or hook bodies, and it does not rewrite or translate assets. Strategy 1 reads its
files once per run, so mid-run edits to those files are not reloaded.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Pydantic AI hooks](https://ai.pydantic.dev/hooks/)
