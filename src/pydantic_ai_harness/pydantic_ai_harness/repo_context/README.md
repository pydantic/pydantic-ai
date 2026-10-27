# Repo Context

Discover and load a repo's accumulated coding-assistant context engineering (CE).

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/repo_context/)

## The problem

A repo accumulates CE for whatever coding assistant worked in it: instruction
files (`CLAUDE.md`/`AGENTS.md`) scattered across the tree, and assets under
`.claude`/`.agents`/`.codex`/`.grok` (skills, sub-agents, hooks). An agent that
loads only the top-level instruction file misses the ancestor context and has no
idea the rest of the setup exists, so it cannot honor all of it.

## The solution

Autoloaded instruction symlinks resolving outside the directory being scanned are skipped, including shared dotfiles in another directory.

`RepoContext` bundles three strategies, each independently toggleable. Construct it with `RepoContext(...)` in an `Agent`'s `capabilities`, with a workspace attached to the run:

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import RepoContext

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[LocalWorkspace('.'), RepoContext(home_dir=Path.home())],
)

result = agent.run_sync('Summarize the coding-assistant setup in this repo.')
print(result.output)
```

`RepoContext` reads from the run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/), starting at its working directory. To point it at a subdirectory, set it on the workspace: `LocalWorkspace('./repo')`. `home_dir` and `asset_roots` are paths in the workspace: relative ones resolve from the working directory, and `~` is not expanded. A run without a workspace fails at its start.

### 1. Walk-up instruction autoload (on by default)

Loads `CLAUDE.md`/`AGENTS.md` from the working directory and every ancestor up to
`home_dir` (inclusive). Precedence is ancestor-first, workspace-last: broadest
context first, most specific last. Files are deduped by the path a symlink resolves
to and by content hash, so a symlinked `CLAUDE.md -> AGENTS.md` or two ancestors
sharing identical content load once. Where the workspace can resolve symlinks, a
symlinked file is labeled with its target's name, so `CLAUDE.md -> AGENTS.md`
reaches the model as `AGENTS.md`.

When `home_dir` is `None` (the default), only the working directory is scanned -- no
walk-up. Pass the workspace home path explicitly to walk up to it. For a remote sandbox, use its home (for example `home_dir='/home/user'` on E2B), not the agent host's `Path.home()`.

### 2. Asset inventory (off by default)

Set `expose_inventory_tool=True` to expose `inventory_agent_context()`, which reports where the repo's CE
assets live -- the `.claude`/`.agents`/`.codex`/`.grok` roots and, within each,
the `skills/` (SKILL.md), `agents/` (`.md`), and `settings.json` (hooks) it
contains. It returns a structured `AgentContextInventory`; it locates assets and
does not parse them.

Rename the tool with `inventory_tool_name`, or scope which roots it scans with
`asset_roots`.
Skill discovery goes at most eight directories deep.

With `FileSystem` on the agent, the inventory tool steps aside for `FileSystem`'s `read_file`, as long as every asset root is inside the working directory (the default roots are). The tool is marked with [`unless_tool_kind`](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#unless-tool-kind) set to `READ_FILE_KIND`, so it is left out of every run step that has a tool of that kind (keep `FileSystem` under its own names for now: a renamed `FileSystem` tool can't be called yet, [#10927](https://github.com/pydantic/pydantic-ai/issues/10927)). The instructions name the asset roots and say to inspect them with file tools, or with the inventory tool when it's offered. They name no reader, so they're the same with or without `FileSystem` and stay static. The decision is made from `asset_roots`, not from `FileSystem`'s access rules: `denied_patterns` that cover the roots, or `allowed_patterns` that leave them out, still drop the inventory tool.

### 3. Nested-on-traversal (off by default)

When the model lists or reads a directory, surface that directory's
`CLAUDE.md`/`AGENTS.md`. This strategy subscribes to `FileReadEvent` and
`DirectoryListedEvent`, so it receives normalized, containment-checked paths
instead of inspecting raw tool arguments. It remains opt-in:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import FileSystem, RepoContext

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[
        LocalWorkspace('.'),
        FileSystem(),
        RepoContext(
            nested_traversal=True,
            nested_inject='pointer',  # or 'contents'
        ),
    ],
)
result = agent.run_sync('List the source directory.')
```

`nested_inject='pointer'` (default) enqueues a one-line note pointing at the
file; `'contents'` enqueues the file body. The note reaches message history
before the next model request. Each directory is surfaced at most once per run.
As with the walk-up, a symlinked instruction file is named after its target.

`FileSystem` emits these events directly. Hosts with other file tools can emit
the same types by importing `FileReadEvent` and `DirectoryListedEvent` from
`pydantic_ai_harness.filesystem`; set `root_dir` to the directory the event's
`path` is relative to.

The traversed location is `root_dir / path`, so a `FileSystem` rooted at a
subdirectory of the working directory still surfaces the right directory. A
traversal that resolves outside the working directory is ignored: it is not nested in
the workspace, so there is no nested context to surface.

`traversal_tool_names` and `traversal_path_arg` are deprecated. Setting either
to a non-default value emits `HarnessDeprecationWarning` and keeps the old
tool-name and argument sniffing path active for hosts that do not emit events.
With the defaults, sniffing is disabled, so a `FileSystem` event cannot deliver
the same note twice.

## Trust

Walk-up files become part of the agent's instructions, so the model gives them
the same weight as the instructions you write yourself. That is what the strategy
is for, and it means the working directory and `home_dir` should hold only
instruction files you would put in your own instructions.

Nested files are different: the model chooses which directories to read, and
those can include vendored code or a checkout someone else wrote. Their contents
and paths reach the model as a user message, not as instructions. The
`'pointer'` note's own wording is the framework's, so it is a `SystemPromptPart`
(a system message where the model supports one mid-conversation, and
`<system>`-tagged user text otherwise), followed by the file's path as a
user-message `<context-file path="..." />` tag.

Each file is wrapped in `<context-file path="...">` tags. The `<` of every
opening or closing `<context-file>` tag inside a file, and of every framing tag
Pydantic AI writes (`<system>`, `<context>`, `<tool_result>` and the rest), is
escaped as `&lt;`, and the path is HTML-escaped, so a file stays inside its own
block.

## Cache cost

Injecting file contents into the system prompt costs prompt-cache stability: a
changed prefix re-bills the whole cached region. `RepoContext` keeps the two
cache-relevant paths separate:

- Strategy 1 reads its files **once at run start** and injects them as system
  instructions, so the cached prefix stays byte-identical across the turns of a
  run. If a file changes between runs of the same conversation, the change is
  [appended](https://pydantic.dev/docs/ai/core-concepts/agent/#appending-instruction-changes), as a diff when that's
  shorter, instead of rewriting the prefix.
- Strategy 3 is volatile (it depends on which directory was just touched), so its
  note is enqueued in the message tail, never in the system prompt, and cannot
  invalidate the cached prefix.

## Configuration

```python {test="skip"}
RepoContext(
    home_dir=None,                  # str | Path | None -- shallowest workspace dir to stop walk-up at, inclusive
    filenames=('CLAUDE.md', 'AGENTS.md'),
    autoload_instructions=True,     # Strategy 1
    expose_inventory_tool=False,    # Strategy 2
    inventory_tool_name='inventory_agent_context',
    nested_traversal=False,         # Strategy 3
    nested_inject='pointer',        # 'pointer' | 'contents'
    traversal_tool_names=frozenset({'list_directory', 'read_file'}),  # deprecated fallback
    traversal_path_arg='path',                                       # deprecated fallback
    asset_roots=('.claude', '.agents', '.codex', '.grok'),
)
```

`workspace_dir=` is deprecated and ignored: the working directory comes from the workspace.

## Scope

`RepoContext` locates and loads CE; it does not parse skill/sub-agent frontmatter
or hook bodies, and it does not modify assets. Strategy 1 reads its
files once per run, so mid-run edits to those files are not reloaded.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Pydantic AI hooks](https://ai.pydantic.dev/hooks/)
