---
title: Repo Context
description: "Load a repo's AGENTS.md and CLAUDE.md instruction files into a Pydantic AI agent and inventory its skills, subagents, and hooks under .claude or .agents."
---

# Repo Context

`RepoContext` discovers and loads a repo's accumulated coding-assistant context engineering (CE): the instruction files (`CLAUDE.md`/`AGENTS.md`) scattered across the tree and the assets under `.claude`/`.agents`/`.codex`/`.grok` (skills, sub-agents, hooks).

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/repo_context/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

![RepoContext has three strategies. On by default, autoload puts one instruction file per directory, AGENTS.md or else CLAUDE.md, into the instructions: the working directory's, plus each ancestor's when walk_up is set, broadest first, once per run. Off by default, the inventory_agent_context tool reports where the .claude, .agents, .codex and .grok roots keep skills, agents and hooks. On by default, nested traversal reacts to FileSystem reading or listing a directory, such as src/api, and queues the contents of that directory's AGENTS.md before the next model request.](../img/harness/repo-context.svg)

## The problem

A repo accumulates CE for whatever coding assistant worked in it: instruction files (`CLAUDE.md`/`AGENTS.md`) scattered across the tree, and assets under `.claude`/`.agents`/`.codex`/`.grok` (skills, sub-agents, hooks). An agent that loads only the top-level instruction file misses the ancestor context and has no idea the rest of the setup exists, so it cannot honor all of it.

## The solution

`RepoContext` bundles three strategies, each independently toggleable. Construct it with `RepoContext(...)` in an `Agent`'s `capabilities`, with a workspace attached to the run:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import RepoContext

agent = Agent(
    '{{ anthropic:large }}',
    capabilities=[LocalWorkspace('.'), RepoContext(walk_up='repo_root')],
)

result = agent.run_sync('Summarize the coding-assistant setup in this repo.')
print(result.output)
```

`RepoContext` reads from the run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/), starting at its working directory. To point it at a subdirectory, set it on the workspace: `LocalWorkspace('./repo')`. `home_dir`, `global_files` and `asset_roots` are paths in the workspace: relative ones resolve from the working directory, and `~` is not expanded. A run without a workspace fails at its start.

### 1. Instruction autoload (on by default)

Loads the working directory's instruction file at run start. Each directory contributes one file: the first of `filenames` that exists there, `AGENTS.md` and then `CLAUDE.md` by default, as Codex picks one per directory. Pass `filenames=('CLAUDE.md', 'AGENTS.md')` to prefer `CLAUDE.md`. Where the workspace can resolve symlinks, a symlinked file is labeled with its target's name, so `CLAUDE.md -> AGENTS.md` reaches the model as `AGENTS.md`.

`walk_up` adds the working directory's ancestors:

- `None` (the default) scans only the working directory.
- `'repo_root'` walks up to the nearest directory, starting at the working directory, that contains one of `root_markers` (`('.git',)` by default), as Codex does. A `.git` file, as in a linked worktree, counts like a `.git` directory. When no such directory exists, only the working directory is scanned.
- `'all'` walks every ancestor up to the filesystem root, as Claude Code does.

`home_dir` also walks up, to that workspace directory inclusive, and with `walk_up='all'` it stops the walk there. It cannot be combined with `walk_up='repo_root'`. For a remote sandbox, use its home (for example `home_dir='/home/user'` on E2B), not the agent host's `Path.home()`.

`global_files` names user-global instruction files, such as `/home/user/.codex/AGENTS.md`. They load before the walk-up, so they come first and have the lowest precedence; a missing one is skipped, and one may be a symlink to anywhere, since it is the user's own.

Precedence is global-first, then ancestor-first, workspace-last: broadest context first, most specific last. Files are deduped by content hash, so two ancestors (or a global file and a project file) with identical content load once. A file in an ancestor is labeled relative to the working directory, such as `../AGENTS.md`, not by its absolute path; a global file outside the working directory's ancestry is labeled by its absolute path.

### 2. Asset inventory (off by default)

Set `expose_inventory_tool=True` to expose `inventory_agent_context()`, which reports where the repo's CE assets live -- the `.claude`/`.agents`/`.codex`/`.grok` roots and, within each, the `skills/` (`SKILL.md`), `agents/` (`.md`), and `settings.json` (hooks) it contains. It returns a structured `AgentContextInventory`; it locates assets and does not parse them.

Rename the tool with `inventory_tool_name`, or scope which roots it scans with `asset_roots`.
Skill discovery goes at most eight directories deep.

With `FileSystem` on the agent, the inventory tool steps aside for `FileSystem`'s `read_file`, as long as every asset root is inside the working directory (the default roots are). The tool is marked with [`unless_tool_kind`][pydantic_ai.tools.ToolDefinition.unless_tool_kind] set to `READ_FILE_KIND`, so it is left out of every run step that has a tool of that kind (keep `FileSystem` under its own names for now: a renamed `FileSystem` tool can't be called yet, [#10927](https://github.com/pydantic/pydantic-ai/issues/10927)). The instructions name the asset roots and say to inspect them with file tools, or with the inventory tool when it's offered. They name no reader, so they're the same with or without `FileSystem` and stay static. The decision is made from `asset_roots`, not from `FileSystem`'s access rules: `denied_patterns` that cover the roots, or `allowed_patterns` that leave them out, still drop the inventory tool.

### 3. Nested-on-traversal (on by default)

When the model lists a directory or reads a file in it, surface the
instruction file of that directory and of every directory between it and
the working directory, shallowest first: reading `packages/foo/src/x.py` also
surfaces `packages/foo/AGENTS.md`. This strategy subscribes to `FileReadEvent` and
`DirectoryListedEvent`, so it receives normalized, containment-checked paths
instead of inspecting raw tool arguments. It is on by default, so with
`FileSystem` on the agent nothing else is needed; this switches it to pointers:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import FileSystem, RepoContext

agent = Agent(
    '{{ anthropic:large }}',
    capabilities=[
        LocalWorkspace('.'),
        FileSystem(),
        RepoContext(nested_inject='pointer'),
    ],
)
result = agent.run_sync('List the source directory.')
```

`nested_inject='contents'` (default) enqueues the file body; `'pointer'`
enqueues a one-line note pointing at the file. A
[9-model benchmark](https://github.com/pydantic/pydantic-ai/issues/11683#issuecomment-6510271135)
found the rule in a nested file followed in 99% of runs with `'contents'`, 95%
with `'pointer'`, and 39% with nested traversal off. Pass
`nested_traversal=False` to turn it off. `nested_max_chars` (32,000 by default)
caps the characters of file bodies `'contents'` inlines per conversation,
counting every inlined note still in the history the model sees, so earlier
runs count and a compaction frees room: a file that would go past it is
surfaced as a pointer instead, and `None` removes the cap. The note reaches
message history before the next model request. A directory counts as surfaced while its note
is still in the conversation: a later run of the same conversation doesn't
repeat it, and once compaction summarizes the note away, the next traversal
surfaces it again. The working directory's files are not surfaced again when
the walk-up already loaded them. Every part of a note carries
`{'origin': 'pydantic-ai-harness.repo_context.nested_note.v1', 'directory': ...}`
in its `metadata`, which is how `RepoContext` and compaction recognize it.
As with the walk-up, a symlinked instruction file is named after its target.

With nested traversal on, the instructions also tell the model how instruction
files relate: each applies to its own directory and everything below it, the
more deeply nested file wins a conflict, and the user's direct instructions win
over all of them. The statement is static, so it stays in the cached prefix.

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

Walk-up files become part of the agent's instructions, so the model gives them the same weight as the instructions you write yourself. That is what the strategy is for, and it means the working directory and every directory the walk-up reaches (`home_dir`, the repository root for `walk_up='repo_root'`, or every ancestor for `'all'`) and every `global_files` entry should hold only instruction files you would put in your own instructions.

Nested files are different: the model chooses which directories to read, and those can include vendored code or a checkout someone else wrote. Their contents and paths reach the model as a user message, not as instructions. The `'pointer'` note's own wording is the framework's, so it is a `SystemPromptPart` (a system message where the model supports one mid-conversation, and `<system>`-tagged user text otherwise), followed by the file's path as a user-message `<context-file path="..." />` tag.

Each file is wrapped in `<context-file path="...">` tags. The `<` of every opening or closing `<context-file>` tag inside a file, and of every framing tag Pydantic AI writes (`<system>`, `<context>`, `<tool_result>` and the rest), is escaped as `&lt;`, and the path is HTML-escaped, so a file stays inside its own block.

## Cache cost

Injecting file contents into the system prompt costs prompt-cache stability: a changed prefix re-bills the whole cached region. `RepoContext` keeps the two cache-relevant paths separate:

- Strategy 1 reads its files once at run start and injects them as system instructions, so the cached prefix stays byte-identical across the turns of a run. If a file changes between runs of the same conversation, the change is [appended](../agent.md#appending-instruction-changes), as a diff when that's shorter, instead of rewriting the prefix.
- Strategy 3 is volatile (it depends on which directory was just touched), so its note is enqueued in the message tail, never in the system prompt, and cannot invalidate the cached prefix.

## Configuration

```python {test="skip"}
RepoContext(
    walk_up=None,                   # None | 'repo_root' | 'all' -- how far above the working directory to look
    root_markers=('.git',),         # what marks the repository root for walk_up='repo_root'
    home_dir=None,                  # str | Path | None -- walk up to this workspace dir, inclusive
    global_files=(),                # Sequence[str | Path] -- user-global instruction files, loaded first
    filenames=('AGENTS.md', 'CLAUDE.md'),  # one per directory: the first that exists
    autoload_instructions=True,     # Strategy 1
    expose_inventory_tool=False,    # Strategy 2
    inventory_tool_name='inventory_agent_context',
    nested_traversal=True,          # Strategy 3
    nested_inject='contents',       # 'contents' | 'pointer'
    nested_max_chars=32_000,        # int | None -- cap on inlined nested contents per conversation
    traversal_tool_names=frozenset({'list_directory', 'read_file'}),  # deprecated fallback
    traversal_path_arg='path',                                       # deprecated fallback
    asset_roots=('.claude', '.agents', '.codex', '.grok'),
)
```

`workspace_dir=` is deprecated and ignored: the working directory comes from the workspace.

## Scope

`RepoContext` locates and loads CE; it does not parse skill/sub-agent frontmatter or hook bodies, and it does not modify assets. Strategy 1 reads its files once per run, so mid-run edits to those files are not reloaded.

## Upgrading

`RepoContext()` now loads one instruction file per directory, `AGENTS.md` first and `CLAUDE.md` as the fallback, and surfaces nested instruction files by default, inlining up to 32,000 characters of their contents per conversation before surfacing the rest as pointers (`nested_max_chars=None` removes the cap). To get close to the old behavior, pass `RepoContext(filenames=('CLAUDE.md', 'AGENTS.md'), nested_traversal=False)`. It only approximates it: no setting loads both files from one directory any more, so a directory holding both loads only `CLAUDE.md`. Durable runs started before the upgrade record fewer workspace calls than a new run makes, so let them finish or version the deployment first.

## Further reading

- [Pydantic AI capabilities](../capabilities/overview.md)
- [Pydantic AI hooks](../hooks.md)

## API reference

::: pydantic_ai_harness.repo_context.RepoContext

::: pydantic_ai_harness.repo_context.AgentContextInventory

::: pydantic_ai_harness.repo_context.AssetRoot
