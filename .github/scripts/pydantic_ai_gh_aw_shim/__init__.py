"""Claude Code tools the Pydantic AI gh-aw shim exposes to the agent.

Each tool lives in its own module so individual implementations can be
swapped without touching the registry or the other tools. The toolset
builder below is the public surface the main shim consumes.

To add a new tool: drop `mytool.py` next to this file exporting one
callable, then add the `(name, callable, description)` row in
`_BASE_TOOLS`.

To replace a tool: edit the matching `<tool>.py` file. The signature is
what gh-aw / Claude pass; the docstring becomes the tool's description.

Two tools are *not* in this package:

- **WebFetch** — registered as a `pydantic_ai.capabilities.NativeTool`
  wrapping `pydantic_ai.native_tools.WebFetchTool`. The model fetches
  server-side through Anthropic's native web-fetch capability.
- **Task** — sub-agent dispatcher; lives in the main shim because it
  needs the shim-level `Agent` factory, the `INSTRUCTIONS` /
  `SUBAGENT_INSTRUCTIONS` / `RUN_TRIGGER` constants, and the
  event-stream handler. Pass it to `build_claude_code_toolset(task=...)` to
  add it as a regular tool.
"""

from collections.abc import Awaitable, Callable
from typing import TypeAlias

from pydantic_ai import RunContext
from pydantic_ai.tools import Tool
from pydantic_ai.toolsets import FunctionToolset

from .bash import bash
from .edit import edit_file
from .exit_plan_mode import exit_plan_mode
from .glob import glob_search
from .grep import grep
from .list_dir import list_dir
from .multi_edit import multi_edit
from .read import read_file
from .todo_write import todo_write
from .write import write_file

# All Claude Code tools are sync callables returning `str`. Their argument
# signatures vary by tool (Claude's `Bash` takes `(command, timeout?)`,
# `MultiEdit` takes `(file_path, edits)`, etc.), so the precise per-tool
# shape is enforced at the tool's own definition site — at the registry
# layer the meaningful contract is "callable that returns a string the
# model can read".
ClaudeCodeToolFn: TypeAlias = Callable[..., str]

# `Task` is the only async tool exposed by the shim. Its signature is
# fully pinned here so that consumers of `build_claude_code_toolset(task=...)`
# pass a compatible callable.
TaskCallable: TypeAlias = Callable[[RunContext[None], str, str], Awaitable[str]]

__all__ = [
    'MUTATING_TOOLS',
    'CLAUDE_CODE_TOOL_NAMES',
    'READ_ONLY_SUBAGENT_TOOLS',
    'bash',
    'build_claude_code_toolset',
    'edit_file',
    'exit_plan_mode',
    'glob_search',
    'grep',
    'list_dir',
    'multi_edit',
    'read_file',
    'todo_write',
    'write_file',
]


# Claude tool name → (callable, one-line description). The function names
# stay idiomatic snake_case; pydantic-ai's `Tool` exposes them under the
# Claude names so the model sees the Claude Code surface it was trained on.
_BASE_TOOLS: tuple[tuple[str, ClaudeCodeToolFn, str], ...] = (
    ('Bash', bash, 'Run a shell command in the repository workspace.'),
    ('Read', read_file, 'Read a UTF-8 text file (optional line offset/limit).'),
    ('Write', write_file, 'Create or overwrite a workspace text file.'),
    ('Edit', edit_file, 'Replace a string in a workspace file.'),
    ('MultiEdit', multi_edit, 'Apply multiple string replacements to one file atomically.'),
    ('Grep', grep, 'Recursively regex-search workspace files.'),
    ('Glob', glob_search, 'List workspace paths matching a glob pattern.'),
    ('LS', list_dir, "List a workspace directory's entries."),
    ('TodoWrite', todo_write, "Record the agent's task checklist."),
    ('ExitPlanMode', exit_plan_mode, 'Signal the end of planning and proceed.'),
)


# Claude Code Claude Code tool names the shim implements as Python callables.
# Excludes `WebFetch` (a `NativeTool` capability) and `Task` (registered by
# the main shim via `build_claude_code_toolset(task=...)`). Kept as a separate
# tuple for tests / introspection that just need the name list.
CLAUDE_CODE_TOOL_NAMES: tuple[str, ...] = tuple(name for name, _, _ in _BASE_TOOLS)

# Tools that mutate the workspace — withheld in `plan` permission mode.
MUTATING_TOOLS = frozenset({'Bash', 'Write', 'Edit', 'MultiEdit'})

# Tools handed to read-only `Task` sub-agents — strictly non-mutating and
# excluding `Task` itself to prevent recursive sub-agent spawning.
# `WebFetch` is wired separately as a `NativeTool` capability.
READ_ONLY_SUBAGENT_TOOLS = frozenset({'Read', 'Grep', 'Glob', 'LS', 'TodoWrite', 'ExitPlanMode'})


def build_claude_code_toolset(*, task: TaskCallable | None = None) -> FunctionToolset[None]:
    """Build the shim's Claude Code tool `FunctionToolset`.

    Pass `task=` to register the sub-agent dispatcher as an additional
    tool named `Task`. Sub-agent toolsets call this with `task=None` so
    sub-agents can't spawn their own sub-agents.

    Filter for permission mode / allow-list with `.filtered(predicate)`
    on the returned toolset (see `select_claude_code_toolset` in the main
    shim).
    """
    tools: list[Tool[None]] = [Tool(fn, name=name, description=desc) for name, fn, desc in _BASE_TOOLS]
    if task is not None:
        tools.append(
            Tool(
                task,
                name='Task',
                description='Dispatch a read-only sub-agent to investigate a focused task and return its findings.',
            )
        )
    return FunctionToolset(tools=tools)
