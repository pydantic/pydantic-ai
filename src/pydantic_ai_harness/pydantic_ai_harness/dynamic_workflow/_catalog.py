"""The sub-agent catalog: `WorkflowAgent`, name validation, and the `run_workflow` description."""

from __future__ import annotations

import keyword
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic

from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.exceptions import UserError
from pydantic_ai.function_signature import FunctionSignature
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness.dynamic_workflow._prelude import PRIVATE_PREFIX, render_helper_stubs


def _is_valid_sandbox_name(name: str) -> bool:
    """Whether `name` can be exposed as a sandbox function: a non-keyword Python identifier.

    `str.isidentifier()` alone is not enough -- Python keywords (`for`, `class`, `async`, ...) are
    valid identifiers but cannot be used as function names, so the model could never call them.
    Callers guard the empty/`None` case before this is reached.
    """
    return name.isidentifier() and not keyword.iskeyword(name)


# Every sub-agent is exposed with the same fixed parameters -- `(*, task: str)` -- and a per-agent
# return schema. Render each catalog entry through core's `FunctionSignature` (the renderer
# code_mode and Pydantic AI already use). This keeps the catalog format consistent across
# capabilities, forces keyword-only `task` to match `dispatch` (which reads `kwargs['task']`), and
# renders docstrings safely -- a hand-rolled f-string breaks on a newline or a quote inside a
# description.
_SUB_AGENT_PARAMS_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {'task': {'type': 'string'}},
    'required': ['task'],
}
_NO_CONFLICTING_TYPE_NAMES: frozenset[str] = frozenset()


def _agent_return_schema(agent: AbstractAgent[AgentDepsT, object]) -> dict[str, Any] | None:
    """The sub-agent's output JSON schema, or `None` when it cannot be derived."""
    try:
        return agent.output_json_schema()
    except Exception:
        return None


def agent_signature(name: str, agent: AbstractAgent[AgentDepsT, object]) -> FunctionSignature:
    """Build the sandbox function signature for one sub-agent."""
    return FunctionSignature.from_schema(
        name=name,
        parameters_schema=_SUB_AGENT_PARAMS_SCHEMA,
        return_schema=_agent_return_schema(agent),
    )


def _render_agent_block(
    signature: FunctionSignature,
    description: str | None,
    *,
    conflicting_type_names: frozenset[str] = _NO_CONFLICTING_TYPE_NAMES,
) -> str:
    """Render one sub-agent as the async function signature shown to the model."""
    return signature.render(
        '...',
        description=description,
        is_async=True,
        conflicting_type_names=conflicting_type_names,
    )


_BASE_DESCRIPTION = """\
Write and run a Python orchestration script in a sandbox to coordinate multiple sub-agents.

Use this to break a task across specialized sub-agents and combine their results in a single step --
fan work out in parallel, chain one agent's output into the next, vote across several, or loop until
done -- instead of delegating to one sub-agent at a time.

The sandbox uses Monty, a subset of Python. Key restrictions:
- **No third-party libraries**.
- **Importable standard-library modules**: `sys`, `typing`, `asyncio`, `math`, `json`, `re`,
  `unicodedata`, `datetime`, `time`, `random`, `os`, and `pathlib`. Import what you use at the top
  of the script. Filesystem, environment, and clock operations are not configured for workflow
  scripts.
- **No clock or randomness**: `datetime.datetime.now()`, `datetime.date.today()`, `time.time()`,
  and unseeded `random` fail. `time.sleep` and `asyncio.sleep` really wait.

Each sub-agent below is an async function. Await it and pass `task` by keyword:
`result = await reviewer(task="...")`, not `reviewer("...")`; all parameters are keyword-only. A
sub-agent returns that agent's output: a string by default, or -- if it has a structured
`output_type` -- a dict, whose fields you read by subscript (`r["field"]`), not attribute
(`r.field`). Each sub-agent call is an independent run with no memory of earlier calls; include all
needed context in `task`. A failed sub-agent call surfaces as `RuntimeError`: catch it with
`try`/`except RuntimeError`, or let it abort the whole script and retry.\
"""

_RESULT_DESCRIPTION = """\
`asyncio.gather` also works, over positional awaitables and without keyword arguments such as
`return_exceptions=True`. Other task creation and wait APIs are unavailable.

The last expression's value is captured as the result -- you do **not** need to `print()` it, and
printing produces a string representation, not structured data. Use `print()` only for debug logging.
Return shapes: no print returns the last expression value (or `{}` if it is `None`); print plus a
non-`None` value returns `{"output": "<printed text>", "result": <last expression>}`; print plus
`None` returns `{"output": "<printed text>"}`. If a script fails after some sub-agent calls complete,
bounded previews of up to the 20 most recent results are reported so a retry can reuse untruncated
values.\
"""

_HELPER_DOCS: dict[str, str] = {
    'agent': (
        '`await agent(task, *, name=None, schema=None, model=None, phase=None)`: run one sub-agent '
        '({default}). `schema` is a JSON schema (`{{"type": "object", ...}}`) the output must follow for '
        'this call, so the result is a dict. `model` names a model (such as `"openai:gpt-5"`) to run it '
        'with. `phase` also calls `phase(phase)` first.'
    ),
    'parallel': (
        '`await parallel(tasks)`: run awaitables (such as `agent(...)` calls) or zero-argument functions '
        'concurrently and return their results in order. A failed item becomes `None`, so one failure does not sink the batch.'
    ),
    'pipeline': (
        '`await pipeline(items, *stages)`: run every item through each stage in turn, without waiting '
        'for the other items between stages. A stage is called as `stage(prev, item, index)`, where `prev` '
        "is the previous stage's result (the item itself for the first stage), and may be `async` or "
        'return an awaitable. A stage that raises or returns `None` ends that item with `None`.'
    ),
    'workflow': (
        '`await workflow(name, args=None)`: run a saved workflow inline and return its result, sharing '
        "this script's sub-agent budget. Raises `RuntimeError` if it fails."
    ),
    'log': '`log(message)`: report progress to the user.',
    'phase': '`phase(title)`: start a named phase of progress for the user.',
    'budget': '`budget()`: `{{"max": ..., "used": ..., "remaining": ...}}` sub-agent calls for this run.',
    'args': '`args`: the `args` passed to this tool, or `{{}}`.',
}


def render_description(
    catalog: Mapping[str, WorkflowAgent[AgentDepsT]],
    *,
    max_agent_calls: int,
    max_concurrent_agents: int,
    max_items_per_call: int,
    default_agent: str | None,
    omitted_helpers: Collection[str],
) -> str:
    """Render the `run_workflow` tool description: the sandbox, the helpers, and the sub-agent catalog."""
    signatures = {name: agent_signature(name, entry.agent) for name, entry in catalog.items()}
    signature_list = list(signatures.values())
    conflicting = FunctionSignature.get_conflicting_type_names(signature_list)
    type_blocks = FunctionSignature.render_type_definitions(signature_list, conflicting)
    function_blocks = [
        _render_agent_block(signatures[name], entry.resolved_description, conflicting_type_names=conflicting)
        for name, entry in catalog.items()
    ]
    listing = '```python\n' + '\n\n'.join([*type_blocks, *function_blocks]) + '\n```'
    if default_agent is not None:
        default = f'`name` defaults to `{default_agent}`'
    elif len(catalog) == 1:
        default = f'`name` defaults to `{next(iter(catalog))}`, the only one'
    else:
        default = '`name` picks it'
    helpers = '\n'.join(
        '- ' + doc.format(default=default) for name, doc in _HELPER_DOCS.items() if name not in omitted_helpers
    )
    limits = (
        f'This run can make at most {max_agent_calls} sub-agent calls in total -- one budget shared across '
        'every `run_workflow` call in the run, not per script; plan fan-out width accordingly. At most '
        f'{max_concurrent_agents} sub-agents run at once, and `parallel` and `pipeline` take at most '
        f'{max_items_per_call} items per call.'
    )
    return (
        f'{_BASE_DESCRIPTION}\n\nThese helpers are already defined:\n{helpers}\n\n{_RESULT_DESCRIPTION}'
        f'\n\n{limits}\n\nAvailable sub-agents:\n\n{listing}'
    )


def build_type_check_stubs(
    catalog: Mapping[str, WorkflowAgent[AgentDepsT]], *, omitted_helpers: Collection[str]
) -> str:
    """Render the helpers and sub-agent signatures as stubs for Monty's static type checker.

    Fed to `checkout(type_check=True, type_check_stubs=...)`, they let `feed_start` reject a
    positional `task`, a misspelled function, or a wrong-typed argument before the script runs --
    costing a retry but no sub-agent budget.
    """
    signatures = [agent_signature(name, entry.agent) for name, entry in catalog.items()]
    conflicting = FunctionSignature.get_conflicting_type_names(signatures)
    parts = ['import asyncio\nfrom typing import Any, TypedDict, NotRequired, Literal']
    parts.extend(render_helper_stubs(shadowed=omitted_helpers))
    parts.extend(FunctionSignature.render_type_definitions(signatures, conflicting))
    parts.extend(
        signature.render('raise NotImplementedError()', is_async=True, conflicting_type_names=conflicting)
        for signature in signatures
    )
    return '\n\n'.join(parts)


def render_reveal(
    name: str,
    catalog: Mapping[str, WorkflowAgent[AgentDepsT]],
    tool_name: str,
) -> str:
    """Announcement enqueued when a sub-agent is revealed mid-run.

    Delivered as a conversation message (not folded into the cached tool description), so the
    prompt-cache prefix stays stable while the model still learns the agent is now callable.
    """
    signatures = {agent_name: agent_signature(agent_name, entry.agent) for agent_name, entry in catalog.items()}
    signature = signatures[name]
    conflicting = FunctionSignature.get_conflicting_type_names(list(signatures.values()))
    type_blocks = FunctionSignature.render_type_definitions([signature], conflicting)
    function_block = _render_agent_block(
        signature, catalog[name].resolved_description, conflicting_type_names=conflicting
    )
    block = '\n\n'.join([*type_blocks, function_block])
    return f'A new sub-agent is now available to call from inside the `{tool_name}` script:\n\n```python\n{block}\n```'


@dataclass(frozen=True)
class WorkflowAgent(Generic[AgentDepsT]):
    """One sub-agent exposed to the orchestration script as an async function.

    `WorkflowAgent` is the per-use-site override for when the agent's own `name`
    or `description` is not what this workflow should show, such as renaming the
    sandbox function or re-describing the agent for this catalog. Passing a bare
    agent to `DynamicWorkflow(agents=[...])` is equivalent to `WorkflowAgent(agent)`.
    """

    agent: AbstractAgent[AgentDepsT, object]
    """The sub-agent to run when the script calls this function."""

    name: str | None = None
    """Sandbox function name; must be a valid Python identifier and unique across the
    workflow. Falls back to the agent's `name`."""

    description: str | None = None
    """Description shown to the model in the sub-agent catalog, rendered as the sandbox
    function's docstring. An explicit value overrides the agent's own `description`.
    When neither is set, the model sees only the bare signature."""

    @property
    def resolved_name(self) -> str | None:
        """The sandbox function name: the explicit `name`, else the agent's `name`."""
        return self.name or self.agent.name

    @property
    def resolved_description(self) -> str | None:
        """The catalog description: the explicit `description`, else the agent's own `description`."""
        return self.description or self.agent.description


def validate_workflow_agent(entry: WorkflowAgent[AgentDepsT], existing_names: set[str]) -> str:
    """Validate one sub-agent entry against the names already taken."""
    name = entry.resolved_name
    if not name:
        raise UserError(
            'DynamicWorkflow sub-agent has no `name` and its agent has no `name`; '
            'set `WorkflowAgent(name=...)` so it can be exposed as a sandbox function.'
        )
    if not _is_valid_sandbox_name(name):
        raise UserError(
            f'DynamicWorkflow sub-agent name {name!r} cannot be exposed as a sandbox function: '
            'it must be a Python identifier that is not a reserved keyword. Rename it.'
        )
    if name.startswith(PRIVATE_PREFIX):
        raise UserError(
            f'DynamicWorkflow sub-agent name {name!r} starts with {PRIVATE_PREFIX!r}, which is reserved '
            'for the sandbox helpers. Rename it.'
        )
    if name in existing_names:
        raise UserError(f'DynamicWorkflow has two sub-agents named {name!r}; names must be unique.')
    return name


def index_workflow_agents(
    agents: Sequence[WorkflowAgent[AgentDepsT]],
) -> dict[str, WorkflowAgent[AgentDepsT]]:
    """Index validated sub-agent entries by resolved sandbox name."""
    if not agents:
        raise UserError('DynamicWorkflow requires at least one sub-agent in `agents`.')
    by_name: dict[str, WorkflowAgent[AgentDepsT]] = {}
    existing_names: set[str] = set()
    for entry in agents:
        name = validate_workflow_agent(entry, existing_names)
        existing_names.add(name)
        by_name[name] = entry
    return by_name
