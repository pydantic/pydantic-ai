"""Back an agent's Agent Control configuration with one Logfire variable.

The contract itself -- what an `AgentConfig` holds, how leniently it validates, and what a published
value does to a request -- lives in [`logfire.agent_control`][], shared by every framework adapter and
by the Logfire UI. This module is the Pydantic AI half of it: the capability wiring, the toolset that
carries managed definitions to the model, the bridge between the contract's string block ids and
[`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id], and the hint span an agent
describes itself to Logfire on.

Nothing here writes to a Logfire variable. Every agent says what its code says on a span carrying its
code baseline, and creating a config from that -- or refreshing a stored baseline that no longer
matches -- is a Logfire-side flow. So a deployment needs no write scope, and the code baseline can
never race an edit saved in the UI.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import logfire
from logfire.agent_control import (
    AGENT_VARIABLE_PREFIX,
    SCHEMA_SHA256,
    AgentConfig,
    AgentConfigSettings,
    AgentSupport,
    ApplyIssue,
    Block,
    InstructionBlock,
    OnUnmatched,
    Section,
    ToolDef,
    UnmatchedConfigError,
    apply_instructions,
    apply_settings,
    apply_tool_definitions,
    build_baseline,
    canonical_json,
    report_issues,
)
from logfire.variables import Variable
from logfire.variables.abstract import NoOpVariableProvider

from pydantic_ai import AbstractToolset, RunContext, TemplateStr, ToolDefinition, WrapperToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, CombinedCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models import ModelRequestContext, infer_model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai_harness.logfire._managed_variable import (
    ManagedVariableCapability,
    resolution_reason,
)

if TYPE_CHECKING:
    from logfire.variables import ResolvedVariable

    from pydantic_ai.agent.abstract import AbstractAgent, AgentModelSettings
    from pydantic_ai.capabilities import AgentModel, ModelSelection
    from pydantic_ai.capabilities.abstract import WrapRunHandler
    from pydantic_ai.models import ModelSelectionContext
    from pydantic_ai.run import AgentRunResult

_AGENT_CONTROL_ID = 'agent-control'
"""The capability ID an `AgentControl` carries, so two of them read as one managed config."""

# Where a renamed tool records the code-side name it routes back to, on its own definition's
# `metadata`, so `call_tool` reads it off the very tool the model was handed. Recomputing the mapping
# from a fresh listing instead answers about whatever occupies that advertised name *now*: a dynamic
# toolset that put a different tool behind it in between supplies its code-side name, and the call
# then runs the tool the model chose while a name-based policy -- `ApprovalRequiredToolset` above all
# -- is handed the other one's name and authorizes that instead. It is why the contract's `routes`
# mapping is not what this adapter routes on. The key is removed again on the way through, so nothing
# downstream sees it.
_ROUTES_TO_METADATA_KEY = '__agent_control_routes_to__'

# Drop warnings already emitted in this process, keyed by the message itself. See `_warn_dropped`.
_warned_drops: set[str] = set()

# Code identities already hinted in this process. The key is the Logfire *config* the variable
# resolves through, the variable's name, and the agent name as written. A process serving two
# Logfire projects reports to both, while two differently named agents that normalize to one
# variable each report their own baseline.
#
# The config rather than the `Logfire` instance, because `Variable.__init__` stores
# `logfire_instance.with_settings(...)`, which builds a new `Logfire` per variable, and `Logfire` has
# no value equality. Keyed on the instance, every `Variable` carried its own key, so the guard only
# ever deduplicated repeat runs of one long-lived agent: an agent constructed per request reported a
# hint on every request. `with_settings` passes `config` through by reference, so it is the object
# that is actually one per process per project -- which is what "once per process" was always meant
# to mean.
#
# Keyed by `id(config)` rather than the config itself because `LogfireConfig` is a mutable dataclass
# and so unhashable, and the config is held alongside its id so that it cannot be collected and have
# that id handed to a different config -- which would suppress a hint that was never sent. The type
# is spelled `object` rather than `LogfireConfig`, which logfire does not export: nothing here does
# anything with it except keep it alive.
_config_hint_emitted: dict[int, tuple[object, set[tuple[str, str | None]]]] = {}
_config_hint_lock = threading.Lock()

_CONFIG_HINT_SPAN_NAME = 'agent_control_config_hint'
"""The span an unconfigured agent reports itself on, and the name a Logfire-side query selects it by.

Static and separate from the message, so the message can be reworded without moving what the platform
indexes on. The attributes it carries are documented on `_emit_config_hint`.
"""

_CONFIG_HINT_MESSAGE = 'Agent Control reported the code baseline for this agent'
"""The hint span's message. The agent it is about is the trace it sits in, and the variable it names
is an attribute, so the message stays the same string for every agent."""

_FRAMEWORK = 'pydantic-ai'
"""Which Agent Control SDK produced a hint.

The contract is shared with the TypeScript cores and the Logfire UI, and the ids a baseline addresses
its instruction blocks by are each implementation's own, so a consumer has to know whose baseline it
is reading.
"""

_SUPPORT = AgentSupport(
    sections=frozenset(('instructions', 'model', 'settings', 'tool_definitions')),
    settings=frozenset(AgentConfigSettings.model_fields),
    precedence='exact',
    resolution_unit='run',
)
"""What this adapter can do with a published config, for the apply helpers and the Logfire editor.

Every section and every canonical setting. Pydantic AI has somewhere to put all four, and the
contract's canonical keys are `ModelSettings` keys by definition -- the same invariant that lets a
published patch be handed to Pydantic AI as one -- so they are read off `AgentConfigSettings` rather
than restated here: a key the contract gains is one this adapter can already lower, and listing them
by hand would report it as a key Pydantic AI has no equivalent for until someone updated the list.

`precedence='exact'` because managed settings are contributed per request and merged under the keys
the caller passed to `run(model_settings=...)`, so the contract's order -- code, then published, then
the run -- holds key by key rather than being inferred from a diff against defaults.
`resolution_unit='run'` because the variable is resolved once per run, so a version published
mid-run reaches the next run, which is what lets every span of a run agree on the version that
produced it. Both happen to be the defaults, and are declared anyway: this value is what the editor
reads instead of guessing, and a field nobody wrote is indistinguishable from one nobody considered.

`destinations` is left empty, which says one unnamed sink. Pydantic AI assembles its prompt from many
blocks, but they are one prompt: an added block names no destination because there is only the one.

Nothing here describes a model. A published `temperature` a reasoning model refuses still reaches the
provider, because which settings a model takes is the provider's answer to give and not a table for
this adapter to carry.
"""

_MAX_BASELINE_BYTES = 1 << 20
"""How much serialized baseline a hint span will carry, in UTF-8 bytes.

Span attributes share a row budget in the low tens of megabytes, and the backend enforces it by
truncating a long string *in place* -- which for JSON means an attribute that still looks like a
string and no longer parses. So the budget is enforced here instead, an order of magnitude under the
row's, with room left for the rest of the span and for a consumer's own overhead. A baseline over it
is reduced by whole sections rather than cut mid-string; see `_serialize_baseline`.
"""

BaselineReduction = Literal['none', 'tool_definitions', 'omitted']
"""What a baseline gave up to fit `_MAX_BASELINE_BYTES`, as `agent_control.baseline_reduction` reports it."""


def _warn_dropped(message: str) -> None:
    """Surface a dropped managed value once per process.

    A drop means Logfire shows one thing and the agent does another, which has to be visible. But the
    config is resolved on every single run, so warning per drop would bury that signal under its own
    repetition. Deduplicating on the message rather than on the field costs nothing in clarity -- each
    message names its subject and the offending value -- and still lets a *different* unrecognized
    value surface later, which a per-field guard would swallow. A concurrent first run can at worst
    duplicate the warning, which is not worth a lock.

    The contract's own parser deduplicates the same way, in its own set: the two never emit the same
    message, so one message still reaches the user once however it was produced.
    """
    if message in _warned_drops:
        return
    _warned_drops.add(message)
    try:
        warnings.warn(message)
    except Exception:  # pragma: no cover
        # Emitted filter-independently, for the reason the contract's own `warn_dropped` is: every
        # drop reported here happens while a managed config is being read or applied, and
        # `docs/agent-control.md` promises a published config can never crash a run. Under
        # `-W error` / `filterwarnings = ['error']` `warnings.warn` raises, which does not make
        # anything stricter -- a drop raised during resolution is caught by logfire's fallback and
        # silently un-manages every section that did apply. Strictness is `on_unmatched='error'`,
        # which raises `UserError` naming every entry and is untouched by this.
        pass


def _report(policy: OnUnmatched, issues: Sequence[ApplyIssue]) -> None:
    """Report what a request could not apply, as the error Pydantic AI refuses a run with.

    Called with every section's issues at once, once the request is planned, and not from validation.
    Validation runs inside Logfire's resolution, which turns any exception into a fallback to the
    code-defined agent, so an `'error'` raised there would be swallowed and un-manage the whole config
    instead of stopping the run. And the same validation builds the code baseline, where a key this
    SDK has no field for is the agent's own `extra_headers` rather than anything anyone published.

    The policy itself is the contract's, applied by `report_issues`, so every message reads the same
    under `'warn'` as under `'error'` and a kind of issue the contract adds later reaches the user
    without this adapter learning about it. Only the exception is this adapter's:
    [`UserError`][pydantic_ai.exceptions.UserError] is what Pydantic AI raises for a
    misconfiguration, which is what a published config this deployment cannot apply is. It is
    translated from the contract's error rather than rebuilt, so it names every issue the request
    got wrong.
    """
    try:
        report_issues(policy, issues)
    except UnmatchedConfigError as exc:
        raise UserError(str(exc)) from exc


def _reset_config_hint_guard() -> None:  # pyright: ignore[reportUnusedFunction]
    """Clear the once-per-process hint guard. Intended for tests only."""
    with _config_hint_lock:
        _config_hint_emitted.clear()


def _serialize_baseline(baseline: AgentConfig) -> tuple[str | None, BaselineReduction, int]:
    """Serialize a baseline to fit `_MAX_BASELINE_BYTES`, dropping whole sections when it does not.

    Returns the JSON to put on the hint span (`None` when even the reduced baseline does not fit),
    which reduction was applied, and the full baseline's size in UTF-8 bytes before any of it.

    Reduction drops `tool_definitions` first and then gives up, rather than trimming text to a
    budget. The contract already bounds a single instruction block, so a baseline this large is one
    with an unbounded *number* of things in it, and tool definitions are where that goes: every
    advertised tool, its description, and an entry per parameter. They are also the section a config
    needs least -- the editor can offer a tool override without one, while a baseline with no
    instructions has nothing to show at all. Either way the result is a whole, valid `AgentConfig`
    and the reduction is named on the span, so a consumer reads a baseline that is complete or
    knowingly partial, never one that parses into something the agent does not do.
    """
    serialized = _dump(baseline)
    size = len(serialized.encode())
    if size <= _MAX_BASELINE_BYTES:
        return serialized, 'none', size
    reduced = _dump(baseline.model_copy(update={'tool_definitions': None}))
    if len(reduced.encode()) <= _MAX_BASELINE_BYTES:
        return reduced, 'tool_definitions', size
    return None, 'omitted', size


def _dump(baseline: AgentConfig) -> str:
    """The baseline as the JSON a hint carries, indented the way a variable's `example` is read."""
    return json.dumps(baseline.model_dump(exclude_none=True), indent=2)


def _baseline_sha256(baseline: AgentConfig) -> str:
    """The digest that says whether two reports describe the same code.

    Always over the *whole* baseline, never over the JSON the span ended up carrying. A baseline too
    large for a span attribute is reported with its tool definitions dropped, or with no baseline at
    all, and a digest taken after that would change when nothing about the agent had -- and would
    leave every oversize report looking like every other one, which is the case that has nothing else
    to tell it apart by. Which makes this the digest of a document the span does not always carry: a
    consumer compares it with another hint's, and verifies it against `agent_control.baseline` only
    when `agent_control.baseline_reduction` is `'none'`.

    What makes that verification hold is that the hint's attributes are exempt from Logfire's
    scrubbing, which is on by default and matches substrings: an instruction block reading "Order
    tools are authoritative for status and refunds" would otherwise be carried as
    `[Scrubbed due to 'auth']`, rewriting the document after this digest and the byte count were taken
    over it. The exemption is `BaseScrubber.SAFE_KEYS` in the `logfire` package, which is what makes
    it one list for every Agent Control SDK rather than a thing each adapter arranges for itself.
    """
    return hashlib.sha256(canonical_json(baseline.model_dump(exclude_none=True))).hexdigest()


def _deployment_attributes(instance: logfire.Logfire) -> dict[str, Any]:
    """Which deployment reported a baseline, from what the Logfire instance already knows.

    The variable a config lives in is derived from the agent's name alone, so two services that each
    define a `checkout_assistant` land on one `agent__checkout_assistant` -- and so do the same
    service's dev and prod deployments, since a variable is one value per project and the dev/prod
    split is its labels. Without this a consumer cannot tell whose code it is looking at. The
    resource attributes on the span carry some of it, but a contract the platform indexes has to say
    what it promises, and OTel resource attributes are not something this span has promised.

    Read off the instance the hint is emitted on rather than configured here: a process serving two
    Logfire projects configures each separately, and asking the user to restate a service name they
    already gave `logfire.configure()` is a second place for the two to disagree. `service_version`
    is whatever Logfire resolved for the running code, which is the current commit when the process
    runs in a git checkout and has not been told otherwise.

    Anything the SDK does not know is left off rather than sent as an empty string: absent is a state
    a consumer can act on, where `''` is a value it has to learn to disbelieve.
    """
    config = instance.config
    attributes: dict[str, Any] = {
        'agent_control.service_name': config.service_name,
        'agent_control.environment': config.environment,
        'agent_control.service_version': config.service_version,
    }
    return {name: value for name, value in attributes.items() if value}


def _toolset_key(toolset: AbstractToolset[Any]) -> str:
    """The string a tool's toolset is reported and matched as: its `id`, or its label when it has none.

    One rule for both directions, so the value the baseline shows in the editor is exactly the value
    an override can be narrowed by.
    """
    return toolset.id or toolset.label


def _instruction_key(part: InstructionPart) -> str | None:
    """The string key a managed config addresses this part by, or `None` if nothing addresses it.

    Pydantic AI issues a structured [`InstructionId`][pydantic_ai.messages.InstructionId]; a managed
    config arrives as JSON and names parts by the string that id renders to. Bridging the two once
    here keeps every caller comparing keys of the same kind.

    The contract keeps ids free-form and reserves only `'agent'` as the cross-framework name for the
    prompt as written; the rest of the namespace belongs to each implementation, and these are
    Pydantic AI's: `'agent'` for the agent's own literal instructions, `'toolset:<id>'` and
    `'capability:<id>'` for what a toolset or capability contributes, and `'agent:<declared name>'` /
    `'capability:<id>:<declared name>'` for a single declared block. Blocks Pydantic AI cannot key --
    a callable passed to `Agent(instructions=...)`, a toolset with no `id` of its own -- cannot be
    addressed at all.
    """
    return str(part.id) if part.id is not None else None


def _blocks(parts: Sequence[InstructionPart]) -> list[Block]:
    """The assembled instruction parts as the contract's neutral blocks."""
    return [Block(text=part.content, id=_instruction_key(part), dynamic=part.dynamic) for part in parts]


def _without_implicit_drops(config: AgentConfig, blocks: Sequence[Block]) -> AgentConfig:
    """The config without the instruction entries that would drop a block they never asked to drop.

    The contract drops an addressed block when its entry's `instructions` is `None`, and validates an
    entry whose `instructions` key is missing to that same `None` while ignoring keys it does not know.
    So `{"id": "agent", "text": "..."}` -- one misspelled key -- removes the agent's whole prompt and
    says nothing. The stored JSON schema describes `null` as the way to drop a block, so an explicit
    `null` is the only thing this takes as one: an entry whose `instructions` was never set is left out
    with a warning and its block keeps the code-defined text.

    Only entries that would actually drop something are held back -- ones addressing a static block
    this request assembles. Every other entry still reaches the contract, which reports an `id` nothing
    carries, or only a dynamic block carries, under `on_unmatched` as it would any other.
    """
    instructions = config.instructions
    if not isinstance(instructions, list):
        return config
    droppable = {block.id for block in blocks if block.id is not None and not block.dynamic}
    kept: list[str | InstructionBlock] = []
    for entry in instructions:
        if (
            isinstance(entry, InstructionBlock)
            and entry.id in droppable
            and 'instructions' not in entry.model_fields_set
        ):
            _warn_dropped(
                f'Managed instruction entry for block {entry.id!r} has no `instructions` key, so it is not applied '
                'and the block keeps its code-defined text. Publish `"instructions": null` to drop the block, or '
                'the replacement text to replace it; a misspelled key such as `text` reads as a missing one.'
            )
        else:
            kept.append(entry)
    if len(kept) == len(instructions):
        return config
    return config.model_copy(update={'instructions': kept})


ConfigProvider = Callable[[], 'AgentConfig | None']
ToolsObserver = Callable[[list['ToolsetTool[Any]']], None]
IssuePlanner = Callable[[Section, Sequence[ApplyIssue]], None]
"""Hand one section's issues to the capability, which reports every section's together."""


def _advertised_tool(
    tool: ToolsetTool[AgentDepsT], applied: ToolDef, *, code_name: str, toolset: AbstractToolset[AgentDepsT]
) -> ToolsetTool[AgentDepsT]:
    """The tool as the model will be shown it, remembering the name a call routes back to.

    Only the LLM-facing fields move: a renamed tool keeps its implementation, its argument validation,
    and the parameter names and types it was defined with.
    """
    tool_def: ToolDefinition = tool.tool_def
    changes: dict[str, Any] = {}
    if applied.name != code_name:
        changes['name'] = applied.name
        changes['metadata'] = {**(tool_def.metadata or {}), _ROUTES_TO_METADATA_KEY: code_name}
    if applied.description != tool_def.description:
        changes['description'] = applied.description
    if applied.parameters_json_schema is not tool_def.parameters_json_schema:
        changes['parameters_json_schema'] = applied.parameters_json_schema
    if not changes:
        return tool
    # `replace` preserves concrete `ToolDefinition` subclasses and fields added by the framework.
    return replace(tool, toolset=toolset, tool_def=replace(tool_def, **changes))


@dataclass
class _ToolDefinitionOverridesToolset(WrapperToolset[AgentDepsT]):
    """Overlay advertised definitions while routing calls to their code-side tool names."""

    get_config: ConfigProvider = field(repr=False, compare=False)
    observe_code_tools: ToolsObserver = field(repr=False, compare=False)
    plan_issues: IssuePlanner = field(repr=False, compare=False)

    def _effective_tools(
        self, config: AgentConfig, tools: dict[str, ToolsetTool[AgentDepsT]]
    ) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Advertise what the managed config says, keyed by the name the model will call.

        Which override wins, what a colliding rename costs, and which patch reached nothing are all
        the contract's to decide; this hands it the tools this listing advertises and applies what
        comes back. Pydantic AI advertises every tool into one flat namespace, which is the
        `collision_scope='global'` the contract defaults to.

        Planned on every listing, because tool availability is dynamic -- a toolset can advertise
        different tools from one step to the next -- and a report from any one listing is a report
        about that listing. Reported by the capability, with the other sections' issues, once the
        request this listing belongs to is planned.
        """
        applied = apply_tool_definitions(
            [
                ToolDef(
                    name=name,
                    description=tool.tool_def.description,
                    parameters_json_schema=tool.tool_def.parameters_json_schema,
                    toolset=_toolset_key(tool.toolset),
                )
                for name, tool in tools.items()
            ],
            config,
        )
        advertised = {
            applied_def.name: _advertised_tool(tool, applied_def, code_name=code_name, toolset=self)
            for (code_name, tool), applied_def in zip(tools.items(), applied.tools)
        }
        self.plan_issues('tool_definitions', applied.issues)
        return advertised

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return tools with managed definitions and collision-safe advertised names."""
        tools = await super().get_tools(ctx)
        self.observe_code_tools(list(tools.values()))
        config = self.get_config()
        if config is None or not config.tool_definitions:
            return tools
        return self._effective_tools(config, tools)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Translate a renamed model call back to its original implementation and context name."""
        metadata = tool.tool_def.metadata or {}
        original_name = metadata.get(_ROUTES_TO_METADATA_KEY)
        if isinstance(original_name, str) and original_name != name:
            ctx = replace(ctx, tool_name=original_name)
            routed_metadata = {key: value for key, value in metadata.items() if key != _ROUTES_TO_METADATA_KEY}
            tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name, metadata=routed_metadata or None))
            return await super().call_tool(original_name, tool_args, ctx, tool)
        return await super().call_tool(name, tool_args, ctx, tool)


@dataclass
class AgentControl(ManagedVariableCapability[AgentDepsT, AgentConfig]):
    """Manage an agent's config through one `agent__<name>` Logfire variable.

    The variable holds an [`AgentConfig`][logfire.agent_control.AgentConfig], the contract every
    Agent Control SDK shares. Each present section -- `instructions`, `model`, `settings`, or
    `tool_definitions` -- is managed from Logfire, while an absent section keeps the code-defined
    behavior. Removing a section in Logfire deliberately reverts that section to code.

    Instructions are the one section that **composes with** the agent instead of patching it, so it
    works in two ways, and which one an entry uses is the difference between a prompt that reads well
    and one sent to the model twice.

    An entry with **no `id` adds** a block. Adding is also the one way the same text can reach the
    model twice: seeding a config from an agent's observed prompt while that text stays in
    `Agent(instructions=...)` sends every block of it twice over.

    An entry **with an `id` swaps out** the block Pydantic AI assembled under that key -- replacing its
    text, or dropping it with an explicit `instructions=None`; an entry that leaves `instructions` out
    is skipped with a warning rather than read as a drop. This is how a managed config reaches text no
    capability owns: the agent's own literal, a toolset's, an MCP server's, one `@agent.instructions`
    function out of several. See [`InstructionPart.id`][pydantic_ai.messages.InstructionPart.id] for
    which blocks have a key at all.

    Both are applied in `before_model_request`, after every contribution has been assembled, so an
    override addresses what the model was about to be sent and an added block lands at the end of the
    run of static blocks -- last in the prompt as written, and still inside the prefix a provider can
    cache. An override leaves its block's position and its `dynamic` flag alone, so no override moves
    that cache boundary either. An added block is static for the same reason, unless `render_template`
    makes it a per-run rendering.

    When `name` is omitted, the variable name is derived from the agent's telemetry name using the
    same normalization as the Logfire UI, which is lossy: `checkout-assistant` and `checkout_assistant`
    both resolve `agent__checkout_assistant`, so two agents that differ only in punctuation -- in one
    service or across several in the same project -- share one managed config. Pass an explicit `name`
    to keep them apart. The variable is resolved once per run, and its label and version baggage
    remains active for the whole run.

    The managed `model` is sourced during model selection, so it slots in with the right precedence:
    a call-site `run(model=...)` beats it -- one run of one process, deliberately overriding what is
    published -- and it beats everything else, the agent's constructor model and any other
    capability's alike. A fully model-less `Agent(None, ...)` -- named or nameless -- can therefore be
    driven entirely from managed config. Managed settings work the same way, patching over the
    agent's and every other capability's; settings passed to `run()` still win.
    Static or callable targeting inputs participate in model selection, and the resulting variable
    resolution is reused for the rest of the run so the selected model and applied config cannot
    diverge. An unknown managed model warns and keeps the code model.

    Tool overrides change only the definitions shown to the model. Renames route back to the original
    implementation, collisions retain the original name, and an override the contract can't validate
    is dropped with a warning while its siblings still apply. An override names its tool by code-side
    `name`, narrowed to one toolset's tool of that name when it sets `toolset` too. Parameter names,
    types, requiredness, validation, and implementation stay code-owned.

    A published entry that reaches nothing -- an instruction `id` no assembled block carries (or only
    a dynamic one carries), a tool override no advertised tool matches, a rename another tool already
    answers to, a parameter patch the tool has no parameter for, a settings key the contract has no
    field for, a whole section it does not know -- is governed by `on_unmatched`, applied once per
    request to every section at once. The default warns once per process rather than raising, because
    one config is applied across deployments that need not all install the same toolsets, and a
    toolset can advertise different tools from one step to the next; an entry that reaches nothing
    here may be exactly right somewhere else. `'error'` is for the deployment that
    would rather stop than run with part of its published config silently unapplied.

    Missing, invalid, or unreachable remote values degrade to the code-defined agent through Logfire's
    resolution fallback, which is why a value the contract doesn't recognize degrades the narrowest
    unit that contains it -- one setting, one tool override, one section -- rather than the whole
    config.

    Nothing here ever writes to a variable. Every agent emits one `agent_control_config_hint` span
    per process carrying an `AgentConfig`-shaped snapshot of the code-side agent, whether or not a
    config resolved, and turning that into a real config -- or refreshing a stored baseline the code
    has moved on from -- is a Logfire-side flow the project's owner clicks through. So the credential
    this needs is read-only, and a code baseline can never overwrite something saved in the UI.

    The snapshot is taken from whichever model request comes first in the process, so for
    instructions or a toolset that vary with `deps`, run input, or the step within a run it is one
    point-in-time sample rather than a description of the agent, and an agent that never reaches a
    model request reports nothing. Its `instructions` are the code-defined
    blocks, listed separately with the `id` that addresses each block and a `dynamic` flag. This lets
    the UI offer an override per block instead of one copy-the-whole-prompt button that would produce
    exactly the duplication described above.

    ```python
    import logfire
    from pydantic_ai import Agent

    from pydantic_ai_harness.logfire import AgentControl

    logfire.configure()
    agent = Agent(
        'anthropic:claude-fable-5-1',
        name='checkout_assistant',
        instructions='You are a checkout assistant.',
        capabilities=[AgentControl(label='production')],
    )
    result = agent.run_sync('Refund my last order.')
    ```

    Runtime `{{...}}` placeholders pass through unless `render_template=True`. During a run,
    `logfire.managed.applied_sections` lists the present sections. The `model` section is reported
    when present even if a call-site model outranked it for that run.
    """

    name: str | Variable[AgentConfig] | None = None
    """Bare variable name, pre-built variable, or `None` to derive it from the agent name.

    A nameless capability derives its variable (and can source the model) from the agent's `name`,
    which must then be one the agent was given explicitly.
    """
    render_template: bool = False
    """Render `{{...}}` placeholders in *added* instruction text against run dependencies when enabled.

    An added block is then a rendering of this run's `deps` rather than fixed text, so it is
    contributed as a dynamic block -- the same way Pydantic AI treats any other
    [`TemplateStr`][pydantic_ai.TemplateStr] -- and sorts after the static prompt rather than inside
    the prefix a provider caches.

    An entry that addresses an existing block by `id` is applied to the assembled request and is never
    templated: it replaces a block with exactly the text that was published.
    """
    on_unmatched: OnUnmatched = field(default='warn', kw_only=True)
    """What to do with a published entry that reaches nothing in this deployment.

    That is an instruction `id` no assembled block carries, or that only a dynamic block carries; a
    tool override whose `name` (and `toolset`, when set) matches no tool a toolset advertises; a
    rename another advertised tool already answers to; a parameter patch naming a parameter the tool
    does not have; a `settings` key this version of the contract has no field for; and a top-level
    key it has no section for at all. Each is a place where Logfire shows one thing and the agent
    does another.

    - `'warn'` (the default) emits a `UserWarning` once per process per message.
    - `'error'` raises [`UserError`][pydantic_ai.exceptions.UserError] naming every entry this
      request could not apply, failing the run.
    - `'ignore'` applies nothing and says nothing.

    Every section is planned before any of it is reported, so the strictest policy is also the most
    complete one: `'error'` names the settings key *and* the tool override *and* the instruction
    entry, rather than whichever section the agent graph happened to reach first.

    Warning rather than raising is the default because tool availability is dynamic: one config is
    applied across deployments that need not all install the same toolsets, and a toolset can
    advertise different tools from one request to the next, so an entry that reaches nothing now is
    not necessarily wrong. Once per process rather than once per run for the same reason every other
    drop is: the config is resolved on every run, and the signal has to survive its own repetition.
    """

    # Override the inherited default ID: a stable id is what tells Pydantic AI an agent has one
    # managed config and that two of it are one configuration stated twice, rather than two anonymous
    # capabilities that collide. Declared in the class body rather than assigned in `__post_init__`,
    # which is where the framework reads it from.
    id: str | None = field(default=_AGENT_CONTROL_ID, kw_only=True)

    _selection_resolved: ContextVar[ResolvedVariable[AgentConfig] | None] = field(init=False, repr=False)
    _code_model: ContextVar[str | None] = field(init=False, repr=False)
    _code_settings: ContextVar[Mapping[str, Any] | None] = field(init=False, repr=False)
    """The settings in force before this capability's patch, snapshotted for the baseline.

    Held as a plain mapping rather than `ModelSettings`: `RunContext.model_settings` is whatever the
    session is running with, which is `RealtimeModelSettings` in a realtime session, and the only
    thing done with it is handing it to the contract's baseline builder, which takes any mapping.
    Narrowing to one of the two would either cast away a real case or drop a realtime agent's
    baseline.
    """
    _code_tools: ContextVar[list[ToolsetTool[Any]] | None] = field(init=False, repr=False)
    _planned_issues: ContextVar[Mapping[Section, tuple[ApplyIssue, ...]]] = field(init=False, repr=False)
    """What the sections planned for the request being assembled, until it is reported.

    The four sections are applied by three different hooks, and the policy is applied to all of them
    at once, so the two that run first leave their issues here for the last one to report. Keyed by
    section and replaced rather than appended, because every section is planned afresh on every
    request and a toolset listed twice must not be reported twice.
    """

    def __post_init__(self) -> None:
        self._selection_resolved = ContextVar('agent_control_selection_resolved', default=None)
        self._code_model = ContextVar('agent_control_code_model', default=None)
        self._code_settings = ContextVar('agent_control_code_settings', default=None)
        self._code_tools = ContextVar('agent_control_code_tools', default=None)
        self._planned_issues = ContextVar('agent_control_planned_issues', default={})
        # The empty config is the only code-side default there is. The agent itself is what a
        # published value is layered onto, so a second place to say the same thing would only be
        # somewhere for the two to disagree.
        self._setup_variable(
            self.name,
            prefix=AGENT_VARIABLE_PREFIX,
            value_type=AgentConfig,
            default=AgentConfig(),
        )

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> AbstractCapability[AgentDepsT]:
        """Bind alongside the companion that carries the overriding half of this capability.

        See [`_AgentControlOverrides`][] for why the two halves cannot be one capability. The pair is
        assembled here rather than at construction so `AgentControl` stays the leaf a user builds,
        passes around, and finds with
        [`find_capability`][pydantic_ai.capabilities.find_capability] -- flattening a
        `CombinedCapability` keeps its children as siblings, so a container in this position would
        take the user's capability out of the tree entirely.
        """
        if self._deferred is not None and not agent.name:
            raise UserError(
                "`AgentControl` without an explicit `name` reads the agent's `name`, and this agent has none. "
                'Pydantic AI would infer one from the variable the agent is assigned to, so renaming that '
                'variable would silently point the agent at a different managed config -- which is why it is '
                'refused rather than inferred. Give the agent a `name=...`, or pass an explicit `name` to '
                '`AgentControl`.'
            )
        return CombinedCapability([self, _AgentControlOverrides(self)])

    def _resolve_for_selection(
        self, variable: Variable[AgentConfig], ctx: ModelSelectionContext[AgentDepsT]
    ) -> ResolvedVariable[AgentConfig]:
        """Resolve with the same callable targeting inputs that the authoritative run will reuse."""
        targeting_key = self.targeting_key(ctx) if callable(self.targeting_key) else self.targeting_key  # pyright: ignore[reportArgumentType]
        attributes = self.attributes(ctx) if callable(self.attributes) else self.attributes  # pyright: ignore[reportArgumentType]
        return variable.get(targeting_key=targeting_key, attributes=attributes, label=self.label)

    def _model_selector(self) -> Callable[[ModelSelectionContext[AgentDepsT]], ModelSelection]:
        """Build the per-run selector a nameless `get_model` returns.

        Pydantic AI evaluates a selector once per new request step, but the managed model is a
        run-stable config value, so the selector memoizes its first choice and every later step
        reuses it -- one resolve per run, matching the static (named) path rather than re-reading the
        variable each step. A fresh selector (and fresh memo) is built per `get_model` call, i.e. per
        run, so nothing leaks across runs. On the first evaluation it derives the backing variable
        from `ctx.agent` (the same derivation `wrap_run` performs), reads the managed model, and
        falls back to the model Pydantic AI already selected (`ctx.model`) when none is managed --
        raising only when there is no model at all (a nameless, model-less agent with nothing
        published yet), so the misconfiguration surfaces clearly instead of downstream.
        """
        selected: list[ModelSelection] = []

        def select(ctx: ModelSelectionContext[AgentDepsT]) -> ModelSelection:
            if not selected:
                code_model = ctx.model
                self._code_model.set(f'{code_model.system}:{code_model.model_name}' if code_model is not None else None)
                resolved = self._resolve_for_selection(self._ensure_variable_for_agent(ctx.agent), ctx)
                model = resolved.value.model
                if model is not None:
                    try:
                        infer_model(model)
                    except (UserError, ValueError) as error:
                        _warn_dropped(
                            f'Managed agent config selects unknown model {model!r} ({error}); keeping the code model.'
                        )
                        model = None
                if model is not None:
                    selected.append(model)
                elif ctx.model is not None:
                    selected.append(ctx.model)
                else:
                    raise UserError(
                        'A nameless `AgentControl` on a model-less agent has no model to run: the agent '
                        'defines no model and none is published in Logfire yet. Give the agent a model, '
                        'pass one to `run(model=...)`, or publish a `model` in the managed config.'
                    )
                # Handed off only once selection has succeeded. `wrap_run` is what clears this, and a
                # selection that raises never reaches it -- so setting it earlier would leave this run's
                # instructions, settings and tool overrides in the context for whatever runs next.
                self._selection_resolved.set(resolved)
            return selected[0]

        return select

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Wrap the agent toolset with managed LLM-facing definition overlays."""
        return _ToolDefinitionOverridesToolset(
            wrapped=toolset,
            get_config=self._current_config,
            observe_code_tools=self._observe_code_tools,
            plan_issues=self._plan_issues,
        )

    def _observe_code_tools(self, tools: list[ToolsetTool[Any]]) -> None:
        self._code_tools.set(tools)

    def _current_config(self) -> AgentConfig | None:
        """The active run's managed config, or `None` outside a resolved run."""
        resolved = self.resolved
        return None if resolved is None else resolved.value

    def _plan_issues(self, section: Section, issues: Sequence[ApplyIssue]) -> None:
        """Hold one section's issues until the request that produced them is fully planned."""
        self._planned_issues.set({**self._planned_issues.get(), section: tuple(issues)})

    def _report_planned(self) -> None:
        """Apply `on_unmatched` to everything this request planned and could not apply, once.

        Called once the last section has been planned, which is what the contract's apply helpers
        return their issues for: `'error'` fails on every entry the request would have got wrong
        rather than on whichever section happened to be planned first -- which used to mean the
        strictest policy reported the least. The order is the order the sections were planned, and
        they are cleared as they are read, so a report belongs to the request that produced it.
        """
        planned = self._planned_issues.get()
        self._planned_issues.set({})
        _report(self.on_unmatched, [issue for issues in planned.values() for issue in issues])

    def _warn_if_unreadable(self, variable: Variable[AgentConfig], resolved: ResolvedVariable[AgentConfig]) -> None:
        """Warn once per variable when its value could not have been read, as opposed to not published.

        Logfire reports both as `'code_default'`, so the reason alone cannot tell an agent with no
        config yet from one that will never see the config it has. The provider can:
        `get_variable_provider()` is what `Variable.get` resolved through, and on a configured instance
        with an API key it creates the remote provider on first call, so a
        [`NoOpVariableProvider`][logfire.variables.abstract.NoOpVariableProvider] here means no value
        could have arrived. The usual cause is a credential: Logfire variables need an API key
        with `project:read_variables`, which a write token is not, and `LOGFIRE_API_KEY` is only read
        by a configured instance. With the key in the environment and still no provider, the instance
        is the one that was never configured, typically because the application configured its own
        and this capability resolved on the default one.
        """
        if resolution_reason(resolved) != 'code_default':
            return
        if not isinstance(variable.logfire_instance.config.get_variable_provider(), NoOpVariableProvider):
            return
        on_default_instance = self.logfire_instance is None and not isinstance(self.name, Variable)
        if not os.environ.get('LOGFIRE_API_KEY'):
            cause = (
                'Reading Logfire variables needs an API key with the `project:read_variables` scope (a write token '
                'cannot read them), set as `LOGFIRE_API_KEY` or passed as `logfire.configure(api_key=...)`.'
            )
        elif on_default_instance:
            cause = (
                '`LOGFIRE_API_KEY` is set, but the default Logfire instance has not been configured, and only a '
                'configured instance reads it. Call `logfire.configure()`.'
            )
        else:
            cause = (
                '`LOGFIRE_API_KEY` is set, but the Logfire instance this capability was given has not been '
                'configured, and only a configured instance reads it. Pass the instance `logfire.configure()` '
                'returned.'
            )
        if on_default_instance:
            cause += (
                ' This capability resolves on the default Logfire instance: if your application configures its '
                'own, pass it as `AgentControl(logfire_instance=...)`.'
            )
        _warn_dropped(
            f'`AgentControl` cannot read {variable.name!r}: its Logfire instance has no variable provider, so the '
            f'agent runs as defined in code and nothing published in Logfire applies. {cause}'
        )

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Add applied-section baggage inside the base's once-per-run resolution context."""
        resolved = self._selection_resolved.get() or self._resolve(ctx)
        self._warn_if_unreadable(self._ensure_variable(ctx), resolved)
        with resolved:
            token = self._resolved.set(resolved)
            try:
                sections = ','.join(
                    name
                    for name in ('instructions', 'model', 'settings', 'tool_definitions')
                    if getattr(resolved.value, name) is not None
                )
                if sections:
                    with logfire.set_baggage(**{'logfire.managed.applied_sections': sections}):
                        return await handler()
                return await handler()
            finally:
                self._resolved.reset(token)
                self._selection_resolved.set(None)
                # Issues belong to the request that planned them. A run that failed between planning
                # a section and reporting it leaves some here, and the next run plans its own.
                self._planned_issues.set({})

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Report the agent's code baseline, read off the first request it assembles in this process.

        Runs on this outermost half deliberately: the snapshot has to describe the agent *before*
        managed values reach it, and the overriding half applies the managed instructions from the
        innermost position, after this. The request is left untouched here.
        """
        self._emit_config_hint(ctx, request_context)
        return request_context

    def _managed_settings(self, config: AgentConfig) -> ModelSettings:
        """The managed settings patch, in Pydantic AI's flat `ModelSettings` shape.

        Every field name in the section is already a `ModelSettings` key -- that is what makes them
        canonical -- so the contract's patch is the patch, and Pydantic AI merges it with the
        precedence `_AgentControlOverrides` exists to give it, and `_SUPPORT` is where that claim is
        declared rather than left to be inferred.

        This is also the one helper handed the whole config, so it is where the contract names a
        top-level key this release has no section for, alongside the settings keys it has no field
        for. Both come back rather than being reported here, and are reported with the rest of the
        request's.
        """
        applied = apply_settings(config, support=_SUPPORT)
        self._plan_issues('settings', applied.issues)
        # A `dict[str, Any]` of canonical keys is a `ModelSettings` by construction, which is a
        # `TypedDict` and therefore not something `isinstance` can narrow to.
        return cast(ModelSettings, applied.settings)

    def _apply_instructions(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Apply the managed `instructions` section to the blocks this request assembled.

        What an entry does -- add a block, replace one, drop one -- and which entries reached nothing
        are the contract's to decide. This maps the assembled parts onto the blocks it takes, and maps
        what comes back onto parts again: a block that passed through keeps the very part it came from,
        a replaced one keeps its part's `id`, `name`, and `dynamic` flag with the published text swapped
        in, and an added one becomes a new part. Pydantic AI sorts static blocks ahead of dynamic ones
        so a provider can cache the stable prefix, so carrying the flag over is what keeps an override
        from moving that boundary on every request.

        A dynamic block is deliberately not addressable, and the contract refuses it rather than
        applying it. A function that returns a constant is flagged dynamic all the same, which is the
        right call rather than a gap to close: `dynamic` is what decides which side of the provider's
        cache breakpoint a block sits on, and nothing about a function's shape says whether its text
        came from the run. An author who wants fixed text addressable says so on the part instead of on
        the function -- `InstructionPart(content=..., name='style')` is static by default and is
        accepted anywhere instructions are, a toolset's `get_instructions()` included.

        The new parameters are assigned onto the given context rather than returned on a
        `dataclasses.replace` copy of it, which would look tidier and be wrong: `ModelRequestContext`
        declares `model_id` and `streaming` as `init=False`, the agent graph sets both immediately
        before calling this hook, and `replace()` re-initializes them to `None`/`False`. Losing them
        costs a streamed run its streaming flag and a durable-execution worker the selection token it
        re-resolves an aliased model from. Reported upstream.
        """
        config = self._current_config()
        if config is None or not config.instructions:
            return request_context
        parameters = request_context.model_request_parameters
        parts = list(parameters.instruction_parts or [])
        blocks = _blocks(parts)
        applied = apply_instructions(blocks, _without_implicit_drops(config, blocks))
        # A block that passed through untouched is the object it went in as, so identity is what says
        # which part it came from. A replaced one is a copy carrying the same `id`, and only a
        # non-dynamic part can be replaced, so consuming those in order matches them up even when an
        # agent assembles several blocks under one key.
        by_block = {id(block): part for block, part in zip(blocks, parts)}
        replaceable: dict[str, list[InstructionPart]] = {}
        for block, part in zip(blocks, parts):
            if block.id is not None and not block.dynamic:
                replaceable.setdefault(block.id, []).append(part)
        applied_parts: list[InstructionPart] = []
        for block in applied.blocks:
            part = by_block.get(id(block))
            if part is not None:
                applied_parts.append(part)
            elif block.id is not None:
                applied_parts.append(replace(replaceable[block.id].pop(0), content=block.text))
            else:
                applied_parts.append(self._added_part(block.text, ctx))
        self._plan_issues('instructions', applied.issues)
        if applied_parts != parts:
            request_context.model_request_parameters = replace(parameters, instruction_parts=applied_parts)
        return request_context

    def _added_part(self, text: str, ctx: RunContext[AgentDepsT]) -> InstructionPart:
        """One added block as a part, rendered against `deps` when `render_template` is set."""
        if not self.render_template:
            return InstructionPart(content=text)
        return InstructionPart(content=TemplateStr[AgentDepsT](text).render(ctx.deps), dynamic=True)

    def _emit_config_hint(self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext) -> None:
        """Report what this agent says in code, once per process, on one span.

        This is how an agent gets onto the Agent Control page, and how it stays accurate once it is
        there. The SDK writes no variable: every agent reports its code baseline, and Logfire turns
        that into a config for an agent it has none for, or offers to refresh a stored baseline the
        code has moved on from. Reporting whether or not a config resolved is what makes the second
        half possible -- an agent that reported only while unconfigured would go quiet the moment
        someone configured it, and its stored baseline would describe the code as it was that day. It
        also matches what the platform has to do anyway: this SDK cannot tell an unknown variable from
        one with nothing published at this label, so deduplicating against the variables that exist is
        the platform's job either way.

        The span is named `agent_control_config_hint` and carries:

        - `agent_control.variable_name` -- the `agent__<key>` variable the config belongs in.
        - `agent_control.agent_name` -- the agent's `name` as written in code. The variable name is
          derived from it lossily, so this is what says *which* agent landed on that key. Left off
          when the capability was given a variable name of its own and the agent has no name at all:
          absent says "there is none", where a null-valued attribute would only raise the question of
          whether that is a name.
        - `agent_control.framework` -- which Agent Control SDK produced the hint; see `_FRAMEWORK`.
        - `agent_control.baseline_source` -- `'code'`, the contract's
          [`BaselineSource`][logfire.agent_control.BaselineSource] for a baseline read off the running
          code rather than observed from traffic.
        - `agent_control.schema_sha256` -- the contract schema this baseline was built against, so a
          consumer stores the matching JSON schema on the variable rather than guessing, and can tell
          a baseline from an older SDK from one it wrote the schema for.
        - `agent_control.baseline` -- the `AgentConfig` snapshot, as the JSON a variable's `example`
          holds. Absent when `agent_control.baseline_reduction` is `'omitted'`.
        - `agent_control.baseline_reduction` -- `'none'`, `'tool_definitions'`, or `'omitted'`: what
          the baseline had to give up to fit `_MAX_BASELINE_BYTES`. Always present, so a partial
          baseline is partial on the record rather than by inference.
        - `agent_control.baseline_bytes` -- the full baseline's UTF-8 size before any reduction.
        - `agent_control.baseline_sha256` -- the digest of the whole baseline; see `_baseline_sha256`.
          Present whatever `agent_control.baseline_reduction` says, and taken before any reduction, so
          two reports of the same code agree and two oversize reports of different code do not.
        - `agent_control.service_name`, `agent_control.environment`, `agent_control.service_version`
          -- which deployment reported it; see `_deployment_attributes`. Each is left off when the
          Logfire instance does not know it.
        - `agent_control.resolution_reason` -- why the run's variable resolved the way it did
          (`'resolved'`, `'code_default'`, ...). Since every agent reports, the span's existence no
          longer says whether one had a config, and this is the fact a consumer needs to tell a
          baseline that wants a config created from it from one that may only be refreshing a stale
          `example`. It is the same string the capability's `resolved` exposes, read where the hint
          is already built, so the platform learns it without a second query.

        Attributes rather than one nested blob because the hint is a contract with a consumer that
        queries it: a name it filters on and a size it can threshold have to be columns, and the
        baseline is the only one of them that is a document. They are prefixed with the capability's
        name the way every other harness capability's are.

        Emitted on the variable's own Logfire instance rather than on `ctx.tracer`, which is where
        this capability's operational telemetry would go. A hint is addressed to the project that
        would hold the config -- a process serving two projects has to reach each -- and it must not
        depend on core instrumentation being active to arrive. Emitted as a span rather than a log
        record for one more reason: a log below the configured `min_level` is dropped, and a signal
        the platform contract depends on cannot be something a logging setting silently withholds.

        The baseline quotes the agent's own instructions and tool descriptions, and is *not* held
        back by `trace_include_content`. What that flag governs is run content -- prompts, tool
        arguments, outputs -- and a dynamic instruction block contributes its seam and never its
        rendering. Tool definitions come from the assembled request, however, so a dynamic toolset's
        model-facing names and descriptions must not contain request data or secrets. Settings are
        reduced to the canonical keys, which keeps `extra_headers` and `extra_body` out. What remains
        is the model-facing agent definition, and an agent wired to Agent Control is one whose author
        asked for that text to be editable in this Logfire project.
        """
        resolved = self.resolved
        if resolved is None:
            return
        # The agent's own variable, not a shared attribute: a nameless capability backs one variable
        # per agent, so the hint has to name the one this run resolved.
        variable = self._ensure_variable(ctx)
        config = variable.logfire_instance.config
        agent = ctx.agent
        agent_name = agent.name if agent is not None and agent.name else None
        hint_identity = (variable.name, agent_name)
        with _config_hint_lock:
            _, reported = _config_hint_emitted.setdefault(id(config), (config, set()))
            if hint_identity in reported:
                return
            reported.add(hint_identity)
        # Nothing about describing the agent raises: code-side text the contract cannot hold -- a
        # block past the length bound, a setting it has no word for -- is left out of the baseline and
        # warned about, so an agent whose own prompt is too big to describe keeps making requests.
        code_baseline = self._code_baseline(request_context)
        baseline, reduction, size = _serialize_baseline(code_baseline)
        attributes: dict[str, Any] = {
            'agent_control.variable_name': variable.name,
            'agent_control.framework': _FRAMEWORK,
            'agent_control.baseline_source': 'code',
            'agent_control.schema_sha256': SCHEMA_SHA256,
            'agent_control.baseline_sha256': _baseline_sha256(code_baseline),
            'agent_control.baseline_reduction': reduction,
            'agent_control.baseline_bytes': size,
            'agent_control.resolution_reason': resolution_reason(resolved),
            **_deployment_attributes(variable.logfire_instance),
        }
        if agent_name is not None:
            attributes['agent_control.agent_name'] = agent_name
        if baseline is not None:
            attributes['agent_control.baseline'] = baseline
        # A decision that took no time, so the span opens and closes on the spot.
        with variable.logfire_instance.span(_CONFIG_HINT_MESSAGE, _span_name=_CONFIG_HINT_SPAN_NAME, **attributes):
            pass

    def _code_baseline(self, request_context: ModelRequestContext) -> AgentConfig:
        """The agent as written: what it would do with `AgentControl` removed.

        Instructions come straight off the assembled parts, which at this point carry nothing managed:
        this half runs before the overriding one, which is where every managed block is applied. Model,
        settings, and tool definitions were captured at their override sites, before managed behavior
        replaced them.

        What a baseline may say about each of them is the contract's rule rather than this adapter's:
        a dynamic block contributes its seam and never its rendered text, every top-level parameter is
        listed so an undocumented one can be described from Logfire, and the run's settings are reduced
        to the canonical keys -- which is what keeps `extra_headers` and `extra_body`, where
        authorization headers and signed bodies live, out of a variable every project member can read.
        """
        return build_baseline(
            instructions=_blocks(request_context.model_request_parameters.instruction_parts or []),
            model=self._code_model.get(),
            settings=self._code_settings.get(),
            tools=[
                ToolDef(
                    name=tool.tool_def.name,
                    description=tool.tool_def.description,
                    parameters_json_schema=tool.tool_def.parameters_json_schema,
                    toolset=_toolset_key(tool.toolset),
                )
                for tool in self._code_tools.get() or []
            ],
        )


@dataclass
class _AgentControlOverrides(AbstractCapability[AgentDepsT]):
    """The half of [`AgentControl`][pydantic_ai_harness.logfire.AgentControl] that has to win.

    A capability's [`CapabilityOrdering`][pydantic_ai.capabilities.CapabilityOrdering] position
    answers two questions at once: how its hooks nest around the run, and where its contributions
    land when Pydantic AI merges them. `AgentControl` needs opposite answers. Its resolution's label
    and version baggage has to envelop the whole run including the run span, which means `outermost`
    -- and `outermost` contributions are merged *first*, so every other capability's model and
    settings would be merged over the published ones. A managed value is an override, so the
    contributions that have to beat other capabilities live here instead, pinned `innermost` where
    they are merged last and win whatever order the capabilities were registered in.

    That is why this is a separate capability rather than a few more methods on `AgentControl`, and
    why it is *not* what a user constructs: `AgentControl` stays the leaf they build and find, and
    [`for_agent`][pydantic_ai_harness.logfire.AgentControl.for_agent] stitches this alongside it. The
    two share one object's state by reference, so there is still one variable, one resolution per
    run, and one thing to configure. Decoupling ordering from precedence in the framework itself is
    tracked in [#7420](https://github.com/pydantic/pydantic-ai/issues/7420); until then this split is
    what gives a published value the precedence the Logfire UI promises.
    """

    control: AgentControl[AgentDepsT] = field(repr=False, compare=False)
    """The capability whose state this contributes from; never a copy, so one resolution serves both."""

    def get_ordering(self) -> CapabilityOrdering:
        """Merge last, so a published model or settings beats every other capability's."""
        return CapabilityOrdering(position='innermost')

    def get_model_settings(self) -> AgentModelSettings[AgentDepsT] | None:
        """Contribute the lowered managed settings patch for each model request.

        A published key the contract has no field for is planned here, where the patch is applied,
        rather than from validation; see `_report` for why.
        """

        def model_settings(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            self.control._code_settings.set(dict(ctx.model_settings or {}))  # pyright: ignore[reportPrivateUsage]
            config = self.control._current_config()  # pyright: ignore[reportPrivateUsage]
            if config is None:
                return ModelSettings()
            return self.control._managed_settings(config)  # pyright: ignore[reportPrivateUsage]

        return model_settings

    def get_model(self) -> AgentModel[AgentDepsT] | None:
        """Source the managed model with the right precedence (`run(model=...)` > managed > everything else).

        The selector records the lower-precedence code model, resolves static or callable targeting
        inputs, and saves that exact resolution for `wrap_run`. This keeps model selection consistent
        with the config and telemetry used by the run even when a targeting callable is stateful. The
        selector does not enter the resolution as a context manager; `wrap_run` remains the owner of
        its baggage. A fully model-less agent can be driven from Logfire, and a call-site
        `run(model=...)` still wins. An unknown managed model warns and falls back to the code model.
        """
        return self.control._model_selector()  # pyright: ignore[reportPrivateUsage]

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Apply the managed instruction blocks, then report what this request could not apply.

        Runs from the innermost position for the same reason the model and settings are contributed
        from here: an override another capability's hook could overwrite afterwards is not an
        override. Being last also means every contribution has been assembled, which is what lets it
        reach text no capability owns -- the agent's own literal, a toolset's, an MCP server's -- and
        what lets an added block land at the end of the assembled static prompt.

        It is the request's last planning step for the same reason, which is why reporting happens
        here: every section has had its say by now, so `on_unmatched='error'` fails on all of it.
        """
        control = self.control
        request_context = control._apply_instructions(ctx, request_context)  # pyright: ignore[reportPrivateUsage]
        control._report_planned()  # pyright: ignore[reportPrivateUsage]
        return request_context
