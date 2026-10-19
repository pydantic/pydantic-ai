"""The pure half of Agent Control: what a published `AgentConfig` does to an agent's request.

Everything here takes plain data in and gives plain data back -- no Logfire, no network, no
framework. An adapter enumerates what its framework is about to send, hands it to these functions
along with the resolved config, and sends back what they return. That is the whole contract, and it
is why an adapter is a hook plus twenty lines rather than a reimplementation of these semantics.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias

from pydantic import ValidationError

from ._config import (
    SETTING_ADAPTERS,
    AgentConfig,
    AgentConfigSettings,
    InstructionBlock,
    InstructionText,
    ParameterOverride,
    ToolDefinitionOverride,
    ToolKey,
    describe_tool_key,
    first_by_key,
    instruction_blocks,
)
from ._reporting import ApplyIssue, warn_dropped
from ._schema import MAX_MODEL_FACING_TEXT_LENGTH
from ._support import AgentSupport, Section
from ._units import MAX_TIMEOUT_SECONDS, is_representable_timeout


@dataclass(frozen=True)
class Block:
    """One instruction block an agent is about to send, as the adapter sees it.

    The neutral shape of "part of a prompt": some text, the `id` a published config addresses it by,
    and whether the framework recomputes it per request. Frameworks that assemble their prompt from
    several sources map each source to a block; frameworks with one prompt string produce one block
    (`id='agent'` by convention, the only id the contract reserves).
    """

    text: str
    """The block's text, exactly as the model would be shown it."""
    id: str | None = None
    """The key a published entry addresses this block by, or `None` if nothing can address it.

    A block with no `id` is not a defect: an anonymous callable prompt genuinely has nothing stable
    to name it by, and publishing it as unaddressable is how the editor knows not to offer an
    override for it.
    """
    dynamic: bool = False
    """Whether the framework recomputes this block on every request.

    A dynamic block cannot be overridden -- see `apply_instructions` -- and its text never leaves the
    process in a baseline, because it is this request's rendering of whatever the run carried.
    """


@dataclass(frozen=True)
class ToolDef:
    """One tool as the model is shown it, which is the only part Agent Control edits.

    Implementation, argument validation, and the parameter names and types stay code-owned; an
    adapter builds a `ToolDef` from its framework's tool, applies what comes back, and never lets a
    published value near what the tool actually does.
    """

    name: str
    """The tool's code-side name, which is what an override names it by."""
    description: str | None = None
    """The description shown to the model."""
    parameters_json_schema: dict[str, Any] = field(default_factory=dict[str, Any])
    """The tool's JSON Schema, of which only top-level parameter descriptions are ever patched."""
    toolset: str | None = None
    """Where the tool came from -- a toolset id, an MCP server name -- or `None` when the framework
    has no such notion. The baseline reports it so the UI can group a long list; an override that
    sets it is narrowed to that group's tool of this `name`."""


@dataclass(frozen=True)
class AppliedInstructions:
    """What `apply_instructions` returns: the blocks to send, and what reached nothing."""

    blocks: list[Block]
    """The blocks to send, with published text swapped in, removed, or added."""
    issues: Sequence[ApplyIssue]
    """Every published instruction entry this request did not apply; see `ApplyIssue`.

    Reported by nothing here. Hand these to
    `report_issues`, with the other sections'
    issues, and the policy the user configured is applied to all of it at once.
    """


CollisionScope: TypeAlias = Literal['global', 'toolset']
"""Where two advertised tool names actually collide, which is a property of the framework.

A rename is only safe if the core can tell whether the new name is already taken, and "taken" is not
the same question everywhere:

- `'global'` (the default): every tool is advertised into one flat namespace, so any two tools with
  the same advertised name collide. Almost every framework -- OpenAI Agents, LangChain, ADK, LiveKit,
  the AI SDK -- works this way.
- `'toolset'`: the runtime name a tool is advertised under already carries its toolset, as the Claude
  Agent SDK's `mcp__<server>__<tool>` does, so two servers may each advertise a `search`, and
  renaming one of them to `lookup` does not collide with another server's `lookup`.

Getting this wrong is not cosmetic in either direction: too wide drops a legal rename, too narrow
advertises two tools under one name and makes a call unroutable.
"""


@dataclass(frozen=True)
class AppliedTools:
    """What `apply_tool_definitions` returns: the tools to advertise, and where a call comes back to."""

    tools: list[ToolDef]
    """The tools in their original order, with managed names and descriptions applied."""
    routes: dict[str, str]
    """Advertised name -> code-side name, for every advertised tool.

    Total rather than rename-only so a dispatcher can look up any name the model calls without
    special-casing the tools that were not renamed. Frameworks differ on whether the implementation
    should see its old name or the new one, so the mapping is handed over rather than applied here.

    Flat, and therefore exact only under `collision_scope='global'`, where an advertised name
    identifies a tool by itself. An adapter that passes `'toolset'` has said that two toolsets may
    advertise the same name, so it routes through `reverse` instead; this mapping keeps the first of
    such a pair.
    """
    forward: dict[ToolKey, str]
    """`(toolset, code-side name)` -> the name that tool is advertised under.

    The direction an adapter needs on the way *out*: rewriting a `tool_choice`, an `active_tools`
    list, or a replayed history entry that names a tool by the name the code gave it. Keyed on the
    pair rather than the bare name so a framework with toolsets rewrites exactly the one it means.
    """
    reverse: dict[ToolKey, str]
    """`(toolset, advertised name)` -> the tool's code-side name.

    The direction an adapter needs on the way *in*: a call arrives under an advertised name, and
    under `collision_scope='toolset'` the adapter also knows which toolset it came from, which is
    what keeps two servers' identically named tools distinguishable.
    """
    issues: Sequence[ApplyIssue]
    """Every published tool entry, or part of one, this request did not apply; see `ApplyIssue`.

    Reported by nothing here; hand them to
    `report_issues` with the rest.
    """


def _oversized(text: str) -> bool:
    """Whether text exceeds the per-request budget, counted the way both cores agree to count it.

    In Unicode code points, which is what `len` is here and what `[...text].length` is in JavaScript
    -- deliberately not JavaScript's `String.length`, which counts UTF-16 units and would give an
    emoji or a CJK extension character twice the weight of the same text in Python, so the same
    published value would fit one core's budget and not the other's.
    """
    return len(text) > MAX_MODEL_FACING_TEXT_LENGTH


def _oversized_entry(text: str, *, instruction_id: str | None = None) -> ApplyIssue:
    """One entry whose text is past the budget, refused rather than truncated."""
    where = f'for instruction block {instruction_id!r} ' if instruction_id is not None else ''
    return ApplyIssue(
        section='instructions',
        reason='oversized-text',
        instruction_id=instruction_id,
        message=(
            f'Managed agent config publishes instruction text {where}of {len(text)} characters, past the '
            f'{MAX_MODEL_FACING_TEXT_LENGTH}-character limit on what one managed entry may add to every model '
            'request; that entry is not applied.'
        ),
    )


def apply_instructions(blocks: Sequence[Block], config: AgentConfig) -> AppliedInstructions:
    """Apply the `instructions` section to the blocks an agent is about to send.

    An entry with an `id` replaces that block's text, or removes the block when its text is `None`.
    An entry with no `id` adds a block. Order and `dynamic` flags of untouched blocks never change,
    and a replaced block keeps both: frameworks group static text ahead of per-request text so a
    provider can cache the stable prefix, so re-ordering or re-flagging a block on the way through
    would move that cache boundary on every request in exchange for nothing.

    Added blocks land at the end of the leading run of static blocks -- after the last static block
    the agent assembles, before the first dynamic one. That is where a managed addition is both last
    in the prompt as written and still inside the cacheable prefix; appending it after per-request
    text would put published text in the volatile tail instead.

    A dynamic block is deliberately not addressable. Its text is recomputed per request from whatever
    the run carries, so replacing it pins one rendering forever and dropping it removes the
    computation -- either way the block stops doing the thing it was written to do, and nothing about
    the managed value says which. Addressing one is refused and reported rather than applied.

    Four decisions come back as `ApplyIssue`s on `issues`, and
    are reported by nothing here: an `id` no block carries (`'unknown-id'`), an `id` only a dynamic
    block carries (`'dynamic-id'`), text past the contract's per-request budget
    (`'oversized-text'`), and the second and later entries naming one `id` (`'duplicate-entry'`).
    Pass them to `report_issues` along with the
    other sections'. The first is decided per request, because the blocks are this request's: an
    agent whose prompt varies with its input can carry a block on one request and not the next.
    """
    entries = instruction_blocks(config)
    overrides, duplicates = first_by_key(
        [(entry.id, entry.instructions) for entry in entries if entry.id is not None],
        lambda key: f'instruction id {key!r}',
    )
    issues: list[ApplyIssue] = [
        ApplyIssue(section='instructions', reason='duplicate-entry', instruction_id=key, message=message)
        for key, message in duplicates
    ]
    added = [entry.instructions for entry in entries if entry.id is None and entry.instructions is not None]
    if not overrides and not added:
        return AppliedInstructions(blocks=list(blocks), issues=issues)

    matched: set[str] = set()
    result: list[Block] = []
    for block in blocks:
        if block.id is None or block.id not in overrides:
            result.append(block)
            continue
        # Addressed, and refused for a reason of its own: it is reported as the dynamic case and not
        # again as a key nothing carries.
        matched.add(block.id)
        if block.dynamic:
            issues.append(
                ApplyIssue(
                    section='instructions',
                    reason='dynamic-id',
                    instruction_id=block.id,
                    message=(
                        f'Managed agent config addresses instruction block {block.id!r}, which the agent '
                        'recomputes per request; a managed value would pin or remove that computation, so it is '
                        'not applied and the block keeps what the code produces.'
                    ),
                )
            )
            result.append(block)
            continue
        replacement = overrides[block.id]
        if replacement is None:
            continue
        if _oversized(replacement):
            issues.append(_oversized_entry(replacement, instruction_id=block.id))
            result.append(block)
            continue
        result.append(replace(block, text=replacement))
    for key in overrides:
        if key not in matched:
            issues.append(
                ApplyIssue(
                    section='instructions',
                    reason='unknown-id',
                    instruction_id=key,
                    message=(
                        f'Managed agent config addresses instruction block {key!r}, which this request does not '
                        'assemble; that entry applies to nothing.'
                    ),
                )
            )

    additions: list[str] = []
    for text in added:
        if _oversized(text):
            issues.append(_oversized_entry(text))
        else:
            additions.append(text)
    if additions:
        boundary = next((index for index, block in enumerate(result) if block.dynamic), len(result))
        result[boundary:boundary] = [Block(text=text) for text in additions]
    return AppliedInstructions(blocks=result, issues=issues)


def _parameter_entry(
    reason: Literal['unknown-parameter', 'no-patchable-schema'], tool: ToolDef, parameter: str, because: str
) -> ApplyIssue:
    return ApplyIssue(
        section='tool_definitions',
        reason=reason,
        toolset=tool.toolset,
        tool=tool.name,
        parameter=parameter,
        message=(
            f'Managed agent config patches parameter {parameter!r} of {describe_tool_key((tool.toolset, tool.name))}, '
            f'which {because}; that patch applies to nothing.'
        ),
    )


def _with_parameters(
    tool: ToolDef, parameters: Mapping[str, ParameterOverride]
) -> tuple[dict[str, Any], list[ApplyIssue]]:
    """Patch top-level parameter descriptions while preserving all schema structure.

    Reports each patch that reached nothing, which is the half that used to be silent: from the
    Logfire UI, a patch naming a parameter the tool does not have looked exactly like one that
    applied.
    """
    parameters_json_schema = tool.parameters_json_schema
    issues: list[ApplyIssue] = []
    properties = parameters_json_schema.get('properties')
    if not isinstance(properties, dict):
        for name in parameters:
            issues.append(_parameter_entry('no-patchable-schema', tool, name, 'has no top-level parameters'))
        return parameters_json_schema, issues
    typed_properties: dict[str, Any] = parameters_json_schema['properties']
    new_properties: dict[str, Any] = {}
    changed = False
    for name, schema in typed_properties.items():
        override = parameters.get(name)
        if override is not None and override.description is not None and isinstance(schema, dict):
            parameter_schema: dict[str, Any] = typed_properties[name]
            new_properties[name] = {**parameter_schema, 'description': override.description}
            changed = True
        else:
            new_properties[name] = schema
    for name, override in parameters.items():
        if name not in typed_properties:
            issues.append(_parameter_entry('unknown-parameter', tool, name, 'has no parameter of that name'))
        elif override.description is not None and not isinstance(typed_properties[name], dict):
            issues.append(
                _parameter_entry('no-patchable-schema', tool, name, 'describes that parameter with no schema object')
            )
    if not changed:
        return parameters_json_schema, issues
    return {**parameters_json_schema, 'properties': new_properties}, issues


def _apply_override(tool: ToolDef, override: ToolDefinitionOverride) -> tuple[ToolDef, list[ApplyIssue]]:
    """Apply the LLM-facing parts of an override, returning the original definition for a no-op."""
    changes: dict[str, Any] = {}
    issues: list[ApplyIssue] = []
    if override.new_name is not None and override.new_name != tool.name:
        changes['name'] = override.new_name
    if override.description is not None and override.description != tool.description:
        changes['description'] = override.description
    if override.parameters:
        schema, issues = _with_parameters(tool, override.parameters)
        if schema is not tool.parameters_json_schema:
            changes['parameters_json_schema'] = schema
    return (replace(tool, **changes) if changes else tool), issues


def _namespace(tool: ToolDef, scope: CollisionScope) -> str | None:
    """The set of advertised names this tool competes in; see `CollisionScope`."""
    return tool.toolset if scope == 'toolset' else None


def apply_tool_definitions(
    tools: Sequence[ToolDef],
    config: AgentConfig,
    *,
    reserved: Collection[str] = (),
    collision_scope: CollisionScope = 'global',
) -> AppliedTools:
    """Apply the `tool_definitions` section to the tools an agent is about to advertise.

    An override is matched on the tool's code-side `name`, narrowed to one group when it sets
    `toolset` too. An override narrowed to a group beats one that only names the tool, so a config can
    say "every `search`" and "the CRM's `search`" at once and the specific entry wins where both
    apply. The unqualified entry is then outranked, not unmatched: it still reached the tool it named,
    so only an entry that matched no tool at all is reported. That is decided per call, because tool
    availability is dynamic -- an agent can advertise different tools from one step to the next -- and
    a report from one listing is a report about that listing.

    A rename onto a name that is already taken is dropped while that override's other patches still
    apply, so every tool keeps a name the model can call. What counts as taken is `collision_scope`
    plus `reserved`, and renames resolve in input order, so the tool that was there first keeps the
    name.

    Parameter descriptions are patched in place and every other piece of schema structure is
    preserved. A patch on a parameter the tool does not have, or on one whose schema is not an object
    to patch a description into, applies nothing -- a parameter is part of the tool's code-defined
    shape, and the baseline is what says which ones exist -- and comes back on `issues` rather than
    being dropped in silence.

    Every decision it makes -- an override no tool matched, a patch on a parameter that is not there,
    a dropped rename, the second entry naming one tool -- comes back as an
    `ApplyIssue` and is reported by nothing here. Hand them to
    `report_issues` with the other sections'.

    Args:
        tools: The tools the agent is about to advertise, in the order it would advertise them.
        config: The resolved managed config.
        reserved: Advertised names this adapter needs kept free -- an OpenAI Agents handoff, a
            provider-side tool the framework adds after this call, anything outside the editable
            list. A rename onto one of them is refused like any other collision. Read as taken in
            every namespace, which is exact for a framework with one and deliberately conservative
            for a framework with several.
        collision_scope: Where two advertised names collide; see
            `CollisionScope`.

    Returns:
        The tools to advertise and the routing maps for them; see
        `AppliedTools`.
    """
    overrides, duplicates = first_by_key(
        [((override.toolset, override.name), override) for override in config.tool_definitions or []],
        describe_tool_key,
    )
    reserved_names = frozenset(reserved)
    # A namespace is the set of advertised names that compete: one for the whole request, or one per
    # toolset. Every code-side name starts out taken, so a rename onto a tool later in the list is a
    # collision rather than a name that is free until that tool is reached.
    taken: dict[str | None, set[str]] = {}
    for tool in tools:
        taken.setdefault(_namespace(tool, collision_scope), set()).add(tool.name)

    issues: list[ApplyIssue] = [
        ApplyIssue(section='tool_definitions', reason='duplicate-entry', toolset=key[0], tool=key[1], message=message)
        for key, message in duplicates
    ]
    matched: set[ToolKey] = set()
    applied: list[ToolDef] = []
    routes: dict[str, str] = {}
    forward: dict[ToolKey, str] = {}
    reverse: dict[ToolKey, str] = {}
    for tool in tools:
        qualified: ToolKey = (tool.toolset, tool.name)
        unqualified: ToolKey = (None, tool.name)
        matched.update(key for key in (qualified, unqualified) if key in overrides)
        override = overrides.get(qualified, overrides.get(unqualified))
        if override is None:
            new_tool = tool
        else:
            new_tool, parameter_entries = _apply_override(tool, override)
            issues.extend(parameter_entries)
        names = taken[_namespace(tool, collision_scope)]
        if new_tool.name != tool.name and (new_tool.name in names or new_tool.name in reserved_names):
            issues.append(
                ApplyIssue(
                    section='tool_definitions',
                    reason='rename-collision',
                    toolset=tool.toolset,
                    tool=tool.name,
                    message=(
                        f'Managed tool definition override renames {tool.name!r} to {new_tool.name!r}, which is '
                        f'already advertised by another tool; keeping the original name {tool.name!r}.'
                    ),
                )
            )
            new_tool = replace(new_tool, name=tool.name)
        else:
            names.add(new_tool.name)
        applied.append(new_tool)
        routes.setdefault(new_tool.name, tool.name)
        forward[qualified] = new_tool.name
        reverse[(tool.toolset, new_tool.name)] = tool.name
    for key in overrides:
        if key not in matched:
            issues.append(
                ApplyIssue(
                    section='tool_definitions',
                    reason='unknown-tool',
                    toolset=key[0],
                    tool=key[1],
                    message=(
                        f'Managed agent config patches {describe_tool_key(key)}, which no toolset advertises for '
                        'this request; that override applies to nothing.'
                    ),
                )
            )
    return AppliedTools(
        tools=applied,
        routes=routes,
        forward=forward,
        reverse=reverse,
        issues=issues,
    )


def _unrepresentable_timeout_message(timeout: float) -> str:
    return (
        f'Managed agent config sets a request timeout of {timeout!r} seconds, which is not a budget a request can '
        f'be given -- it has to be finite, not negative, and no larger than {MAX_TIMEOUT_SECONDS} seconds; that key '
        'is not applied.'
    )


def _section_issues(config: AgentConfig, support: AgentSupport | None) -> list[ApplyIssue]:
    """What this release, and this adapter, cannot do with the sections a value carries.

    Two decisions about the config as a whole rather than about any one entry, which is why they are
    here rather than in one section's helper: a top-level key this release has no section for, and a
    section it does have and the adapter declared it cannot apply.

    The first is the other half of the openness that makes a future `mcp_servers` or `skills` section
    readable by an older SDK. Unknown keys are ignored on purpose -- refusing them would make the day
    a section is added the day every older SDK stops resolving -- but an ignored key that nobody
    hears about is how the first person to publish one gets a silently degraded agent.
    """
    issues = [
        ApplyIssue(
            section=name,
            reason='unknown-section',
            message=(
                f'Managed agent config publishes a {name!r} section, which this version of the Agent Control '
                'contract has no section for; that section is not applied.'
            ),
        )
        for name in config.unrecognized
    ]
    if support is None:
        return issues
    published: tuple[tuple[Section, Any], ...] = (
        ('instructions', config.instructions),
        ('model', config.model),
        ('settings', config.settings),
        ('tool_definitions', config.tool_definitions),
    )
    issues.extend(
        ApplyIssue(
            section=name,
            reason='unsupported-section',
            message=(
                f'Managed agent config publishes a {name!r} section, which this agent framework has no way to '
                'apply; that section is not applied.'
            ),
        )
        for name, value in published
        if value is not None and name not in support.sections
    )
    return issues


@dataclass(frozen=True)
class AppliedSettings:
    """What `apply_settings` returns: the patch to merge, and what reached nothing."""

    settings: dict[str, Any]
    """The canonical settings patch: every key that was published and can be applied."""
    issues: Sequence[ApplyIssue]
    """Every published key, and every whole section, this request did not apply; see `ApplyIssue`.

    Reported by nothing here; hand them to
    `report_issues` with the rest. The shape is
    the same as the other two sections' so an adapter treats all three alike, which is what it could
    not do while this one returned a bare `dict`.
    """


def apply_settings(config: AgentConfig, *, support: AgentSupport | None = None) -> AppliedSettings:
    """The `settings` section as a patch to merge over the agent's own settings.

    Only explicitly set, non-`None` fields are emitted, and every key is already a canonical
    `AgentConfigSettings` name, so `settings` is the patch: merge it over the framework's settings,
    let a per-call setting outrank it, and leave every key it does not mention alone.

    Five kinds of published thing come back on `issues` rather than being silently dropped, all for
    the same reason -- someone published something and the agent did not apply it, which is a gap
    between what Logfire shows and what the agent does:

    - a key this version of the contract has no field for (`'unknown-setting'`), which a newer
      Logfire UI can write;
    - a key this adapter cannot lower into its framework (`'unsupported-setting'`), which it names
      through `AgentSupport.settings`. Leaving
      `support` as `None` says the adapter applies all of them;
    - a `timeout` that is not a representable request budget (`'unrepresentable-timeout'`) --
      negative, not finite, or past
      `MAX_TIMEOUT_SECONDS`. Dropped rather than
      clamped, because clamping turns "no real limit" into a deadline nobody published, and rounding
      a negative one to `0` cancels the request before it is sent;
    - a top-level key this release has no section for (`'unknown-section'`);
    - a section this adapter declared it cannot apply (`'unsupported-section'`).

    The last two are about the whole config rather than about settings, and they are here because
    this is the helper every adapter can call with the whole config and its own support declaration,
    whatever hooks its framework gives it: an adapter that never calls
    `apply_instructions` would otherwise have nowhere to learn that an `instructions` section was
    published at an agent that cannot apply one.

    An adapter that declares it cannot apply the `settings` section at all gets an empty patch and
    the one `'unsupported-section'` issue, whatever its `AgentSupport.settings` says.

    Args:
        config: The resolved managed config.
        support: What this adapter can apply; see
            `AgentSupport`. `None` says it can apply every
            section and every canonical setting, which is the right answer for an adapter that has
            not been taught to declare yet and the wrong one for any that has.
    """
    issues = _section_issues(config, support)
    settings = config.settings
    if settings is None or (support is not None and 'settings' not in support.sections):
        # An adapter that declared it cannot apply the section has already been told so, once, by
        # `_section_issues`. Handing it the patch anyway would be this function contradicting that
        # declaration, and reporting every key in it a second time would bury the one report that
        # matters under a list of keys the adapter was never going to reach.
        return AppliedSettings(settings={}, issues=issues)
    issues.extend(
        ApplyIssue(
            section='settings',
            reason='unknown-setting',
            setting=name,
            message=(
                f'Managed agent config sets {name!r}, which this version of the Agent Control contract has no '
                'model setting for; that key is not applied.'
            ),
        )
        for name in settings.unrecognized
    )
    applied: dict[str, Any] = {}
    # One pass in the canonical field order, so an issue's place in the list is the same in both
    # cores: judging the timeout in a pass of its own would put it ahead of keys the adapter cannot
    # lower here and behind them in TypeScript, for the same published value.
    for name, value in settings.model_dump(exclude_none=True).items():
        if name == 'timeout' and not is_representable_timeout(value):
            issues.append(
                ApplyIssue(
                    section='settings',
                    reason='unrepresentable-timeout',
                    setting='timeout',
                    message=_unrepresentable_timeout_message(value),
                )
            )
            continue
        if support is not None and name not in support.settings:
            issues.append(
                ApplyIssue(
                    section='settings',
                    reason='unsupported-setting',
                    setting=name,
                    message=(
                        f'Managed agent config sets {name!r}, which this agent framework has no equivalent for; '
                        'that key is not applied.'
                    ),
                )
            )
            continue
        applied[name] = value
    return AppliedSettings(settings=applied, issues=issues)


def canonical_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """A framework's own settings reduced to the contract's canonical keys and describable values.

    What a baseline may say about the code side, and nothing more. Two filters, and both are
    load-bearing rather than tidiness:

    - **Keys.** A framework's settings also carry provider-specific ones and `extra_headers` /
      `extra_body` -- exactly where authorization headers and signed bodies live -- and a baseline is
      published to a variable every member of the Logfire project can read.
    - **Values.** A value the contract has a key for but cannot hold is left out and warned about,
      never approximated. A reasoning effort a framework spells `'max'` is not `'xhigh'`, and
      publishing it as one describes the code as doing something it does not do, in the one artifact
      the Logfire editor presents as the truth about the code. A timeout outside the representable
      range goes the same way.

    Values are read leniently, deliberately unlike a published value: these come from a framework's
    own objects rather than from JSON, so an `int` where the contract wants a `float` and a tuple of
    stop sequences both describe the agent perfectly well, while the same shapes arriving over the
    wire would be a UI writing something the stored schema says it may not.
    """
    canonical: dict[str, Any] = {}
    for name, value in settings.items():
        adapter = SETTING_ADAPTERS.get(name)
        if value is None or adapter is None:
            continue
        try:
            canonical_value = adapter.validate_python(value)
        except ValidationError:
            warn_dropped(
                f'The agent runs with {name}={value!r}, which the Agent Control contract cannot describe; '
                'leaving it out of the published baseline.'
            )
            continue
        # Judged after coercion, on the number that would actually be published: a framework holding
        # its timeout as `'-1'` validates to `-1.0`, and checking the raw value would let a budget the
        # contract refuses through on the grounds that it was not a number yet.
        if name == 'timeout' and not is_representable_timeout(canonical_value):
            warn_dropped(
                f'The agent runs with a request timeout of {value!r} seconds, which the Agent Control contract '
                f'cannot describe -- it has to be finite, not negative, and no larger than {MAX_TIMEOUT_SECONDS} '
                'seconds; leaving it out of the published baseline.'
            )
            continue
        canonical[name] = canonical_value
    return canonical


def _baseline_parameters(tool: ToolDef) -> dict[str, ParameterOverride]:
    """Every top-level parameter the editor may describe, documented in code or not."""
    properties = tool.parameters_json_schema.get('properties')
    if not isinstance(properties, dict):
        return {}
    typed_properties: dict[str, Any] = tool.parameters_json_schema['properties']
    parameters: dict[str, ParameterOverride] = {}
    for name in typed_properties:
        description: Any = None
        if isinstance(typed_properties[name], dict):
            parameter_schema: dict[str, Any] = typed_properties[name]
            description = parameter_schema.get('description')
        parameters[name] = ParameterOverride(description=description if isinstance(description, str) else None)
    return parameters


def build_baseline(
    *,
    instructions: Sequence[Block] = (),
    model: str | None = None,
    settings: Mapping[str, Any] | None = None,
    tools: Sequence[ToolDef] = (),
) -> AgentConfig:
    """Describe the agent as written: what it does with Agent Control removed.

    This is what the config hint reports to Logfire for the editor to render as the code baseline to
    diff managed values against. It uses the same fields as a published value to say *what exists*
    rather than *what to change*, which it can because a baseline is documentation: nothing ever
    resolves or applies it.

    Instructions are listed block by block, each with the `id` that addresses it and a `dynamic` flag,
    which is the whole reason the UI can offer an override at all: a joined prompt has no seams in it,
    so a baseline built from that could only be copied wholesale -- and since managed instructions
    *add*, that is how you get the agent's own text sent to the model twice with a frozen
    `Today is <date>` in the middle of it.

    A dynamic block contributes that it exists, not what it said once. Its text is one request's
    rendering, built from whatever that request carried -- a tenant name, a user id, a retrieved
    document -- and the baseline is published to a variable every project member can read. The seam is
    what the editor needs anyway: enough to show the block and that it is not theirs to change. A
    dynamic block with nothing to key it on is left out entirely, since an editor can neither show nor
    address it.

    Every top-level parameter of every tool is listed, carrying its `description` when the code has
    one and an empty entry when it does not. The empty entry is the point: an undocumented parameter
    is exactly the one somebody wants to describe from Logfire, and a baseline that listed only the
    documented ones would hide it from the editor until it had been documented in code first.

    Settings pass through `canonical_settings`, which
    keeps the baseline to the canonical keys and to values the contract can actually hold.
    """
    entries: list[InstructionText | InstructionBlock] = []
    for block in instructions:
        # `Block` is the adapter's own dataclass and constrains nothing, while the contract's `id` is
        # a non-empty string. An empty one is an adapter bug rather than a value to carry, and it is
        # reported and treated as no id at all -- never raised, because this runs on the request path
        # and a baseline is documentation that no request depends on.
        block_id = block.id
        if block_id == '':
            # Said per branch, because the two outcomes differ: a static block is still published,
            # just not addressable, while a dynamic one has nothing left to key it on at all.
            warn_dropped(
                'The agent has an instruction block whose id is the empty string, which cannot address '
                'anything; '
                + (
                    'leaving it out of the published baseline entirely, since a dynamic block with no id '
                    'is one the Logfire editor could neither show nor address.'
                    if block.dynamic
                    else 'publishing that block without an id, so the Logfire editor cannot offer an override for it.'
                )
            )
            block_id = None
        if block.dynamic:
            if block_id is not None:
                entries.append(InstructionBlock(id=block_id, dynamic=True))
            continue
        if not block.text.strip():
            continue
        if _oversized(block.text):
            # Every other code-side value the contract cannot hold is left out and reported; an
            # oversized prompt must not be the one that raises, because `build_baseline` is called
            # on the request path and a baseline is documentation that no request depends on.
            warn_dropped(
                f'The agent runs with an instruction block of {len(block.text)} characters, exceeding the '
                f'{MAX_MODEL_FACING_TEXT_LENGTH}-character limit the Agent Control contract can describe; '
                'leaving it out of the published baseline.'
            )
            continue
        entries.append(InstructionBlock(id=block_id, instructions=block.text, dynamic=False))
    tool_definitions: list[ToolDefinitionOverride] = []
    for tool in tools:
        # `ToolDef` is the adapter's own dataclass too, and a tool with no name is one the contract
        # cannot address and the model cannot call. Reported and left out, for the same reason the
        # instruction blocks above are: nothing about describing the agent may raise into a request.
        if not tool.name:
            warn_dropped(
                'The agent advertises a tool whose name is the empty string, which no managed override '
                'could address; leaving it out of the published baseline.'
            )
            continue
        tool_definitions.append(
            ToolDefinitionOverride(
                name=tool.name,
                description=tool.description or None,
                parameters=_baseline_parameters(tool) or None,
                toolset=tool.toolset,
            )
        )
    canonical = canonical_settings(settings) if settings else {}
    return AgentConfig(
        instructions=entries or None,
        model=model,
        settings=AgentConfigSettings(**canonical) if canonical else None,
        tool_definitions=tool_definitions or None,
    )
