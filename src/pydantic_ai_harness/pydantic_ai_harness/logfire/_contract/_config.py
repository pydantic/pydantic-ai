"""The `AgentConfig` contract: what a Logfire Agent Control variable holds, and how leniently it validates."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Annotated, Any, Literal, TypeAlias, TypeVar, cast, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from ._reporting import warn_dropped
from ._schema import MAX_MODEL_FACING_TEXT_LENGTH

KeyT = TypeVar('KeyT', bound=Hashable)
EntryT = TypeVar('EntryT')


class AgentConfigSettings(BaseModel):
    """Canonical model settings managed as one section of an `AgentConfig`.

    The fields are the contract: the settings every framework Agent Control drives has a knob for,
    under the names Pydantic AI's `ModelSettings` gives them, so a published value lowers into any
    SDK without translation and means the same thing to every reader. Unset fields keep their
    code-defined values. The model itself is the sibling `model` field on `AgentConfig`.

    Nothing else gets through. A key this contract has no field for is dropped rather than forwarded
    -- a key it does not understand is not one it can lower into a request -- and remembered in
    `_unrecognized` so the adapter can report it under its `OnUnmatched` policy, since a newer UI's
    key that an SDK quietly did nothing with is exactly the kind of gap between what Logfire shows
    and what the agent does that has to be visible. A field whose *value* this release doesn't
    recognize -- an effort level a newer contract accepts -- is dropped with a warning so that
    setting alone keeps its code-defined value, rather than failing the enclosing `AgentConfig`.
    """

    model_config = ConfigDict(extra='ignore')

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    parallel_tool_calls: bool | None = None
    timeout: float | None = None
    stop_sequences: list[str] | None = None
    thinking: bool | Literal['minimal', 'low', 'medium', 'high', 'xhigh'] | None = None

    _unrecognized: tuple[str, ...] = PrivateAttr(default=())
    """The keys of the value this was validated from that this contract has no field for, in order.

    Private rather than a field because it is not part of the value: it is what the value asked for
    that this release could not do, which only the code applying the settings has any use for. It is
    not reported from validation -- see `report_unmatched` for why -- and is deliberately ignored on
    a code baseline, where an unrecognized key is the agent's own provider-specific setting rather
    than anything published.
    """

    @property
    def unrecognized(self) -> tuple[str, ...]:
        """The published keys this contract has no field for, for an adapter to report."""
        return self._unrecognized

    @model_validator(mode='wrap')
    @classmethod
    def _remember_unrecognized_keys(
        cls, data: Any, handler: ModelWrapValidatorHandler[AgentConfigSettings]
    ) -> AgentConfigSettings:
        settings = handler(data)
        if isinstance(data, dict):
            settings._unrecognized = tuple(name for name in cast(dict[str, Any], data) if name not in cls.model_fields)
        return settings

    @model_validator(mode='before')
    @classmethod
    def _drop_unrecognized_values(cls, data: Any) -> Any:
        """Drop a versioned field whose value this release doesn't recognize, keeping the rest of the patch.

        Ignoring unknown *keys* is only half of forward compatibility: a newer Logfire UI (or a newer
        contract adding an effort level) writes a value that the stored schema and the Logfire backend
        both accept, and an older SDK would then fail the whole `AgentConfig` over it and revert
        instructions, model, and every tool override to code along with it. Dropping the one field the
        SDK can't act on leaves the value it doesn't understand out of the lowered settings, so that
        setting -- and only that setting -- keeps its code-defined behavior.

        Unrecognized values are rejected, not passed through: the field's public type is the guarantee
        `apply_settings` and its consumers hold, and forwarding an unknown effort level to the
        provider would trade a version-skew problem for a request-time one.
        """
        if not isinstance(data, dict):
            return data
        values = cast(dict[str, Any], data)
        dropped: dict[str, Any] = {}
        for name, adapter in _VERSIONED_SETTINGS.items():
            if name in values:
                try:
                    adapter.validate_python(values[name], strict=True)
                except ValidationError:
                    dropped[name] = values[name]
        if not dropped:
            return values
        for name, value in dropped.items():
            warn_dropped(
                f'Managed agent config sets {name!r} to {value!r}, which this version of the SDK does not '
                f'recognize; ignoring that setting and keeping the rest of the managed config.'
            )
        return {name: value for name, value in values.items() if name not in dropped}

    @model_validator(mode='before')
    @classmethod
    def _drop_malformed_values(cls, data: Any) -> Any:
        """Drop malformed setting values independently so valid siblings still apply."""
        if not isinstance(data, dict):
            return data
        values = cast(dict[str, Any], data)
        kept: dict[str, Any] = {}
        for name, value in values.items():
            field_info = cls.model_fields.get(name)
            if field_info is None:
                # Not this validator's to judge: `extra='ignore'` drops it, and the wrap validator
                # records it for the adapter to report.
                kept[name] = value
                continue
            if _enumerates_values(field_info.annotation):
                # The version-skew validator owns enumerated fields so its warning distinguishes a
                # newer value from structurally malformed input.
                kept[name] = value
                continue
            try:
                SETTING_ADAPTERS[name].validate_python(value, strict=True)
            except ValidationError:
                warn_dropped(
                    f'Managed agent config setting {name!r} has invalid value {value!r}; ignoring that setting '
                    'and keeping the rest of the managed config.'
                )
            else:
                kept[name] = value
        return kept


def _enumerates_values(annotation: object) -> bool:
    """Whether an annotation spells out the values it accepts, i.e. names a `Literal` anywhere."""
    return get_origin(annotation) is Literal or any(_enumerates_values(arg) for arg in get_args(annotation))


SETTING_ADAPTERS: dict[str, TypeAdapter[Any]] = {
    name: TypeAdapter(field_info.annotation) for name, field_info in AgentConfigSettings.model_fields.items()
}
"""One validator per canonical setting, read off the field annotations rather than restated.

Used two ways, with opposite strictness, and the difference is the contract rather than an
inconsistency. A *published* value is validated strictly, because it arrives as JSON that the stored
schema and the Logfire backend already type-checked: `2048` for a `float` is the same number JSON
always writes, but `'0.4'` is a string where the schema says `number`, and accepting it would make
this core apply a value the TypeScript core -- whose parser is strict -- would drop. A *code-side*
value going into a baseline is validated leniently by `canonical_settings`, because it comes from a
framework's own objects, where an `int` for a `float` and a tuple of stop sequences are ordinary.
"""

_VERSIONED_SETTINGS: dict[str, TypeAdapter[Any]] = {
    name: adapter
    for name, adapter in SETTING_ADAPTERS.items()
    if _enumerates_values(AgentConfigSettings.model_fields[name].annotation)
}
"""Per-field validators for the settings whose accepted values grow from release to release.

A `Literal` in a field's annotation is exactly the signal that the field enumerates what *this*
release knows about, so the set of version-skew-prone fields (`thinking`, today) and the values each
one accepts are both read off the annotations rather than restated here, where the two would drift
apart. Fields without a `Literal` are left alone: a wrong type there is malformed rather than merely
newer, and the stored schema already rejects it at write time.
"""


NonEmptyStr: TypeAlias = Annotated[str, Field(min_length=1)]
"""A managed string that has to say something.

`''` is never a meaningful managed value -- not "no model", not "no instructions", just a value someone
left half-filled -- and `None` already means "leave this to code". Rejecting it keeps the two apart at
every level: the stored JSON schema won't accept the write, and a value that reaches an older SDK
anyway degrades that one field instead of the whole config.
"""

InstructionText: TypeAlias = Annotated[str, Field(min_length=1, max_length=MAX_MODEL_FACING_TEXT_LENGTH)]
"""A non-empty instruction block bounded before it can become recurring model input."""


class InstructionBlock(BaseModel):
    """One entry in `AgentConfig.instructions`: a block to add, or a patch on a block the agent assembles.

    Instructions are the one section that composes rather than replaces, so an entry has to say *which*
    text it means. An entry with no `id` adds a block; an entry with an `id` addresses the instruction
    blocks the agent has already assembled under that key.

    | Entry | Effect |
    |---|---|
    | `'text'` (a bare string in the list) | Adds a block. Shorthand for `InstructionBlock(instructions='text')`. |
    | `InstructionBlock(instructions='text')` | Adds a block. |
    | `InstructionBlock(id=key, instructions='text')` | Replaces the text of every block keyed `key`. |
    | `InstructionBlock(id=key)` | Drops every block keyed `key`. |

    An `id` nothing matches applies nothing and is reported under the adapter's `OnUnmatched` policy,
    exactly like an unknown tool name: one config is applied across deployments that need not all
    assemble the same prompt, so the default is a warning rather than an error. Two entries naming the
    same `id` keep the first, with a warning.
    """

    id: str | None = Field(default=None, min_length=1)
    """The id of the instruction block to address, or `None` to add a block.

    `''` is not one: it is the same half-filled field `instructions=''` is, and the stored JSON schema
    already refuses it, so accepting it here would let a baseline be built that the Logfire backend
    rejects on write and let a published entry address a block no adapter can name.

    The contract keeps `id` free-form and reserves only `'agent'` as the cross-framework name for the
    prompt as written; the rest of the namespace belongs to each adapter, which documents its own and
    lists every key it uses in the baseline it publishes. A block the framework cannot name -- an
    anonymous callable prompt, a tool's contribution with nothing to key it on -- cannot be addressed
    at all, which is why the baseline is what an editor offers overrides from.
    """
    instructions: InstructionText | None = None
    """The block's text, or `None` to drop the addressed block.

    `None` is how a block is disabled, which is why `''` is rejected rather than taken as a quiet way to
    blank one: an entry meaning "send nothing here" and an entry someone left half-filled should not look
    identical. An entry with neither `id` nor `instructions` says nothing and is dropped with a warning.
    """
    dynamic: bool | None = None
    """Whether the addressed block is recomputed per request. Informational; ignored when a value is applied.

    Written into a variable's `example` by the baseline snapshot, where it earns its keep: it is how the
    Logfire UI can warn that replacing a computed block -- today's date, the signed-in user -- pins
    whatever it happened to evaluate to when the snapshot was taken.
    """


class ParameterOverride(BaseModel):
    """A patch over one top-level parameter of a tool's LLM-facing definition.

    Only what the model is shown changes. The parameter's name, type, and requiredness stay as
    defined in code, so argument validation is unaffected.
    """

    description: str | None = None
    """Replacement description shown to the model; `None` keeps the code-defined description."""


class ToolDefinitionOverride(BaseModel):
    """A patch over a tool's LLM-facing definition.

    Overrides change only what the model is shown. Schema structure, validation, and execution
    remain code-defined. `name` looks the tool up by its original code-side name, narrowed to one
    toolset's tool of that name when `toolset` is set too. An entry that doesn't validate is dropped
    from `AgentConfig.tool_definitions` with a warning, leaving its siblings in place; two entries
    with the same `name` and `toolset` keep the first, with a warning.
    """

    name: str = Field(min_length=1)
    """The tool's original code-side name, which is what this entry patches.

    A tool the agent does not advertise applies nothing and is reported under the adapter's
    `OnUnmatched` policy, for the same reason an unmatched instruction `id` is.
    """
    new_name: str | None = Field(default=None, min_length=1)
    """Replacement name shown to the model; `None` keeps the original.

    Whether a renamed call reaches the implementation under its old name or its new one is the
    adapter's to document: `AppliedTools.routes` gives it the mapping either way.
    """
    description: str | None = None
    """Replacement description shown to the model; `None` keeps the code-defined description."""
    parameters: dict[str, ParameterOverride] | None = None
    """Patches per top-level parameter name; see `ParameterOverride`.

    A patch naming a parameter the tool does not have applies nothing and is reported under the
    adapter's `OnUnmatched` policy, rather than dropped in silence: a parameter is part of the tool's
    code-defined shape, so a patch on one the tool does not have is the tool having changed, which
    the baseline shows.
    """
    toolset: str | None = None
    """The group the tool came from: the baseline reports it, and on an override it narrows the match.

    An adapter emits whatever names the origin in its framework -- a toolset's id, an MCP server's
    name -- so the Logfire UI can group a long tool list by where each tool comes from without
    inferring it from tool names. An override that sets it applies only to that group's tool of this
    `name`, which is what lets two servers that both advertise a `search` be patched apart; an
    override without it matches by `name` alone, and when both match one tool the qualified entry wins.
    """


def _entry_errors(error: ValidationError, *, whole: str) -> str:
    """Render a rejected entry's failure -- offending field, value, and reason -- for one warning.

    `whole` names the entry itself, for a failure that isn't about any one field (something that isn't an
    object at all): the two sections share this renderer, so neither may be labelled with the other's noun.
    """
    return '; '.join(
        f'{".".join(str(part) for part in details["loc"]) or whole}={details["input"]!r} ({details["msg"]})'
        for details in error.errors()
    )


def first_by_key(
    entries: list[tuple[KeyT, EntryT]], describe: Callable[[KeyT], str]
) -> tuple[dict[KeyT, EntryT], list[tuple[KeyT, str]]]:
    """Index entries by key, keeping the first of any duplicates and naming the rest.

    Both list sections address things by key, so both can be written with the same key twice -- by a
    hand-edited value, or by a UI bug. Keeping the first matches how a colliding rename is resolved in
    `apply_tool_definitions`: the request stays predictable and the ignored entry is named, rather than
    the last writer silently winning depending on how the JSON happened to be ordered. `describe`
    renders a key for that message, since a tool's key is a pair and reads badly as a bare tuple.

    Returns the index plus every duplicate key with the message naming it, for the caller to turn into
    a `'duplicate-entry'` issue of its own section. Deliberately not warned from here: this is an
    apply-time decision about a value, so it belongs under the caller's `OnUnmatched` policy like
    every other one -- warning directly, as this used to, meant `'ignore'` still warned and `'error'`
    did not raise.
    """
    indexed: dict[KeyT, EntryT] = {}
    duplicates: list[tuple[KeyT, str]] = []
    for key, entry in entries:
        if key in indexed:
            duplicates.append(
                (
                    key,
                    f'Managed agent config names {describe(key)} more than once; keeping the first entry '
                    f'and ignoring the rest.',
                )
            )
            continue
        indexed[key] = entry
    return indexed, duplicates


ToolKey: TypeAlias = tuple[str | None, str]
"""What a tool override is matched on: the group it is narrowed to (`None` for any) and the tool's name."""


def describe_tool_key(key: ToolKey) -> str:
    toolset, name = key
    return f'tool {name!r}' if toolset is None else f'tool {name!r} from toolset {toolset!r}'


class AgentConfig(BaseModel):
    """The schema contract shared with the Logfire Agent Control UI.

    Every managed value is a patch on the code-defined agent. A key present in the value is managed
    from Logfire; an absent key keeps code-defined behavior. Removing a key in Logfire is therefore
    a deliberate revert to code.

    Nothing an SDK fails to understand costs more than the part that contains it. Extra keys retain
    Pydantic's default `ignore` behavior, and a *value* it cannot make sense of -- an effort level a
    newer contract accepts, a tool override that doesn't validate -- drops only its own setting or
    its own override, with a warning. Both rules exist for the same reason: a value that fails
    validation falls back through Logfire's resolution to the code-defined agent *in its entirety*,
    so without them one unfamiliar key or enum value would silently un-manage the instructions, the
    model, and every tool override alongside it. An extra key is nonetheless *remembered* and
    reported under the adapter's `OnUnmatched` policy, at both levels: a `settings` key this release
    has no field for, and a top-level key it has no section for. Ignoring what it cannot do is what
    keeps a future section readable by an older SDK; saying so is what stops the first person who
    publishes one from getting a silently degraded agent.
    """

    model_config = ConfigDict(protected_namespaces=())

    instructions: InstructionText | list[InstructionText | InstructionBlock] | None = None
    """Instruction blocks to add to -- or swap out of -- the ones the agent assembles in code.

    Blocks with no `id` are *added*, which is why a bare string means one added block and text that also
    lives in the agent's own prompt reaches the model twice. Blocks *with* an `id` swap out the block the
    agent assembled under that key -- replacing its text, or dropping it with `instructions=None`; see
    `InstructionBlock`.

    A bare string is exactly `[InstructionBlock(instructions='text')]` and is kept as written rather
    than rewritten into the list form, so a published value stays the shape its author chose and
    successive versions stay readable as a diff. An entry that doesn't validate is dropped with a
    warning, leaving its siblings and the rest of the config alone.
    """
    model: NonEmptyStr | None = None
    """A model string in `'provider:model'` form, such as `'anthropic:claude-fable-5-1'`; `None` keeps the code model.

    The contract's canonical form is a `provider:model` string with Pydantic AI's provider ids, which
    is also what the Logfire model catalog knows. An adapter maps it to its framework's own convention
    and passes a string with no `:` through untouched, so a framework-native id also works.

    Non-empty for a blunt reason: `''` is not "no model", it is a model named `''`, and every framework
    rejects it on every request the agent makes. Publishing one would take the agent down, and the
    resolution fallback cannot catch it because the config itself is perfectly valid.

    A value that is not a non-empty string costs only this field: it is dropped with a warning and
    the agent keeps its code-defined model, exactly the way a malformed setting or a malformed tool
    override costs only its own entry. Failing the whole `AgentConfig` over it would revert the
    instructions, the settings, and every tool override alongside it.
    """
    settings: AgentConfigSettings | None = None
    """Canonical model settings patch; see `AgentConfigSettings`."""
    tool_definitions: list[ToolDefinitionOverride] | None = None
    """LLM-facing overlays, each naming the tool it patches; see `ToolDefinitionOverride`."""

    _unrecognized: tuple[str, ...] = PrivateAttr(default=())
    """The top-level keys of the value this was validated from that this release has no section for.

    Private rather than a field for the same reason `AgentConfigSettings._unrecognized` is: it is not
    part of the value, it is what the value asked for that this release could not do, and only the
    code applying the config has any use for it. Not reported from validation -- see
    `report_unmatched` for why -- and deliberately ignored on a code baseline, which is built from
    keyword arguments rather than from a mapping and so never has any.
    """

    @property
    def unrecognized(self) -> tuple[str, ...]:
        """The published top-level keys this release has no section for, for an adapter to report."""
        return self._unrecognized

    @model_validator(mode='wrap')
    @classmethod
    def _remember_unrecognized_sections(cls, data: Any, handler: ModelWrapValidatorHandler[AgentConfig]) -> AgentConfig:
        config = handler(data)
        if isinstance(data, dict):
            config._unrecognized = tuple(name for name in cast(dict[str, Any], data) if name not in cls.model_fields)
        return config

    @field_validator('model', mode='before')
    @classmethod
    def _drop_invalid_model(cls, data: Any) -> Any:
        """Drop a model this contract cannot use, keeping the rest of the config.

        The one section with no entries to degrade one at a time, so the section itself is the unit.
        Leniency is per section everywhere else, and `model` was the last field whose failure took the
        whole value down with it -- which is also where the two cores disagreed, since the TypeScript
        one has always dropped just the field.
        """
        if data is None or (isinstance(data, str) and data):
            return data
        warn_dropped(
            f'Managed agent config selects invalid model {data!r}; ignoring that section and keeping the rest '
            'of the managed config.'
        )
        return None

    @field_validator('instructions', mode='before')
    @classmethod
    def _drop_invalid_instructions(cls, data: Any) -> Any:
        """Drop an instruction entry that doesn't validate, keeping its siblings and the rest of the config.

        An entry is the natural unit of degradation here, the same way a tool override is: each one adds
        or addresses exactly one block, so an entry an SDK can't make sense of -- an empty string where
        text was meant, an entry that says neither what nor where, something that is neither a string nor
        an object -- can be left out while every other block still applies. Without this, one bad entry
        would fail the whole `AgentConfig` and revert the model, the settings, and every tool override to
        code alongside it.

        A bare string is left alone for the field itself to validate: it is one block by definition, so
        there is no sibling to save by rescuing it, and preserving the shape keeps a published value
        looking the way its author wrote it. Entries in a list are returned already validated so the
        field doesn't validate them a second time.
        """
        if isinstance(data, str):
            if 0 < len(data) <= MAX_MODEL_FACING_TEXT_LENGTH:
                return data
            if len(data) > MAX_MODEL_FACING_TEXT_LENGTH:
                warn_dropped(
                    f'Managed instructions section contains {len(data)} characters, exceeding the '
                    f'{MAX_MODEL_FACING_TEXT_LENGTH}-character limit; ignoring that section and keeping the rest '
                    'of the managed config.'
                )
                return None
            warn_dropped(
                "Managed instructions section is invalid -- instructions=''; ignoring that section and keeping "
                'the rest of the managed config.'
            )
            return None
        if not isinstance(data, list):
            if data is not None:
                warn_dropped(
                    f'Managed instructions section has invalid container {data!r}; ignoring that section and '
                    'keeping the rest of the managed config.'
                )
                return None
            return data
        blocks: list[InstructionBlock] = []
        # The bound is on the text this section adds to every model request, so it has to be the total
        # across entries, not just each one: a section written as one string is capped, and the same
        # text written as ten entries has to be capped too or the limit means nothing. Only text that
        # actually survives is charged against it -- an entry dropped for being malformed adds nothing
        # to the request, so charging it would let one bad entry shrink the budget for the good ones,
        # and would make the entries a value keeps depend on the ones it does not.
        remaining = MAX_MODEL_FACING_TEXT_LENGTH
        for entry in cast(list[Any], data):
            text: object | None = entry if isinstance(entry, str) else None
            if isinstance(entry, dict):
                # Runtime narrowing cannot recover a dictionary's generic parameters from untyped JSON.
                typed_entry = cast(dict[str, object], entry)
                text = typed_entry.get('instructions')
            if isinstance(text, str) and len(text) > remaining:
                warn_dropped(
                    f'Managed instruction entry contains {len(text)} characters, which does not fit in the '
                    f'{remaining} remaining of the {MAX_MODEL_FACING_TEXT_LENGTH}-character limit across all '
                    'entries; ignoring that entry and keeping the rest of the managed config.'
                )
                continue
            try:
                block = InstructionBlock.model_validate({'instructions': entry} if isinstance(entry, str) else entry)
            except ValidationError as error:
                warn_dropped(
                    f'Managed instruction entry {entry!r} is invalid -- {_entry_errors(error, whole="entry")}; '
                    f'ignoring that entry and keeping the rest of the managed config.'
                )
                continue
            if block.id is None and block.instructions is None:
                warn_dropped(
                    f'Managed instruction entry {entry!r} has neither an `id` to address nor text to add; '
                    f'ignoring that entry and keeping the rest of the managed config.'
                )
                continue
            if isinstance(text, str):
                remaining -= len(text)
            blocks.append(block)
        return blocks

    @field_validator('tool_definitions', mode='before')
    @classmethod
    def _drop_invalid_overrides(cls, data: Any) -> Any:
        """Drop an override entry that doesn't validate, keeping its siblings and the rest of the config.

        A tool override is the natural unit here: each entry patches exactly one tool, so an entry the
        SDK can't validate -- a missing or empty `name`, a field carrying a shape it doesn't know,
        something that isn't an object at all -- can be left out while every other tool keeps its managed
        definition. Without this, one bad entry would fail the whole `AgentConfig` and revert the
        agent's instructions, model, and settings to code as well. (Merely *unknown* keys inside an
        entry are ignored by `ToolDefinitionOverride` itself and cost nothing.)

        Entries are returned already validated so the field doesn't validate them a second time.
        """
        if not isinstance(data, list):
            if data is not None:
                warn_dropped(
                    f'Managed tool definitions section has invalid container {data!r}; ignoring that section and '
                    'keeping the rest of the managed config.'
                )
                return None
            return data
        overrides: list[ToolDefinitionOverride] = []
        for entry in cast(list[Any], data):
            try:
                overrides.append(ToolDefinitionOverride.model_validate(entry))
            except ValidationError as error:
                warn_dropped(
                    f'Managed tool definition override {entry!r} is invalid -- '
                    f'{_entry_errors(error, whole="override")}; ignoring that override and keeping the rest of '
                    'the managed config.'
                )
        return overrides

    @field_validator('settings', mode='before')
    @classmethod
    def _drop_invalid_settings_container(cls, data: Any) -> Any:
        if isinstance(data, AgentConfigSettings):
            return data
        if data is None or isinstance(data, dict):
            return None if data is None else cast(dict[str, Any], data)
        warn_dropped(
            f'Managed settings section has invalid container {data!r}; ignoring that section and keeping the rest '
            'of the managed config.'
        )
        return None


def instruction_blocks(config: AgentConfig) -> list[InstructionBlock]:
    """The `instructions` section as blocks, whichever of its two shapes was written."""
    instructions = config.instructions
    if instructions is None:
        return []
    if isinstance(instructions, str):
        return [InstructionBlock(instructions=instructions)]
    return [InstructionBlock(instructions=entry) if isinstance(entry, str) else entry for entry in instructions]
