from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, replace
from typing import Generic

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai._utils import dataclasses_no_defaults_repr
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    CapabilityInstructionSource,
    InstructionBaselinePart,
    InstructionChangePolicy,
    InstructionDeltaPart,
    InstructionId,
    InstructionPart,
    InstructionSource,
    ModelMessage,
    ModelRequest,
    _instruction_text_hash,  # pyright: ignore[reportPrivateUsage]
    post_compaction_window,
)
from pydantic_ai.template import TemplateStr

from . import _system_prompt
from .tools import SystemPromptFunc

AgentInstruction = TemplateStr[AgentDepsT] | str | InstructionPart | SystemPromptFunc[AgentDepsT]
"""One instruction: literal text, a function computing it, or an `InstructionPart` declaring both the
text and how it should be treated — its [`name`][pydantic_ai.messages.InstructionPart.name] (qualified
into an id against whatever source contributes it) and whether it counts as
[`dynamic`][pydantic_ai.messages.InstructionPart.dynamic] for prompt caching."""

AgentInstructions = AgentInstruction[AgentDepsT] | Sequence[AgentInstruction[AgentDepsT]] | None


def validate_instruction_id_segment(id: str, *, kind: str) -> None:
    """Reject values that cannot be represented unambiguously in an instruction id."""
    if ':' in id:
        raise UserError(f'{kind} {id!r} cannot contain a colon because `:` is reserved as an instruction ID delimiter.')


def validate_instruction_name(name: str) -> None:
    """Reject names an author cannot declare on an instruction part.

    A name is one segment of the id built around it, so it must not be able to spell a key by itself.
    `'agent'` is the only one it could reach: every other key is namespaced, and a colon is rejected
    above.
    """
    validate_instruction_id_segment(name, kind='Instruction name')
    if name == 'agent':
        raise UserError(
            "Instruction name 'agent' is reserved for the agent's own instructions; choose a different name."
        )


@dataclass(frozen=True, repr=False)
class SourcedInstruction(Generic[AgentDepsT]):
    """A lazy instruction recipe with the name and key its content should be addressed by."""

    instruction: AgentInstruction[AgentDepsT]

    _: KW_ONLY

    name: str | None = None
    id: InstructionId | None = None
    dynamic: bool = False
    on_change: InstructionChangePolicy = 'rewrite'
    framework_default: bool = False
    """Whether `on_change` is the framework's choice for a block it owns rather than its author's.

    Such a block uses `'rewrite'` without a warning where appending isn't supported, such as in a
    realtime session or when its capability is loaded on demand.
    """

    __repr__ = dataclasses_no_defaults_repr


def append_on_change_by_default(
    instructions: Sequence[SourcedInstruction[AgentDepsT]], *, key: str
) -> list[SourcedInstruction[AgentDepsT]]:
    """Make the computed blocks of a framework-owned capability append their changes.

    Built-in and harness capabilities whose instructions change between requests or runs use this, so
    their changes extend the cached prompt rather than rewriting it. A block with no id of its own,
    because its capability has no `id`, is keyed `capability:<key>`. Literal text, which can't change,
    and a block whose author chose a policy are left as they are.
    """
    return [
        replace(
            sourced,
            id=sourced.id or InstructionId(CapabilityInstructionSource(key), name=sourced.name),
            on_change='append',
            framework_default=True,
        )
        if sourced.dynamic and sourced.on_change == 'rewrite'
        else sourced
        for sourced in instructions
    ]


def explicitly_appends(
    instructions: Sequence[SourcedInstruction[AgentDepsT]], parts: Sequence[InstructionPart]
) -> bool:
    """Whether an author, rather than the framework, asked for `on_change='append'` on any of `parts`.

    `parts` are what `instructions` (and any toolsets) resolved to.
    """
    framework_ids = {sourced.id for sourced in instructions if sourced.framework_default}
    return any(part.on_change == 'append' and part.id not in framework_ids for part in parts)


def sourced_instruction(
    instruction: AgentInstruction[AgentDepsT], source: InstructionSource | None
) -> SourcedInstruction[AgentDepsT]:
    """Attribute one instruction recipe to the source that authored it.

    The single place a declared name meets its source, so every author applies the same rule: with a
    source the name is qualified into an [`InstructionId`][pydantic_ai.messages.InstructionId] beneath
    it, and without one there is no key to qualify against, so the name stays a name and the part
    stays unaddressable.

    A caller passes `None` for a recipe its source does not speak for -- a callable the agent was
    built with, or instructions belonging to a single run rather than to the agent.
    """
    name = instruction.name if isinstance(instruction, InstructionPart) else None
    if name is not None:
        validate_instruction_name(name)
    return SourcedInstruction(
        instruction,
        name=name,
        id=InstructionId(source, name=name) if source is not None else None,
        dynamic=not isinstance(instruction, (str, InstructionPart)),
    )


async def resolve_sourced_instructions(
    instructions: Sequence[SourcedInstruction[AgentDepsT]], run_context: RunContext[AgentDepsT]
) -> list[InstructionPart]:
    """Resolve authored instructions into the parts sent to the model.

    Literal strings with the same source key form one addressable part. An
    [`InstructionPart`][pydantic_ai.messages.InstructionPart] always remains independent so its
    cache treatment applies only to its own text, while callable instructions are resolved lazily
    against the current `RunContext`.
    """
    parts: list[InstructionPart] = []
    group: list[InstructionPart] = []
    pending_parts: list[InstructionPart] = []
    group_key: InstructionId | None = None

    def flush_group() -> None:
        if content := InstructionPart.join(group):
            parts.append(InstructionPart(content=content, id=group[0].id))
        group.clear()
        parts.extend(pending_parts)
        pending_parts.clear()

    for sourced in instructions:
        instruction = sourced.instruction
        if isinstance(instruction, InstructionPart):
            if not (content := instruction.content.strip()) and instruction.on_change != 'append':
                continue
            flush_group()
            group_key = None
            parts.append(replace(instruction, content=content, id=sourced.id))
        elif isinstance(instruction, str):
            if not (content := instruction.strip()):
                continue
            if group and (sourced.id is None or group_key != sourced.id):
                flush_group()
            group_key = sourced.id
            group.append(InstructionPart(content=content, id=sourced.id))
        else:
            content = await _system_prompt.SystemPromptRunner[AgentDepsT](instruction).run(run_context)
            if content or sourced.on_change == 'append':
                part = InstructionPart(
                    content=content or '',
                    name=sourced.name,
                    id=sourced.id,
                    dynamic=sourced.dynamic,
                    on_change=sourced.on_change,
                )
                if group:
                    pending_parts.append(part)
                else:
                    parts.append(part)
    flush_group()
    return parts


def normalize_instructions(
    instructions: AgentInstructions[AgentDepsT],
) -> list[AgentInstruction[AgentDepsT]]:
    if instructions is None:
        return []
    # Note: TemplateStr is callable (__call__) so it's handled by the callable branch
    if isinstance(instructions, (str, InstructionPart)) or callable(instructions):
        return [instructions]
    return list(instructions)


def normalize_toolset_instruction_parts(
    result: str | InstructionPart | Sequence[str | InstructionPart] | None,
) -> list[InstructionPart]:
    """Normalize a toolset `get_instructions` result into non-empty parts, ids untouched.

    A toolset may return a single `str` or `InstructionPart`, a sequence of either, or `None`. Plain
    strings are treated as dynamic (they come from an external/changeable source) and whitespace-only
    content is dropped. Ids are left exactly as the author wrote them, so whoever interprets them can
    still tell a key issued below from a segment declared here.
    """
    if not result:
        return []
    items = [result] if isinstance(result, (str, InstructionPart)) else result
    parts: list[InstructionPart] = []
    for item in items:
        part = item if isinstance(item, InstructionPart) else InstructionPart(content=item, dynamic=True)
        if part.content.strip() or part.on_change == 'append':
            parts.append(part)
    return parts


def _normalize_append_parts(current: list[InstructionPart]) -> list[InstructionPart]:
    """Downgrade append-mode blocks without a unique identity to `'rewrite'`, and strip the others."""
    counts = Counter(part.id for part in current)
    normalized: list[InstructionPart] = []
    for part in current:
        if part.on_change == 'append' and (part.id is None or counts[part.id] > 1):
            warnings.warn(
                f'Instruction block {str(part.id) if part.id is not None else part.name!r} uses '
                "`on_change='append'` without a unique instruction identity; rewriting its prefix. "
                'Declare a unique `name` and an owning capability/toolset `id` where applicable.',
                UserWarning,
                stacklevel=3,
            )
            part = replace(part, on_change='rewrite')
        normalized.append(replace(part, content=part.content.strip()) if part.on_change == 'append' else part)
    return normalized


def instruction_window(messages: Sequence[ModelMessage]) -> InstructionBaselinePart | None:
    """The baseline of the instruction window `messages` end in, or `None` if they aren't in one.

    The window starts at the latest [`InstructionBaselinePart`][pydantic_ai.messages.InstructionBaselinePart]
    after the latest compaction. An empty one ends the previous window and opens one with no blocks, whose
    append-mode blocks are then delivered as changes rather than in the prefix.
    """
    baseline: InstructionBaselinePart | None = None
    for message in post_compaction_window(messages):
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, InstructionBaselinePart):
                    baseline = part
    return baseline


def window_prefix(baseline: InstructionBaselinePart) -> list[InstructionPart]:
    """The instruction parts every request in `baseline`'s window was sent: the baseline, minus empty blocks."""
    return [part for part in baseline.instructions if part.content]


def _recorded_instruction_values(
    messages: Sequence[ModelMessage], baseline: InstructionBaselinePart
) -> dict[str, str | None]:
    """Each append-mode block's value as the model last received it in `baseline`'s window."""
    values: dict[str, str | None] = {
        str(part.id): part.content or None
        for part in baseline.instructions
        if part.on_change == 'append' and part.id is not None
    }
    in_window = False
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if part is baseline:
                    in_window = True
                elif in_window and isinstance(part, InstructionDeltaPart):
                    values[part.id] = part.content
    return values


def _continues_window(baseline: InstructionBaselinePart, current: list[InstructionPart]) -> bool:
    """Whether `current` can be sent with `baseline`'s prefix, appending changes to its append-mode blocks.

    Rewrite-mode blocks are sent as they are, so they must be unchanged, in the same order; a block whose
    policy changed is one of them. An append-mode block whose id didn't survive deserialization (a source
    namespace this version doesn't know) can't be tagged, so it can't anchor a window either.
    """
    if any(part.on_change == 'append' and part.id is None for part in baseline.instructions):
        return False
    return [part for part in baseline.instructions if part.on_change != 'append'] == [
        part for part in current if part.on_change != 'append'
    ]


def update_instruction_history(
    messages: Sequence[ModelMessage],
    instructions: list[InstructionPart] | None,
    targets: Sequence[ModelRequest],
) -> list[InstructionPart] | None:
    """Preserve the instruction prefix and append changed instruction blocks to the outgoing request.

    Run after history processing: only a baseline that survived processing can anchor the prefix. While
    the rewrite-mode blocks stay the same, every request is sent the prefix its window opened with, and
    each append-mode block's changes are recorded on `targets` as
    [`InstructionDeltaPart`][pydantic_ai.messages.InstructionDeltaPart]s. When the prefix has to change
    anyway, the request opens a new window with the current values, which costs no more cache than the
    change itself. `targets` are the requests that both this request and the persisted history carry.
    """
    normalized = _normalize_append_parts(instructions or [])
    baseline = instruction_window(messages)

    if instructions is None:
        # Unset parts preserve the recorded request text and end the window, so its old changes stop applying.
        if baseline is not None and baseline.instructions:
            _record_baseline(targets, [])
        return None
    if baseline is not None and not _continues_window(baseline, normalized):
        baseline = None
        if not any(part.on_change == 'append' for part in normalized):
            # Nothing appends any more, so only end the window its blocks' old changes belong to.
            _record_baseline(targets, [])
            return normalized
    if baseline is None:
        if not any(part.on_change == 'append' for part in normalized):
            return normalized
        baseline = _record_baseline(targets, normalized)
    else:
        recorded = _recorded_instruction_values(messages, baseline)
        current = {str(part.id): part.content or None for part in normalized if part.on_change == 'append'}
        changes = [
            _instruction_change(instruction_id, recorded.get(instruction_id), current.get(instruction_id))
            for instruction_id in dict.fromkeys([*recorded, *current])
            if recorded.get(instruction_id) != current.get(instruction_id)
        ]
        if changes:
            for target in targets:
                target.parts = [*target.parts, *changes]

    # Parts keep only their content: each model renders the append-mode ones for its own delivery path in
    # `Model.prepare_request`.
    prefix = window_prefix(baseline)
    for target in targets:
        target.instructions = InstructionPart.join(prefix)
    return prefix


def _record_baseline(targets: Sequence[ModelRequest], instructions: list[InstructionPart]) -> InstructionBaselinePart:
    """Open a window at this request, or close the current one when `instructions` is empty."""
    baseline = InstructionBaselinePart(instructions=[replace(part) for part in instructions])
    for target in targets:
        target.parts = [*target.parts, baseline]
    return baseline


def _instruction_change(instruction_id: str, previous: str | None, content: str | None) -> InstructionDeltaPart:
    """Record a block's change, with the hash of the text it replaces so a model can send it as a diff."""
    base_hash = _instruction_text_hash(previous) if previous is not None and content is not None else None
    return InstructionDeltaPart(id=instruction_id, content=content, base_hash=base_hash)
