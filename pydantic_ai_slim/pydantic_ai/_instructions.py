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
    CompactionPart,
    InstructionBaselinePart,
    InstructionChangePolicy,
    InstructionDeltaPart,
    InstructionDigest,
    InstructionId,
    InstructionPart,
    InstructionSource,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    _instruction_text_hash,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.template import TemplateStr

from . import _system_prompt

InstructionFunc = _system_prompt.PromptFunc[AgentDepsT, str | InstructionPart]
"""A function computing an instruction: it may or may not take `RunContext`, may or may not be async,
and returns text, an [`InstructionPart`][pydantic_ai.messages.InstructionPart], or `None`.

A returned `InstructionPart` declares its [`name`][pydantic_ai.messages.InstructionPart.name] and
[`on_change`][pydantic_ai.messages.InstructionPart.on_change] policy from inside the function, the way
a registering decorator would. Whatever it says about `dynamic`, a computed part counts as dynamic."""

AgentInstruction = TemplateStr[AgentDepsT] | str | InstructionPart | InstructionFunc[AgentDepsT]
"""One instruction: literal text, a function computing it, or an `InstructionPart` declaring both the
text and how it should be treated — its [`name`][pydantic_ai.messages.InstructionPart.name] (qualified
into an id against whatever source contributes it), its
[`on_change`][pydantic_ai.messages.InstructionPart.on_change] policy, and whether it counts as
[`dynamic`][pydantic_ai.messages.InstructionPart.dynamic] for prompt caching. A function can return an
`InstructionPart` to declare the same from inside it."""

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
    source: InstructionSource | None = None
    """The source a name the instruction declares when it's evaluated is qualified against.

    Set even where `id` is `None` because the recipe is unnamed, so a function that names the part it
    returns is addressed the same as one named where it was registered.
    """
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


def explicitly_appended_blocks(
    instructions: Sequence[SourcedInstruction[AgentDepsT]], parts: Sequence[InstructionPart]
) -> list[str]:
    """The blocks among `parts` an author, rather than the framework, asked `on_change='append'` for, by id or name.

    `parts` are what `instructions` (and any toolsets) resolved to.
    """
    framework_ids = {sourced.id for sourced in instructions if sourced.framework_default}
    return [
        str(part.id) if part.id is not None else part.name or 'unnamed'
        for part in parts
        if part.on_change == 'append' and part.id not in framework_ids
    ]


def sourced_instruction(
    instruction: AgentInstruction[AgentDepsT],
    source: InstructionSource | None,
    *,
    name_source: InstructionSource | None = None,
) -> SourcedInstruction[AgentDepsT]:
    """Attribute one instruction recipe to the source that authored it.

    The single place a declared name meets its source, so every author applies the same rule: with a
    source the name is qualified into an [`InstructionId`][pydantic_ai.messages.InstructionId] beneath
    it, and without one there is no key to qualify against, so the name stays a name and the part
    stays unaddressable.

    A caller passes `None` for a recipe its source does not speak for -- a callable the agent was
    built with, or instructions belonging to a single run rather than to the agent. `name_source`,
    which defaults to `source`, is what a function that names the part it returns is qualified
    against: the agent's callables speak for nobody until they do.
    """
    name = instruction.name if isinstance(instruction, InstructionPart) else None
    if name is not None:
        validate_instruction_name(name)
    return SourcedInstruction(
        instruction,
        name=name,
        id=InstructionId(source, name=name) if source is not None else None,
        source=name_source or source,
        dynamic=not isinstance(instruction, (str, InstructionPart)),
    )


async def resolve_instruction_function(
    sourced: SourcedInstruction[AgentDepsT],
    function: TemplateStr[AgentDepsT] | InstructionFunc[AgentDepsT],
    run_context: RunContext[AgentDepsT],
) -> InstructionPart | None:
    """Run the instruction function `sourced` was registered as, returning the part it contributes, if any.

    An append-mode part is kept when empty so its withdrawal can be recorded.
    """
    result = await _system_prompt.SystemPromptRunner[AgentDepsT, str | InstructionPart](function).run(run_context)
    if isinstance(result, InstructionPart):
        part = _computed_part(sourced, result)
    else:
        part = InstructionPart(
            content=result or '',
            name=sourced.name,
            id=sourced.id,
            dynamic=sourced.dynamic,
            on_change=sourced.on_change,
        )
    return part if part.content or part.on_change == 'append' else None


def _computed_part(sourced: SourcedInstruction[AgentDepsT], result: InstructionPart) -> InstructionPart:
    """Resolve a part an instruction function returned against the recipe it was registered as.

    The part's `name` and `on_change` apply as if the function had been registered with them: a name is
    qualified against the recipe's source, and either side asking to append wins over the `'rewrite'`
    default. A name the registration already declared can't be renamed from inside the function.
    """
    name = sourced.name
    instruction_id = sourced.id
    if result.name is not None and result.name != sourced.name:
        validate_instruction_name(result.name)
        if sourced.name is not None:
            raise UserError(
                f'An instruction function registered with name {sourced.name!r} returned a part named '
                f'{result.name!r}; leave the returned part unnamed or use the same name.'
            )
        name = result.name
        instruction_id = InstructionId(sourced.source, name=name) if sourced.source is not None else None
    return InstructionPart(
        content=result.content,
        name=name,
        id=instruction_id,
        dynamic=True,
        on_change=sourced.on_change if sourced.on_change == 'append' else result.on_change,
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
        elif (part := await resolve_instruction_function(sourced, instruction, run_context)) is not None:
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
    """Downgrade append-mode blocks without a unique identity to `'rewrite'`, and strip the others.

    Append-mode blocks are kept when empty so a withdrawal can be recorded; a downgraded one is dropped then,
    like any rewrite-mode block with no content.
    """
    counts = Counter(part.id for part in current)
    normalized: list[InstructionPart] = []
    for part in current:
        if part.on_change == 'append' and (part.id is None or counts[part.id] > 1):
            warnings.warn(f'{_unaddressable_reason(part)}; rewriting its prefix instead.', UserWarning, stacklevel=2)
            if not part.content.strip():
                continue
            part = replace(part, on_change='rewrite')
        normalized.append(replace(part, content=part.content.strip()) if part.on_change == 'append' else part)
    return normalized


def _unaddressable_reason(part: InstructionPart) -> str:
    """Why an `on_change='append'` block can't be addressed, and how to fix it."""
    if part.id is not None:
        return (
            f"Instruction blocks share the id {str(part.id)!r} and use `on_change='append'`, so changes can't say "
            'which one they replace. Give each a different `name`'
        )
    if part.name is None:
        return (
            "An instruction block without a `name` uses `on_change='append'`, so its changes can't say what they "
            'replace. Pass `name=` to the instructions decorator or `InstructionPart`'
        )
    return (
        f"Instruction block {part.name!r} uses `on_change='append'`, but its capability or toolset has no `id` to "
        'qualify the name with. Give it an `id`'
    )


def _opens_instruction_window(message: ModelMessage) -> bool:
    """Whether the instruction window changes at `message`: a request recording a baseline, or a compaction."""
    if isinstance(message, ModelResponse):
        return any(isinstance(part, CompactionPart) for part in message.parts)
    return any(isinstance(part, InstructionBaselinePart) for part in message.parts)


def advance_instruction_window(
    window: InstructionBaselinePart | None, message: ModelMessage
) -> InstructionBaselinePart | None:
    """The baseline of the instruction window in force after `message`, given the one in force before it.

    A request recording an [`InstructionBaselinePart`][pydantic_ai.messages.InstructionBaselinePart] opens a
    window; an empty one ends the previous window and opens one with no blocks, whose append-mode blocks are
    then delivered as changes rather than in the prefix. A compaction ends the window, as it does for all
    state derived from history (see [`post_compaction_window`][pydantic_ai.messages.post_compaction_window]).
    """
    if isinstance(message, ModelResponse):
        return None if _opens_instruction_window(message) else window
    for part in message.parts:
        if isinstance(part, InstructionBaselinePart):
            window = part
    return window


def instruction_window(messages: Sequence[ModelMessage]) -> InstructionBaselinePart | None:
    """The baseline of the instruction window `messages` end in, or `None` if they aren't in one."""
    window: InstructionBaselinePart | None = None
    for message in messages:
        window = advance_instruction_window(window, message)
    return window


def instruction_window_start(messages: Sequence[ModelMessage]) -> int:
    """The index of the message the current instruction window began at, or 0 if none has.

    Changes recorded before it belong to an earlier window, which the prefix moved away from.
    """
    return max((index for index, message in enumerate(messages) if _opens_instruction_window(message)), default=0)


def window_prefix(baseline: InstructionBaselinePart, current: Sequence[InstructionPart]) -> list[InstructionPart]:
    """The instruction parts a request continuing `baseline`'s window is sent.

    The append-mode blocks at their baseline values, in their baseline positions, minus empty ones, with the
    rewrite-mode blocks from `current`, which [`continues_window`][pydantic_ai._instructions.continues_window]
    has checked are the ones the baseline records.
    """
    rewritten = iter(part for part in current if part.on_change != 'append')
    return [
        block if isinstance(block, InstructionPart) else next(rewritten)
        for block in baseline.instructions
        if not isinstance(block, InstructionPart) or block.content
    ]


@dataclass
class InstructionValues:
    """Each append-mode block's value as the model last received it, folded forward through a window's changes."""

    values: dict[InstructionId, str | None]

    @classmethod
    def at(cls, baseline: InstructionBaselinePart | None) -> InstructionValues:
        """The values at the start of `baseline`'s window."""
        return cls(
            {
                part.id: part.content or None
                for part in (baseline.instructions if baseline is not None else [])
                if isinstance(part, InstructionPart) and part.id is not None
            }
        )

    def apply(self, change: InstructionDeltaPart) -> str | None:
        """Fold in `change`, returning the value it replaces."""
        assert change.id is not None
        previous = self.values.get(change.id)
        self.values[change.id] = change.content
        return previous


def _recorded_instruction_values(
    messages: Sequence[ModelMessage], baseline: InstructionBaselinePart
) -> dict[InstructionId, str | None] | None:
    """Each append-mode block's value as the model last received it in `baseline`'s window.

    `None` if the window has a change this version can't address, written by a newer one.
    """
    values = InstructionValues.at(baseline)
    in_window = False
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if part is baseline:
                    in_window = True
                elif in_window and isinstance(part, InstructionDeltaPart):
                    if part.id is None:
                        return None
                    values.apply(part)
    return values.values


def continues_window(baseline: InstructionBaselinePart, current: Sequence[InstructionPart]) -> bool:
    """Whether `current` can be sent with `baseline`'s prefix, appending changes to its append-mode blocks.

    Rewrite-mode blocks are sent as they are, so they must be unchanged, in the same order, per the baseline's
    digests; a block whose policy changed is one of them. An append-mode block whose id didn't survive
    deserialization (a source namespace this version doesn't know) can't be tagged, so it can't anchor a
    window either.
    """
    if any(isinstance(block, InstructionPart) and block.id is None for block in baseline.instructions):
        return False
    digests = [block for block in baseline.instructions if isinstance(block, InstructionDigest)]
    rewritten = [part for part in current if part.on_change != 'append']
    return len(digests) == len(rewritten) and all(map(InstructionDigest.matches, digests, rewritten))


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

    recorded = _recorded_instruction_values(messages, baseline) if baseline is not None else None
    if instructions is None:
        # Unset parts preserve the recorded request text and end the window, so its old changes stop applying. A
        # window with no blocks and no changes has nothing to end.
        if baseline is not None and (baseline.instructions or recorded != {}):
            _record_baseline(targets, [])
        return None
    if baseline is not None and (recorded is None or not continues_window(baseline, normalized)):
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
        assert recorded is not None
        current = {part.id: part.content or None for part in normalized if part.on_change == 'append' and part.id}
        changes = [
            _instruction_change(instruction_id, recorded.get(instruction_id), current.get(instruction_id))
            for instruction_id in dict.fromkeys([*recorded, *current])
            if recorded.get(instruction_id) != current.get(instruction_id)
        ]
        if changes:
            for target in targets:
                target.parts = [*target.parts, *changes]

    # Parts keep only their content: each model renders the append-mode ones in `Model.prepare_request`.
    prefix = window_prefix(baseline, normalized)
    for target in targets:
        target.instructions = InstructionPart.join(prefix)
    return prefix


def recorded_prefix(baseline: InstructionBaselinePart, instructions: str) -> list[InstructionPart] | None:
    """The instruction parts a request in `baseline`'s window was sent, rebuilt from its recorded `instructions`.

    The joined text is split into the baseline's blocks at the blank lines that join them: an append-mode block
    must match its recorded content, and a rewrite-mode block must match its digest's hash, so the parts are
    exactly the ones the request was sent. `None` if the text isn't made of the baseline's blocks.
    """
    blocks = [block for block in baseline.instructions if not isinstance(block, InstructionPart) or block.content]

    def split(start: int, index: int) -> list[InstructionPart] | None:
        if index == len(blocks):
            return [] if start == len(instructions) else None
        block = blocks[index]
        last = index == len(blocks) - 1
        if isinstance(block, InstructionPart):
            ends = [start + len(block.content)] if instructions.startswith(block.content, start) else []
        else:
            ends = range(start, len(instructions) + 1)
        for end in ends:
            # Blocks are joined with a blank line, and the last one ends the text.
            if end != len(instructions) if last else not instructions.startswith('\n\n', end):
                continue
            if isinstance(block, InstructionPart):
                part = block
            elif _instruction_text_hash(instructions[start:end]) == block.content_hash:
                part = InstructionPart(content=instructions[start:end], id=block.id, dynamic=block.dynamic)
            else:
                continue
            if (rest := split(end if last else end + 2, index + 1)) is not None:
                return [part, *rest]
        return None

    return split(0, 0)


def _record_baseline(targets: Sequence[ModelRequest], instructions: list[InstructionPart]) -> InstructionBaselinePart:
    """Open a window at this request, or close the current one when `instructions` is empty."""
    baseline = InstructionBaselinePart(
        instructions=[
            replace(part) if part.on_change == 'append' else InstructionDigest.of(part) for part in instructions
        ]
    )
    for target in targets:
        target.parts = [*target.parts, baseline]
    return baseline


def _instruction_change(
    instruction_id: InstructionId, previous: str | None, content: str | None
) -> InstructionDeltaPart:
    """Record a block's change, with the hash of the text it replaces so a model can send it as a diff."""
    base_hash = _instruction_text_hash(previous) if previous is not None and content is not None else None
    return InstructionDeltaPart(id=instruction_id, content=content, base_hash=base_hash)
