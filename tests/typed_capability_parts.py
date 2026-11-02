"""Pins how type checkers see capability parts: which content speaks in the system voice, and who decides authority.

`reportUnnecessaryTypeIgnoreComment` is on, so an ignore below that stops being needed fails the type check.
Nothing here is executed.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_type

from pydantic_ai.messages import (
    Authority,
    CapabilityPart,
    DataCapabilityPart,
    InstructionCapabilityPart,
    ModelSource,
    OperatorSource,
    OperatorText,
    StructuredContent,
    UserContent,
    UserSource,
    WorkspaceSource,
)
from pydantic_ai.workspaces import Workspace


@dataclass(kw_only=True)
class NotePart(DataCapabilityPart, namespace='typed', tag='note'):
    pass


@dataclass(kw_only=True)
class TextNotePart(DataCapabilityPart[str], namespace='typed', tag='text_note'):
    pass


@dataclass(kw_only=True)
class ReminderPart(InstructionCapabilityPart, namespace='typed'):
    pass


def system_content(fetched: str, left: int) -> None:
    # A literal, or text the code marks as the operator's, speaks in the system voice.
    ReminderPart(content='Be brief.', source=OperatorSource())
    ReminderPart(content=OperatorText(f'{left} requests left.'), source=OperatorSource())
    # Anything else doesn't: a string from elsewhere, or one built from a value, isn't known to be the operator's.
    ReminderPart(content=fetched, source=OperatorSource())  # pyright: ignore[reportArgumentType]
    ReminderPart(content=f'{left} requests left.', source=OperatorSource())  # pyright: ignore[reportArgumentType]
    ReminderPart(content=['Be brief.'], source=OperatorSource())  # pyright: ignore[reportArgumentType]


def framed_content(fetched: str, items: list[UserContent]) -> None:
    # Data takes any user content.
    NotePart(content=fetched, source=OperatorSource())
    NotePart(content=items, source=OperatorSource())
    # Unless the kind narrows it.
    note = TextNotePart(content=fetched, source=OperatorSource())
    assert_type(note.content, str)
    TextNotePart(content=items, source=OperatorSource())  # pyright: ignore[reportArgumentType]


def workspace_sources(workspace: Workspace) -> None:
    # Authority is the workspace's to say: a source can only be built from one.
    source = WorkspaceSource.from_workspace(workspace, 'AGENTS.md')
    assert_type(source.authority, Authority)
    WorkspaceSource(path='AGENTS.md', authority='operator')  # pyright: ignore[reportCallIssue]


def source_authorities() -> None:
    # Every source has a typed authority.
    assert_type(OperatorSource().authority, Authority)
    assert_type(UserSource().authority, Authority)
    assert_type(ModelSource('trajectory judge').authority, Authority)


def reading_parts(part: CapabilityPart) -> None:
    # The base has no `content` field: read `model_content()`, or narrow to a voice.
    part.content  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
    assert_type(part.model_content(), str | StructuredContent | Sequence[UserContent])
    if isinstance(part, DataCapabilityPart):
        # Narrowed to the generic base, `content`'s type argument is unknown: `model_content()` reads it typed.
        assert_type(part.model_content(), str | StructuredContent | Sequence[UserContent])
    if isinstance(part, TextNotePart):
        assert_type(part.content, str)
