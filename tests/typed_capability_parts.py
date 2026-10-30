"""Pins how type checkers see capability parts: which content speaks in the system voice, and who decides trust.

`reportUnnecessaryTypeIgnoreComment` is on, so an ignore below that stops being needed fails the type check.
Nothing here is executed.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_type

from pydantic_ai.messages import (
    CapabilityPart,
    FramedCapabilityPart,
    OperatorSource,
    OperatorText,
    SystemCapabilityPart,
    UserContent,
    WorkspaceSource,
)
from pydantic_ai.workspaces import Workspace


@dataclass(kw_only=True)
class NotePart(FramedCapabilityPart, namespace='typed', tag='note'):
    pass


@dataclass(kw_only=True)
class ReminderPart(SystemCapabilityPart, namespace='typed'):
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


def workspace_sources(workspace: Workspace) -> None:
    # Trust is the workspace's to say: a source can only be built from one.
    source = WorkspaceSource.from_workspace(workspace, 'AGENTS.md')
    assert_type(source.trusted, bool)
    WorkspaceSource(path='AGENTS.md', trusted=True)  # pyright: ignore[reportCallIssue]


def reading_parts(part: CapabilityPart) -> None:
    # The base has no content of its own: narrow to a voice to read it.
    part.content  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
    if isinstance(part, FramedCapabilityPart):
        assert_type(part.content, str | Sequence[UserContent])
