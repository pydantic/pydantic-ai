"""Predictive State feature."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ag_ui.core import CustomEvent, EventType
from dotenv import load_dotenv
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.ag_ui import AGUIApp, StateDeps

if TYPE_CHECKING:  # pragma: no cover
    from pydantic_ai import RunContext


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


# Ensure environment variables are loaded.
load_dotenv()

agent: Agent = Agent(
    'openai:gpt-4o-mini',
    output_type=str,
    deps_type=StateDeps[DocumentState],
)

app: AGUIApp = agent.to_ag_ui(deps=StateDeps(DocumentState()))


# Tools which return AG-UI events will be sent to the client as part of the
# event stream, single events and iterables of events are supported.
@agent.tool_plain
def document_predict_state() -> list[CustomEvent]:
    """Enable document state prediction.

    Returns:
        CustomEvent containing the event to enable state prediction.
    """
    return [
        CustomEvent(
            type=EventType.CUSTOM,
            name='PredictState',
            value=[
                {
                    'state_key': 'document',
                    'tool': 'write_document',
                    'tool_argument': 'document',
                },
            ],
        ),
    ]


@agent.instructions()
def story_instructions(ctx: RunContext[StateDeps[DocumentState]]) -> str:
    """Provide instructions for writing document if present.

    Args:
        ctx: The run context containing document state information.

    Returns:
        Instructions string for the document writing agent.
    """
    return f"""You are a helpful assistant for writing documents.

Before you start writing, you MUST call the `document_predict_state`
tool to enable state prediction.

To present the document to the user for review, you MUST use the
`write_document` tool.

When you have written the document, DO NOT repeat it as a message.
If accepted briefly summarize the changes you made, 2 sentences
max, otherwise ask the user to clarify what they want to change.

This is the current document:

{ctx.deps.state.document}
"""
