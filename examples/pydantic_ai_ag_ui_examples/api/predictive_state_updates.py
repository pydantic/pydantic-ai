"""Predictive State feature."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ag_ui.core import CustomEvent, EventType
from pydantic import BaseModel

from pydantic_ai.ag_ui import FastAGUI, StateDeps

from .agent import agent

if TYPE_CHECKING:  # pragma: no cover
    from pydantic_ai import RunContext


_LOGGER: logging.Logger = logging.getLogger(__name__)


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


app: FastAGUI = agent(deps=StateDeps(DocumentState()))


# Tools which return AG-UI events will be sent to the client as part of the
# event stream, single events and iterables of events are supported.
@app.adapter.agent.tool_plain
def document_predict_state() -> list[CustomEvent]:
    """Enable document state prediction.

    Returns:
        CustomEvent containing the event to enable state prediction.
    """
    _LOGGER.info('enabling document state state prediction')
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


@app.adapter.agent.instructions()
def story_instructions(ctx: RunContext[StateDeps[DocumentState]]) -> str:
    """Provide instructions for writing document if present.

    Args:
        ctx: The run context containing document state information.

    Returns:
        Instructions string for the document writing agent.
    """
    _LOGGER.info('story instructions document=%s', ctx.deps.state.document)

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
