"""Message histories shared by the step store tests."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart


def growing_histories(turns: int) -> list[list[ModelMessage]]:
    """The history after each turn; each is a prefix of the next, as across a conversation's runs."""
    histories: list[list[ModelMessage]] = []
    history: list[ModelMessage] = []
    for turn in range(turns):
        history = [
            *history,
            ModelRequest(parts=[UserPromptPart(content=f'question {turn}')]),
            ModelResponse(parts=[TextPart(content=f'answer {turn}')]),
        ]
        histories.append(history)
    return histories
