"""Claude's `TodoWrite` tool — record the agent's task checklist.

For our headless / autonomous shim, the checklist is a no-op acknowledgement
(no UI to render it). Returning a structured ack so the model's planning step
still sees its checklist confirmed.
"""

from typing_extensions import TypedDict


class TodoItem(TypedDict):
    """One entry for `TodoWrite` (Claude's todo schema)."""

    content: str
    status: str
    activeForm: str


def todo_write(todos: list[TodoItem]) -> str:
    """Record the agent's task checklist."""
    summary = '\n'.join(f'  - [{t.get("status", "?")[:1]}] {t.get("content", "")}' for t in todos)
    return f'todos recorded ({len(todos)}):\n{summary}'
