"""Unit tests for `_clean_message_history`'s `ModelRequest` merge.

These are unit tests rather than VCR/public-API tests because the fields they
guard (`run_id`, `conversation_id`, `metadata`) are internal bookkeeping that is
never sent to the model — `metadata` is explicitly off the wire — so a recorded
request wouldn't exercise the regression. The merge is a pure internal transform,
so it's asserted directly.
"""

from __future__ import annotations

from pydantic_ai import ModelRequest, UserPromptPart
from pydantic_ai._agent_graph import _clean_message_history  # pyright: ignore[reportPrivateUsage]


def test_clean_message_history_preserves_request_fields_on_merge() -> None:
    """Merging consecutive `ModelRequest`s must carry over `run_id`, `conversation_id`,
    and `metadata`, not silently drop them to `None`.
    """
    first = ModelRequest(
        parts=[UserPromptPart(content='first')],
        run_id='run-1',
        conversation_id='conv-1',
        metadata={'key': 'value'},
    )
    second = ModelRequest(parts=[UserPromptPart(content='second')])

    cleaned = _clean_message_history([first, second])

    assert len(cleaned) == 1
    merged = cleaned[0]
    assert isinstance(merged, ModelRequest)
    assert [p.content for p in merged.parts if isinstance(p, UserPromptPart)] == ['first', 'second']
    assert merged.run_id == 'run-1'
    assert merged.conversation_id == 'conv-1'
    assert merged.metadata == {'key': 'value'}
