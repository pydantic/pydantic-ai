"""A snapshot saved before tool results recorded their files still continues with those files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from pydantic_ai.messages import (
    DocumentUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
)
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, FileStepStore


def _without_file_records(node: Any) -> Any:
    """`node` as an earlier version saved it, before tool results recorded their files."""
    if isinstance(node, dict):
        mapping = cast(dict[str, Any], node)
        return {key: _without_file_records(value) for key, value in mapping.items() if key != 'content_file_digests'}
    if isinstance(node, list):
        return [_without_file_records(item) for item in cast(list[Any], node)]
    return node


async def test_snapshot_saved_before_file_records_keeps_its_tool_files(tmp_path: Path) -> None:
    files = [UploadedFile('file-1', 'openai'), DocumentUrl('s3://bucket/report.pdf')]
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='read it')]),
        ModelResponse(parts=[ToolCallPart(tool_name='read', tool_call_id='c')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='read', content=files, tool_call_id='c')]),
    ]
    store = FileStepStore(tmp_path / 'runs')
    await store.save_snapshot(ContinuableSnapshot(run_id='run', step_index=0, messages=messages, state='complete'))
    for path in (tmp_path / 'runs' / 'run' / 'snapshots').glob('*.json'):
        path.write_text(json.dumps(_without_file_records(json.loads(path.read_text()))))

    snapshot = await store.latest_snapshot(run_id='run')

    assert snapshot is not None
    tool_return = snapshot.messages[-1].parts[0]
    assert isinstance(tool_return, ToolReturnPart)
    assert tool_return.content == files
