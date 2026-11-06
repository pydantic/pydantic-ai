"""Safe-output completion checks for gh-aw runs.

gh-aw's `safeoutputs` MCP server appends every safe output to
`GH_AW_SAFE_OUTPUTS`. `output_type=str` ends a run on any text-only response,
so a run that narrates instead of publishing ends with an empty sink and gh-aw
reports "produced no safe outputs". The output validator below sends the model
back to work in that case.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator
from typing import Annotated

from pydantic import Field, TypeAdapter, ValidationError

from ._config import is_attention_run, is_review_run

SAFE_OUTPUT_TOOL_PREFIXES = ('safeoutputs_', 'mcp__safeoutputs__')
"""Tool-name prefixes the engine's gateway and the legacy shim use for safe outputs."""

SAFE_OUTPUT_ENTRY_ADAPTER = TypeAdapter(dict[str, object])


def safe_output_type(tool_name: str) -> str | None:
    """The safe-output type a safe-output tool writes, or `None` for any other tool."""
    for prefix in SAFE_OUTPUT_TOOL_PREFIXES:
        if tool_name.startswith(prefix):
            return tool_name.removeprefix(prefix)
    return None


def _safe_output_entries(contents: str) -> Iterator[dict[str, object]]:
    for line in contents.splitlines():
        try:
            yield SAFE_OUTPUT_ENTRY_ADAPTER.validate_json(line)
        except ValidationError:
            continue


def safe_output_status() -> tuple[bool, list[int] | None]:
    """Return whether output is pending and any Attention candidate IDs missing decisions."""
    attention = is_attention_run()
    review = is_review_run()
    path = os.environ.get('GH_AW_SAFE_OUTPUTS')
    if not path:
        if attention or review:
            workflow_kind = 'Attention' if attention else 'Review'
            raise ValueError(f'{workflow_kind} safe-output sink is not configured')
        return False, None
    sink = pathlib.Path(path)
    contents = sink.read_text(encoding='utf-8') if sink.is_file() else ''
    sink_empty = not contents.strip()
    if review:
        review_submitted = any(
            entry.get('type') == 'submit_pull_request_review' for entry in _safe_output_entries(contents)
        )
        return not review_submitted, None
    if not attention or sink_empty:
        return sink_empty, None

    workspace = pathlib.Path(os.environ.get('GITHUB_WORKSPACE', '.'))
    snapshot_path = workspace / 'attention-candidates.json'
    if not snapshot_path.is_file():
        raise ValueError('Attention candidate snapshot is missing')

    snapshot = SAFE_OUTPUT_ENTRY_ADAPTER.validate_json(snapshot_path.read_text(encoding='utf-8'))
    candidate_rows = TypeAdapter(list[dict[str, object]]).validate_python(snapshot['candidates'])
    candidate_numbers = TypeAdapter(list[Annotated[int, Field(ge=1, strict=True)]]).validate_python(
        [candidate['number'] for candidate in candidate_rows]
    )
    if len(candidate_numbers) > 10 or len(candidate_numbers) != len(set(candidate_numbers)):
        raise ValueError('Attention snapshot contains too many or duplicate candidates')
    candidates = set(candidate_numbers)
    if not candidates:
        return sink_empty, []

    recorded: set[int] = set()
    for entry in _safe_output_entries(contents):
        item_number = entry.get('item_number')
        if entry.get('type') == 'record_attention_decision' and isinstance(item_number, str):
            try:
                parsed_number = int(item_number)
            except ValueError:
                continue
            if parsed_number > 0 and str(parsed_number) == item_number:
                recorded.add(parsed_number)
    missing = sorted(candidates - recorded)
    return bool(missing), missing


def safe_output_pending() -> bool:
    """Whether a gh-aw run is missing safe output or Attention decisions."""
    try:
        output_pending, _ = safe_output_status()
    except (KeyError, OSError, ValueError):
        if is_attention_run() or is_review_run():
            return True
        raise
    return output_pending
