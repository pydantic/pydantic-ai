"""Tool output limits: reduce oversized tool returns at production time.

`ToolOutputLimits` intercepts a tool return when it is produced and reduces it --
truncating, spilling to a queryable file, or summarizing -- so an oversized payload does
not persist in history and get re-sent on every later model request. Combine the three
modes through an ordered list of size `bands`.

Spilled payloads are written to the run's workspace by default (`WorkspaceOverflowStore`) and read
back on demand through the registered `read_tool_result` tool, or through `FileSystem`'s
`read_file`, which supersedes it for spills in the run's workspace; the `OverflowStore`
protocol is the seam for any other backend.
"""

from pydantic_ai_harness._warn import warn_class_renamed
from pydantic_ai_harness.tool_output_limits._bands import (
    Action,
    Band,
    Passthrough,
    Spill,
    Summarize,
    SummarizeFunc,
    Truncate,
)
from pydantic_ai_harness.tool_output_limits._capability import READ_TOOL_NAME, READ_TOOL_RESULT_KIND, ToolOutputLimits
from pydantic_ai_harness.tool_output_limits._payload import (
    Serializer,
    TruncationStrategy,
    indented_json,
    json_lines,
)
from pydantic_ai_harness.tool_output_limits._store import LocalFileOverflowStore, OverflowStore, WorkspaceOverflowStore

__all__ = [
    'READ_TOOL_NAME',
    'READ_TOOL_RESULT_KIND',
    'Action',
    'Band',
    'LocalFileOverflowStore',
    'OverflowStore',
    'Serializer',
    'ToolOutputLimits',
    'Passthrough',
    'Spill',
    'Summarize',
    'SummarizeFunc',
    'Truncate',
    'TruncationStrategy',
    'WorkspaceOverflowStore',
    'indented_json',
    'json_lines',
]

_RENAMED = {'LocalFileStore': LocalFileOverflowStore, 'WorkspaceStore': WorkspaceOverflowStore}


def __getattr__(name: str) -> object:
    renamed = _RENAMED.get(name)
    if renamed is not None:
        warn_class_renamed(name, renamed.__name__, __name__)
        return renamed
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
