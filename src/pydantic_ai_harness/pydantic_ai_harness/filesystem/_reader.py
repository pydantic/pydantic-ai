"""How a capability that keeps files in the workspace finds a tool to read them."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic_ai.tools import RunContext

READ_CHARS = 50_000
"""Characters per read that a capability keeping files in the workspace asks a file reader to stay within.

`ToolOutputLimits` caps `read_tool_result` at this and asks for it when handing its spills to
`read_file`; `Coder` sets `max_read_chars` to it so its `read_file` qualifies.
"""


@runtime_checkable
class _FileReader(Protocol):
    """A capability with a tool that reads files in the run's workspace.

    A capability that keeps files in the workspace, such as `ToolOutputLimits` spilling
    oversized results, asks `find_file_reader` whether an active one can read them. If one can,
    it points the model at that tool instead of offering a reader tool of its own. `FileSystem`
    implements it with `read_file`.
    """

    def _file_read_tool(self, ctx: RunContext[Any], path: str, *, max_chars: int) -> str | None:
        """Name the tool that reads the file at `path`, or return `None`.

        `path` is relative to the workspace's working directory. The tool must return at most
        `max_chars` characters per call. Answer from configuration alone, without workspace I/O:
        the question is asked before each model request, where I/O could start a sandbox. When
        the configuration can't settle it, return `None`.
        """
        ...  # pragma: no cover


def find_file_reader(ctx: RunContext[Any], path: str, *, max_chars: int) -> str | None:
    """The tool of the first active `_FileReader` that reads the file at `path`, else `None`.

    `path` is relative to the workspace's working directory. A deferred capability counts once the
    model has loaded it.
    """
    active = ctx.active_capability_ids
    for key, capability in ctx.capabilities.items():
        if key in active and isinstance(capability, _FileReader):
            tool = capability._file_read_tool(ctx, path, max_chars=max_chars)  # pyright: ignore[reportPrivateUsage]
            if tool is not None:
                return tool
    return None
