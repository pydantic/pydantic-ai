"""Complete coding-agent harness assembled from regular capabilities."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import json_repair
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, Capability, CombinedCapability, RawToolArgs
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import AgentDepsT, ToolDefinition

from pydantic_ai_harness.coder._instructions import INSTRUCTIONS
from pydantic_ai_harness.compaction import ClearToolResults, WarnNearLimits
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, MAX_FOREGROUND_WAIT, Shell
from pydantic_ai_harness.tool_output_limits import Band, ToolOutputLimits, Truncate

FILE_TOOL_NAMES: tuple[str, ...] = ('read_file', 'write_file', 'edit_file', 'list_files', 'grep')
"""The `FileSystem` tools `Coder` registers; `shell` covers directory creation, file metadata, and the rest."""


class _RepairToolArguments(AbstractCapability[AgentDepsT]):
    async def before_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
    ) -> RawToolArgs:
        """Repair malformed JSON arguments before normal tool schema validation."""
        if not isinstance(args, str):
            return args
        try:
            json.loads(args)
        except (json.JSONDecodeError, RecursionError):
            with ctx.tracer.start_as_current_span('coder.repair_tool_arguments'):
                try:
                    return json_repair.repair_json(args, skip_json_loads=True, ensure_ascii=False)
                except (ValueError, RecursionError):
                    return args
        return args


class _BoundToolOutputs(ToolOutputLimits[AgentDepsT]):
    id: str | None = None

    def get_toolset(self) -> None:
        """Coder uses bounded truncation, so no spill-retrieval tool is needed."""
        return None


MAX_READ_CHARS = 60000
"""Characters of complete lines per `read_file`, kept under `MAX_OUTPUT_CHARS` so the output cap never cuts a read."""

MAX_OUTPUT_CHARS = 64000
"""Characters kept from any tool result."""


def _file_system(workspace: Path, *, unrestricted: bool) -> FileSystem[AgentDepsT]:
    file_system = FileSystem[AgentDepsT](
        root_dir=workspace, content_hashes=False, max_read_chars=MAX_READ_CHARS, tools=FILE_TOOL_NAMES
    )
    if unrestricted:
        return replace(file_system, root_dir=workspace.anchor, cwd=workspace, protected_patterns=[])
    return file_system


class Coder(CombinedCapability[AgentDepsT]):
    """Autonomous local coding with six tools and context management.

    Commands are unrestricted and can outlive runs. Use an OS sandbox for
    untrusted work. Additional instructions supplement the default guidance.
    """

    def __init__(
        self,
        workspace: str | Path = '.',
        *,
        instructions: str | None = None,
        unrestricted_filesystem: bool = False,
    ) -> None:
        root = Path(workspace).resolve()
        super().__init__(
            [
                _RepairToolArguments[AgentDepsT](),
                Capability[AgentDepsT](instructions=INSTRUCTIONS + ('\n' + instructions if instructions else '')),
                _file_system(root, unrestricted=unrestricted_filesystem),
                Shell[AgentDepsT](
                    cwd=root,
                    denied_commands=[],
                    default_timeout=MAX_FOREGROUND_WAIT,
                    allow_interactive=True,
                    denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
                    tools=['shell'],
                ),
                RepoContext[AgentDepsT](workspace_dir=root, expose_inventory_tool=False),
                ClearToolResults[AgentDepsT](max_fraction=0.7),
                WarnNearLimits[AgentDepsT](max_context_fraction=0.9),
                _BoundToolOutputs[AgentDepsT](
                    id=None, bands=[Band(over=MAX_OUTPUT_CHARS, action=Truncate(max_chars=MAX_OUTPUT_CHARS))]
                ),
            ]
        )
