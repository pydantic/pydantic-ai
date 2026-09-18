"""Complete coding-agent harness assembled from regular capabilities."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability, Capability, CombinedCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.coder._instructions import INSTRUCTIONS
from pydantic_ai_harness.compaction import ClearToolResults, WarnNearLimits
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.repair_tool_arguments import RepairToolArguments
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, MAX_FOREGROUND_WAIT, Shell
from pydantic_ai_harness.tool_output_limits import Band, ToolOutputLimits, Truncate

FILE_TOOL_NAMES: tuple[str, ...] = ('read_file', 'write_file', 'edit_file', 'list_files', 'grep')
"""The `FileSystem` tools `Coder` registers; `shell` covers directory creation, file metadata, and the rest."""


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
    `repo_context=False` leaves out the bundled `RepoContext`, for hosts that
    bind their own and would otherwise load the instruction files twice.
    """

    def __init__(
        self,
        workspace: str | Path = '.',
        *,
        instructions: str | None = None,
        unrestricted_filesystem: bool = False,
        repo_context: bool = True,
    ) -> None:
        root = Path(workspace).resolve()
        capabilities: list[AbstractCapability[AgentDepsT]] = [
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
        ]
        if repo_context:
            capabilities.append(RepoContext[AgentDepsT](workspace_dir=root, expose_inventory_tool=False))
        capabilities += [
            ClearToolResults[AgentDepsT](max_fraction=0.7),
            WarnNearLimits[AgentDepsT](max_context_fraction=0.9),
            _BoundToolOutputs[AgentDepsT](
                id=None, bands=[Band(over=MAX_OUTPUT_CHARS, action=Truncate(max_chars=MAX_OUTPUT_CHARS))]
            ),
            RepairToolArguments[AgentDepsT](),
        ]
        super().__init__(capabilities)
