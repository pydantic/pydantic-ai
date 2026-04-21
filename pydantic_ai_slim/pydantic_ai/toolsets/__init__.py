from typing import Union

from .._run_context import AgentDepsT
from ._dynamic import ToolsetFunc
from .abstract import AbstractToolset, ToolsetTool
from .apply_patch import ApplyPatchToolset
from .approval_required import ApprovalRequiredToolset
from .combined import CombinedToolset
from .deferred_loading import DeferredLoadingToolset
from .external import DeferredToolset, ExternalToolset  # pyright: ignore[reportDeprecated]
from .filtered import FilteredToolset
from .function import FunctionToolset
from .include_return_schemas import IncludeReturnSchemasToolset
from .prefixed import PrefixedToolset
from .prepared import PreparedToolset
from .renamed import RenamedToolset
from .set_metadata import SetMetadataToolset
from .shell import ShellToolset
from .text_editor import TextEditorToolset
from .wrapper import WrapperToolset

AgentToolset = Union[AbstractToolset[AgentDepsT], ToolsetFunc[AgentDepsT]]  # noqa: UP007 — Union needed at runtime (no future annotations)
"""A toolset or a factory function that creates a toolset from a run context."""

__all__ = (
    'AbstractToolset',
    'AgentToolset',
    'ToolsetFunc',
    'ToolsetTool',
    'ApprovalRequiredToolset',
    'CombinedToolset',
    'DeferredLoadingToolset',
    'DeferredToolset',
    'ExternalToolset',
    'FilteredToolset',
    'FunctionToolset',
    'IncludeReturnSchemasToolset',
    'PrefixedToolset',
    'PreparedToolset',
    'RenamedToolset',
    'SetMetadataToolset',
    'ShellToolset',
    'TextEditorToolset',
    'ApplyPatchToolset',
    'WrapperToolset',
)
