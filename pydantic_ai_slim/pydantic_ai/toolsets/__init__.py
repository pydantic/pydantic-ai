from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._dynamic import ToolsetFunc
from .abstract import AbstractToolset, ToolsetTool
from .approval_required import ApprovalRequiredToolset
from .combined import CombinedToolset
from .external import DeferredToolset, ExternalToolset  # pyright: ignore[reportDeprecated]
from .filtered import FilteredToolset
from .function import FunctionToolset
from .prefixed import PrefixedToolset
from .prepared import PreparedToolset
from .renamed import RenamedToolset
from .wrapper import WrapperToolset

if TYPE_CHECKING:
    from .code_mode import CodeModeToolset

__all__ = (
    'AbstractToolset',
    'ToolsetFunc',
    'ToolsetTool',
    'CodeModeToolset',
    'CombinedToolset',
    'ExternalToolset',
    'DeferredToolset',
    'FilteredToolset',
    'FunctionToolset',
    'PrefixedToolset',
    'RenamedToolset',
    'PreparedToolset',
    'WrapperToolset',
    'ApprovalRequiredToolset',
)


def __getattr__(name: str) -> Any:
    if name == 'CodeModeToolset':
        from .code_mode import CodeModeToolset

        return CodeModeToolset
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
