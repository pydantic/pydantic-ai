"""Deprecated import path for native tool classes.

!!! warning "Deprecated import path"
    The `pydantic_ai.builtin_tools` namespace was renamed to `pydantic_ai.native_tools`
    in pydantic-ai 1.x. Import these symbols from `pydantic_ai.native_tools` instead:

    ```python {test="skip" lint="skip"}
    # Before
    from pydantic_ai.builtin_tools import WebSearchTool, AbstractBuiltinTool

    # After
    from pydantic_ai.native_tools import WebSearchTool, AbstractNativeTool
    ```

    `AbstractBuiltinTool`, the `BUILTIN_TOOL_*` registries, and the
    `pydantic_ai.builtin_tools` namespace will be removed in v2.
"""

from __future__ import annotations as _annotations

import warnings
from typing import TYPE_CHECKING, Any

from pydantic_ai._warnings import PydanticAIDeprecationWarning

if TYPE_CHECKING:
    from ..native_tools import (
        DEPRECATED_NATIVE_TOOLS as DEPRECATED_BUILTIN_TOOLS,
        NATIVE_TOOL_TYPES as BUILTIN_TOOL_TYPES,
        NATIVE_TOOLS_REQUIRING_CONFIG as BUILTIN_TOOLS_REQUIRING_CONFIG,
        SUPPORTED_NATIVE_TOOLS as SUPPORTED_BUILTIN_TOOLS,
        AbstractNativeTool as AbstractBuiltinTool,
        CodeExecutionTool,
        FileSearchTool,
        ImageAspectRatio,
        ImageGenerationModelName,
        ImageGenerationTool,
        MCPServerTool,
        MemoryTool,
        UrlContextTool,  # pyright: ignore[reportDeprecated]
        WebFetchTool,
        WebSearchTool,
        WebSearchUserLocation,
        XSearchTool,
    )

__all__ = (
    'AbstractBuiltinTool',
    'WebSearchTool',
    'WebSearchUserLocation',
    'XSearchTool',
    'CodeExecutionTool',
    'WebFetchTool',
    'UrlContextTool',
    'ImageGenerationModelName',
    'ImageGenerationTool',
    'ImageAspectRatio',
    'MemoryTool',
    'MCPServerTool',
    'FileSearchTool',
    'BUILTIN_TOOL_TYPES',
    'DEPRECATED_BUILTIN_TOOLS',
    'SUPPORTED_BUILTIN_TOOLS',
    'BUILTIN_TOOLS_REQUIRING_CONFIG',
)


# Old name → new name in `pydantic_ai.native_tools`. Names not in this dict
# are unchanged in the rename and just need a path-deprecation warning.
_RENAMES: dict[str, str] = {
    'AbstractBuiltinTool': 'AbstractNativeTool',
    'BUILTIN_TOOL_TYPES': 'NATIVE_TOOL_TYPES',
    'DEPRECATED_BUILTIN_TOOLS': 'DEPRECATED_NATIVE_TOOLS',
    'SUPPORTED_BUILTIN_TOOLS': 'SUPPORTED_NATIVE_TOOLS',
    'BUILTIN_TOOLS_REQUIRING_CONFIG': 'NATIVE_TOOLS_REQUIRING_CONFIG',
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    # Lazy import to avoid the warning firing on package import.
    import pydantic_ai.native_tools as _native_tools

    if name in _RENAMES:
        new_name = _RENAMES[name]
        warnings.warn(
            f'`pydantic_ai.builtin_tools.{name}` is deprecated, use `pydantic_ai.native_tools.{new_name}` instead.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        return getattr(_native_tools, new_name)

    warnings.warn(
        f'Importing `{name}` from `pydantic_ai.builtin_tools` is deprecated, '
        f'import it from `pydantic_ai.native_tools` instead.',
        PydanticAIDeprecationWarning,
        stacklevel=2,
    )
    return getattr(_native_tools, name)
