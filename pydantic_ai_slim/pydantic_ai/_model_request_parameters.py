from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from . import _utils
from .builtin_tools import AbstractBuiltinTool

if TYPE_CHECKING:
    from .tools import ToolDefinition
else:  # pragma: no cover
    ToolDefinition = Any

if TYPE_CHECKING:
    from ._output import OutputObjectDefinition
    from .output import OutputMode

__all__ = ('ModelRequestParameters',)


@dataclass(repr=False, kw_only=True)
class ModelRequestParameters:
    """Configuration for an agent's request to a model, specifically related to tools and output handling."""

    function_tools: list[ToolDefinition] = field(default_factory=list)
    builtin_tools: list[AbstractBuiltinTool] = field(default_factory=list)

    output_mode: OutputMode = 'text'
    output_object: OutputObjectDefinition | None = None
    output_tools: list[ToolDefinition] = field(default_factory=list)
    prompted_output_template: str | None = None
    allow_text_output: bool = True
    allow_image_output: bool = False

    @cached_property
    def tool_defs(self) -> dict[str, ToolDefinition]:
        return {tool_def.name: tool_def for tool_def in [*self.function_tools, *self.output_tools]}

    @cached_property
    def prompted_output_instructions(self) -> str | None:
        if self.output_mode == 'prompted' and self.prompted_output_template and self.output_object:
            from ._output import PromptedOutputSchema

            return PromptedOutputSchema.build_instructions(self.prompted_output_template, self.output_object)
        return None

    __repr__ = _utils.dataclasses_no_defaults_repr
