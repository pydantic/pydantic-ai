"""Model and builtin tool configurations for the web chat UI."""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic.alias_generators import to_camel

from pydantic_ai.builtin_tools import (
    BUILTIN_TOOL_ID,
    AbstractBuiltinTool,
    CodeExecutionTool,
    ImageGenerationTool,
    WebSearchTool,
)

AIModelID = Literal[
    'anthropic:claude-sonnet-4-5',
    'openai-responses:gpt-5',
    'google-gla:gemini-2.5-pro',
]


class AIModel(BaseModel, alias_generator=to_camel, populate_by_name=True):
    """Defines an AI model with its associated built-in tools."""

    id: AIModelID
    name: str
    builtin_tools: list[BUILTIN_TOOL_ID]


class BuiltinToolDef(BaseModel):
    """Defines a built-in tool.

    Used by the web chat UI to display tool options.
    """

    id: BUILTIN_TOOL_ID
    name: str
    tool: AbstractBuiltinTool = Field(exclude=True)


_default_tool_ids: list[BUILTIN_TOOL_ID] = ['web_search', 'code_execution', 'image_generation']

_id_to_ui_name: dict[BUILTIN_TOOL_ID, str] = {
    'web_search': 'Web Search',
    'code_execution': 'Code Execution',
    'image_generation': 'Image Generation',
}

_id_to_builtin_tool: dict[BUILTIN_TOOL_ID, AbstractBuiltinTool] = {
    'web_search': WebSearchTool(),
    'code_execution': CodeExecutionTool(),
    'image_generation': ImageGenerationTool(),
}

DEFAULT_BUILTIN_TOOL_DEFS: list[BuiltinToolDef] = [
    BuiltinToolDef(id=tool_id, name=_id_to_ui_name[tool_id], tool=_id_to_builtin_tool[tool_id])
    for tool_id in _default_tool_ids
]


AI_MODELS: list[AIModel] = [
    AIModel(
        id='anthropic:claude-sonnet-4-5',
        name='Claude Sonnet 4.5',
        builtin_tools=[
            'web_search',
            'code_execution',
        ],
    ),
    AIModel(
        id='openai-responses:gpt-5',
        name='GPT 5',
        builtin_tools=[
            'web_search',
            'code_execution',
            'image_generation',
        ],
    ),
    AIModel(
        id='google-gla:gemini-2.5-pro',
        name='Gemini 2.5 Pro',
        builtin_tools=[
            'web_search',
            'code_execution',
        ],
    ),
]
