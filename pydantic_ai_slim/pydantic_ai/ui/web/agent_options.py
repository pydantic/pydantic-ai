"""Model and builtin tool configurations for the web chat UI."""

from typing import Literal

from pydantic import BaseModel

from pydantic_ai.builtin_tools import (
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
BuiltinToolID = Literal['web_search', 'image_generation', 'code_execution']


class AIModel(BaseModel):
    """Defines an AI model with its associated built-in tools."""

    id: str
    name: str
    builtin_tools: list[str]


class BuiltinTool(BaseModel):
    """Defines a built-in tool."""

    id: str
    name: str


BUILTIN_TOOL_DEFS: list[BuiltinTool] = [
    BuiltinTool(id='web_search', name='Web Search'),
    BuiltinTool(id='code_execution', name='Code Execution'),
    BuiltinTool(id='image_generation', name='Image Generation'),
]

BUILTIN_TOOLS: dict[str, AbstractBuiltinTool] = {
    'web_search': WebSearchTool(),
    'code_execution': CodeExecutionTool(),
    'image_generation': ImageGenerationTool(),
}

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
