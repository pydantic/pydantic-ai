"""BrowserUse capability: delegate open-ended web tasks to an autonomous browser-use agent."""

from pydantic_ai_harness.browser_use._capability import BrowserUse
from pydantic_ai_harness.browser_use._model import ChatModelInput, PydanticAIChatModel, resolve_chat_model
from pydantic_ai_harness.browser_use._settings import BrowserAgentSettings
from pydantic_ai_harness.browser_use._toolset import (
    BrowserAgent,
    BrowserAgentFactory,
    BrowserAgentHistory,
    BrowserTask,
    BrowserUseToolset,
    default_browser_agent,
)

__all__ = [
    'BrowserAgent',
    'BrowserAgentFactory',
    'BrowserAgentHistory',
    'BrowserAgentSettings',
    'BrowserTask',
    'BrowserUse',
    'BrowserUseToolset',
    'ChatModelInput',
    'PydanticAIChatModel',
    'default_browser_agent',
    'resolve_chat_model',
]
