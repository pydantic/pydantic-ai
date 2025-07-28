import pytest
from typing import Any, cast

from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart

from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from openai.types.responses import Response, ResponseOutput, ResponseOutputItemFunctionToolCall
    from openai.types.shared import FunctionDefinition
    
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

@pytest.mark.vcr()
async def test_openai_responses_missing_finish_reason(allow_model_requests: None, openai_api_key: str):
    """Test OpenAI Responses API with tool calls to ensure it works correctly.
    
    While this doesn't directly test the missing finish_reason fix (which is primarily
    an issue in the regular API), it ensures tool calls work properly in the responses API
    which uses a different handling mechanism.
    """    
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model)

    @agent.tool_plain
    async def search_tool(query: str) -> str:
        """A simple search tool that returns a fixed response."""
        return f"Results for: {query}"

    # Run a query that should trigger a tool call
    result = await agent.run("Search for pydantic-ai library")
    
    # Verify that at least one tool call was made
    tool_call_found = False
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart) and part.tool_name == 'search_tool':
                tool_call_found = True
                break
    
    assert tool_call_found, "No tool call was made"
    # The actual text response will vary, but we should have gotten a valid response
    assert result.output is not None