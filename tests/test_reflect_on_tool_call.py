"""Tests for reflect_on_tool_call functionality."""

import json
from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

pytestmark = pytest.mark.anyio


class TestReflectOnToolCall:
    """Tests for the reflect_on_tool_call parameter."""

    @pytest.mark.parametrize('reflect_on_tool_call', [True, False])
    def test_agent_constructor_reflects_parameter(self, reflect_on_tool_call: bool) -> None:
        """Test that the reflect_on_tool_call parameter is stored in the agent."""
        agent = Agent(TestModel(), reflect_on_tool_call=reflect_on_tool_call)
        
        # pyright: reportPrivateUsage=false
        assert agent._reflect_on_tool_call is reflect_on_tool_call

    def test_agent_default_to_reflect(self) -> None:
        """Test that by default, reflect_on_tool_call is True."""
        agent = Agent(TestModel())
        
        # pyright: reportPrivateUsage=false
        assert agent._reflect_on_tool_call is True

    def test_with_reflection(self) -> None:
        """Test normal behavior with reflection enabled (default behavior)."""
        def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                # First call - return a tool call
                return ModelResponse(parts=[ToolCallPart('get_data', {'param': 'test'})])
            elif len(messages) == 3:
                # After tool return - return a summary
                return ModelResponse(parts=[TextPart('Data retrieved: test_data')])
            else:
                # This shouldn't happen in this test
                return ModelResponse(parts=[TextPart('Unexpected message sequence')])
                
        agent = Agent(FunctionModel(model_func), reflect_on_tool_call=True)
        
        @agent.tool_plain
        def get_data(param: str) -> str:
            return f'test_data: {param}'
            
        with capture_run_messages() as messages:
            result = agent.run_sync('Get data')
            
        assert result.data == 'Data retrieved: test_data'
        assert len(messages) == 4  # Initial request, first response, tool return, final summary
        
        # Verify the last message is a text summary from the model
        assert isinstance(messages[-1], ModelResponse)
        assert len(messages[-1].parts) == 1
        assert isinstance(messages[-1].parts[0], TextPart)
        assert messages[-1].parts[0].content == 'Data retrieved: test_data'

    def test_without_reflection(self) -> None:
        """Test behavior with reflection disabled."""
        def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                # First call - return a tool call
                return ModelResponse(parts=[ToolCallPart('get_data', {'param': 'test'})])
            else:
                # This shouldn't be reached with reflection disabled
                return ModelResponse(parts=[TextPart('This should not be called')])
                
        agent = Agent(FunctionModel(model_func), reflect_on_tool_call=False)
        
        @agent.tool_plain
        def get_data(param: str) -> str:
            return f'test_data: {param}'
            
        with capture_run_messages() as messages:
            result = agent.run_sync('Get data')
            
        # The result should be the text "End tool call" (based on the implementation)
        assert result.data == 'End tool call'
        
        # There should be 4 messages: initial request, tool call, tool return
        # But no final reflection message from the model
        assert len(messages) == 4
        
        # Check message sequence
        assert isinstance(messages[0], ModelRequest)
        assert isinstance(messages[1], ModelResponse)
        assert isinstance(messages[2], ModelRequest)    
        assert isinstance(messages[3], ModelResponse)
        
        # Verify the last message is a tool return, not a model response
        assert len(messages[2].parts) == 1
        assert isinstance(messages[2].parts[0], ToolReturnPart)
        assert messages[2].parts[0].content == 'test_data: test'

    def test_multiple_tools_with_reflection(self) -> None:
        """Test behavior with multiple tool calls and reflection enabled."""
        def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                # First call - return multiple tool calls
                return ModelResponse(parts=[
                    ToolCallPart('tool_a', {'value': 'a'}),
                    ToolCallPart('tool_b', {'value': 'b'})
                ])
            elif len(messages) == 3:
                # After tool returns - return a summary
                return ModelResponse(parts=[TextPart('Tools executed: A and B')])
            else:
                return ModelResponse(parts=[TextPart('Unexpected message sequence')])
                
        agent = Agent(FunctionModel(model_func), reflect_on_tool_call=True)
        
        @agent.tool_plain
        def tool_a(value: str) -> str:
            return f'Result A: {value}'
            
        @agent.tool_plain
        def tool_b(value: str) -> str:
            return f'Result B: {value}'
            
        with capture_run_messages() as messages:
            result = agent.run_sync('Run tools')
            
        assert result.data == 'Tools executed: A and B'
        assert len(messages) == 4  # Initial request, tool calls, tool returns, final summary

    def test_multiple_tools_without_reflection(self) -> None:
        """Test behavior with multiple tool calls and reflection disabled."""
        def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                # First call - return multiple tool calls
                return ModelResponse(parts=[
                    ToolCallPart('tool_a', {'value': 'a'}),
                    ToolCallPart('tool_b', {'value': 'b'})
                ])
            else:
                # This shouldn't be reached with reflection disabled
                return ModelResponse(parts=[TextPart('This should not be called')])
                
        agent = Agent(FunctionModel(model_func), reflect_on_tool_call=False)
        
        @agent.tool_plain
        def tool_a(value: str) -> str:
            return f'Result A: {value}'
            
        @agent.tool_plain
        def tool_b(value: str) -> str:
            return f'Result B: {value}'
            
        with capture_run_messages() as messages:
            result = agent.run_sync('Run tools')
            
        # The result should be "End tool call" based on the implementation
        assert result.data == 'End tool call'
        
        # There should be 4 messages: initial request, tool calls, tool returns, final summary
        assert len(messages) == 4
        
        # Verify the tool returns
        assert isinstance(messages[2], ModelRequest)
        assert len(messages[2].parts) == 2
        assert isinstance(messages[2].parts[0], ToolReturnPart)
        assert isinstance(messages[2].parts[1], ToolReturnPart)
        assert messages[2].parts[0].content == 'Result A: a'
        assert messages[2].parts[1].content == 'Result B: b'

    def test_streaming_with_reflection(self) -> None:
        """Test streaming behavior with reflection enabled."""
        def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                # First call - return a tool call
                return ModelResponse(parts=[ToolCallPart('get_data', {'param': 'test'})])
            elif len(messages) == 3:
                # After tool return - return a summary
                return ModelResponse(parts=[TextPart('Streaming data retrieved')])
            else:
                return ModelResponse(parts=[TextPart('Unexpected message sequence')])
                
        agent = Agent(FunctionModel(model_func), reflect_on_tool_call=True)
        
        @agent.tool_plain
        def get_data(param: str) -> str:
            return f'streaming_data: {param}'
            
        with capture_run_messages() as messages:
            # Using run method instead of run_stream_sync
            result = agent.run_sync('Stream data')
            
        # Verify the interaction completed
        assert result.data == 'Streaming data retrieved'
        
        # Final message should be a text response
        assert isinstance(messages[-1], ModelResponse)
        assert len(messages[-1].parts) == 1
        assert isinstance(messages[-1].parts[0], TextPart)

    def test_streaming_without_reflection(self) -> None:
        """Test streaming behavior with reflection disabled."""
        def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                # First call - return a tool call
                return ModelResponse(parts=[ToolCallPart('get_data', {'param': 'test'})])
            else:
                # This shouldn't be reached with reflection disabled
                return ModelResponse(parts=[TextPart('This should not be called')])
                
        agent = Agent(FunctionModel(model_func), reflect_on_tool_call=False)
        
        @agent.tool_plain
        def get_data(param: str) -> str:
            return f'streaming_data: {param}'
            
        # Using run method instead of run_stream_sync
        with capture_run_messages() as messages:
            result = agent.run_sync('Stream data')
            
        # The result should be "End tool call" based on the implementation
        assert result.data == 'End tool call'
        
        # There should be 4 messages total
        assert len(messages) == 4
        
        # The last message should be the "End tool call" response
        assert isinstance(messages[-1], ModelResponse)
        assert len(messages[-1].parts) == 1
        assert isinstance(messages[-1].parts[0], TextPart)
        assert messages[-1].parts[0].content == 'End tool call'
        
        # The second-to-last message should be the tool return
        assert isinstance(messages[-2], ModelRequest)
        assert len(messages[-2].parts) == 1
        assert isinstance(messages[-2].parts[0], ToolReturnPart)
        assert messages[-2].parts[0].content == 'streaming_data: test'

    def test_result_schema_with_reflection(self) -> None:
        """Test structured result with reflection enabled."""
        class ResultData(BaseModel):
            value: str
            
        def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                # First call - return a tool call
                return ModelResponse(parts=[ToolCallPart('get_data', {'param': 'test'})])
            elif len(messages) == 3:
                # After tool return - return a structured result
                assert info.result_tools
                return ModelResponse(parts=[
                    ToolCallPart(info.result_tools[0].name, {'value': 'structured_value'})
                ])
            else:
                return ModelResponse(parts=[TextPart('Unexpected message sequence')])
                
        agent = Agent(
            FunctionModel(model_func), 
            result_type=ResultData,
            reflect_on_tool_call=True
        )
        
        @agent.tool_plain
        def get_data(param: str) -> str:
            return f'data: {param}'
            
        with capture_run_messages() as messages:
            result = agent.run_sync('Get structured data')
            
        assert isinstance(result.data, ResultData)
        assert result.data.value == 'structured_value'
        
        # There should be 5 messages: 
        # 1. initial request 
        # 2. tool call
        # 3. tool return
        # 4. final result call
        # 5. final result return
        assert len(messages) == 5
        
        # Verify message sequence
        assert isinstance(messages[0], ModelRequest)  # Initial request
        assert isinstance(messages[1], ModelResponse)  # Tool call
        assert isinstance(messages[2], ModelRequest)  # Tool return
        assert isinstance(messages[3], ModelResponse)  # Result call
        assert isinstance(messages[4], ModelRequest)  # Result return
        
        # Verify the result tool call
        assert isinstance(messages[3].parts[0], ToolCallPart)
        assert messages[3].parts[0].tool_name.startswith('final_result')

    def test_result_schema_without_reflection(self) -> None:
        """Test structured result with reflection disabled."""
        class ResultData(BaseModel):
            value: str
            
        def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                # First call - return both a tool call and a structured result
                assert info.result_tools
                return ModelResponse(parts=[
                    ToolCallPart('get_data', {'param': 'test'}),
                    ToolCallPart(info.result_tools[0].name, {'value': 'no_reflection_value'})
                ])
            else:
                return ModelResponse(parts=[TextPart('This should not be called')])
                
        agent = Agent(
            FunctionModel(model_func), 
            result_type=ResultData,
            reflect_on_tool_call=False
        )
        
        @agent.tool_plain
        def get_data(param: str) -> str:
            return f'data: {param}'
            
        with capture_run_messages() as messages:
            result = agent.run_sync('Get structured data without reflection')
            
        assert isinstance(result.data, ResultData)
        assert result.data.value == 'no_reflection_value'
        
        # Expected messages:
        # 1. Initial request
        # 2. Tool call + result call response
        # 3. Tool return + result return request
        assert len(messages) == 3
        
        # Verify message sequence
        assert isinstance(messages[0], ModelRequest)  # Initial request
        assert isinstance(messages[1], ModelResponse)  # Tool call + result call
        assert isinstance(messages[2], ModelRequest)  # Tool return + result return
        
        # Verify there are two parts in the tool response
        assert len(messages[1].parts) == 2
        assert isinstance(messages[1].parts[0], ToolCallPart)
        assert isinstance(messages[1].parts[1], ToolCallPart)
        assert messages[1].parts[0].tool_name == 'get_data'
        assert messages[1].parts[1].tool_name.startswith('final_result')
        
        # Verify there are two parts in the tool return
        assert len(messages[2].parts) == 2
        assert isinstance(messages[2].parts[0], ToolReturnPart)
        assert isinstance(messages[2].parts[1], ToolReturnPart)
        assert messages[2].parts[0].tool_name == 'get_data'
        assert messages[2].parts[1].tool_name.startswith('final_result') 