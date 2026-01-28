"""Agent configuration for the Temporal streaming example.

This module defines the agent setup with MCP toolsets, model configuration,
and custom tools for data analysis.
"""
from datetime import timedelta
from typing import Any

from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityConfig

from pydantic_ai import Agent, FilteredToolset, ModelSettings
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from .datamodels import AgentDependencies


async def get_mcp_toolsets() -> dict[str, FilteredToolset[AgentDependencies]]:
    """
    Initialize MCP toolsets for the agent.

    Returns:
        A dictionary mapping toolset names to filtered toolsets.
    """
    yf_server = MCPServerStdio(
        command='uvx',
        args=['mcp-yahoo-finance'],
        timeout=240,
        read_timeout=240,
        id='yahoo',
    )
    return {'yahoo': yf_server.filtered(lambda ctx, tool_def: True)}


async def get_claude_model(parallel_tool_calls: bool = True, **kwargs: Any) -> AnthropicModel:
    """
    Create and configure the Claude model.

    Args:
        parallel_tool_calls: Whether to enable parallel tool calls.
        **kwargs: Environment variables including API keys.

    Returns:
        Configured AnthropicModel instance.
    """
    model_name: str = 'claude-sonnet-4-5-20250929'
    api_key: str | None = kwargs.get('anthropic_api_key', None)
    model: AnthropicModel = AnthropicModel(
        model_name=model_name,
        provider=AnthropicProvider(api_key=api_key),
        settings=ModelSettings(
            temperature=0.5,
            max_tokens=64000,
            parallel_tool_calls=parallel_tool_calls,
        ),
    )

    return model


async def build_agent(stream_handler: EventStreamHandler[AgentDependencies],
                      **kwargs: Any) -> TemporalAgent[AgentDependencies, str]:
    """
    Build and configure the agent with tools and temporal settings.

    Args:
        stream_handler: Optional event stream handler for streaming responses.
        **kwargs: Environment variables including API keys.

    Returns:
        TemporalAgent instance ready for use in Temporal workflows.
    """
    system_prompt = """
    You are an expert financial analyst that knows how to search for financial data on the web.
    """
    agent_name = 'YahooFinanceSearchAgent'

    toolsets = await get_mcp_toolsets()
    agent: Agent[AgentDependencies, str] = Agent[AgentDependencies, str](
        name=agent_name,
        model=await get_claude_model(**kwargs),
        toolsets=[*toolsets.values()],
        system_prompt=system_prompt,
        event_stream_handler=stream_handler,
        deps_type=AgentDependencies,
    )

    temporal_agent = TemporalAgent(
        wrapped=agent,
        model_activity_config=ActivityConfig(
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=50),
        ),
        toolset_activity_config={
            toolset_id: ActivityConfig(
                start_to_close_timeout=timedelta(minutes=3),
                retry_policy=RetryPolicy(
                    maximum_attempts=3, non_retryable_error_types=['ToolRetryError']
                ),
            )
            for toolset_id in toolsets.keys()
        },
    )
    return temporal_agent
