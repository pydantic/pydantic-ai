"""Agent configuration for the Temporal streaming example.

This module defines the agent setup with MCP toolsets, model configuration,
and custom tools for data analysis.
"""

from datetime import timedelta

from mcp_run_python import code_sandbox
from pydantic_ai import Agent, FilteredToolset, ModelSettings, RunContext
from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityConfig

from datamodels import AgentDependencies


async def get_mcp_toolsets() -> dict[str, FilteredToolset]:
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


async def get_claude_model(parallel_tool_calls: bool = True, **env_vars):
    """
    Create and configure the Claude model.

    Args:
        parallel_tool_calls: Whether to enable parallel tool calls.
        **env_vars: Environment variables including API keys.

    Returns:
        Configured AnthropicModel instance.
    """
    model_name = 'claude-sonnet-4-5-20250929'
    api_key = env_vars.get('anthropic_api_key')
    model = AnthropicModel(
        model_name=model_name,
        provider=AnthropicProvider(api_key=api_key),
        settings=ModelSettings(
            **{
                'temperature': 0.5,
                'n': 1,
                'max_completion_tokens': 64000,
                'max_tokens': 64000,
                'parallel_tool_calls': parallel_tool_calls,
            }
        ),
    )

    return model


async def build_agent(stream_handler=None, **env_vars):
    """
    Build and configure the agent with tools and temporal settings.

    Args:
        stream_handler: Optional event stream handler for streaming responses.
        **env_vars: Environment variables including API keys.

    Returns:
        TemporalAgent instance ready for use in Temporal workflows.
    """
    system_prompt = """
    You are an expert financial analyst that knows how to search for financial data on the web.
    You also have a Data Analyst background, mastering well how to use pandas for tabular operations.
    """
    agent_name = 'YahooFinanceSearchAgent'

    toolsets = await get_mcp_toolsets()
    agent = Agent(
        name=agent_name,
        model=await get_claude_model(**env_vars),
        toolsets=[*toolsets.values()],
        system_prompt=system_prompt,
        event_stream_handler=stream_handler,
        deps_type=AgentDependencies,
    )

    @agent.tool(name='run_python_code')
    async def run_python_code(ctx: RunContext[None], code: str) -> str:
        """Execute Python code in a sandboxed environment with pandas and numpy available."""
        async with code_sandbox(dependencies=['pandas', 'numpy']) as sandbox:
            result = await sandbox.eval(code)
            return result

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
