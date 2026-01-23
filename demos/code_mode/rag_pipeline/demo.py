"""RAG Pipeline Demo - Shared code for web and evals.

Demonstrates multi-step RAG pipeline: search → conditional fetch → upsert → rerank → answer.
"""

from __future__ import annotations

import os

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.combined import CombinedToolset

from .guard import PineconeGuardToolset

# =============================================================================
# Configuration
# =============================================================================

MODELS = [
    'gateway/anthropic:claude-sonnet-4-5',
    'gateway/openai:gpt-5.2',
    'gateway/gemini:gemini-3-flash-preview',
]

DEFAULT_MODEL = 'gateway/anthropic:claude-sonnet-4-5'
MAX_RETRIES = 5

# Pinecone index configuration
PINECONE_INDEX_NAME = 'pydantic-kb'
PINECONE_NAMESPACE = 'default'

# Threshold for "good" search results
SCORE_THRESHOLD = 0.7

# =============================================================================
# Prompt
# =============================================================================

PROMPT = """
Answer these questions about Pydantic AI using the knowledge base.

For EACH question:
1. Search the "pydantic-kb" index (topK=3)
2. If no good results (top score < 0.7):
   - Use tavily_search to find the answer on the web
   - Upsert the web result to the index for future queries
   - Search again to verify upsert worked
3. Rerank the top results using rerank_documents with model="bge-reranker-v2-m3"
   NOTE: documents arg must be list of strings OR list of {"text": "..."} objects.
   Extract the "content" field from search results as the "text" value.
   IMPORTANT: Always use model="bge-reranker-v2-m3" for better multilingual support.
4. Extract the best answer from the top reranked result

Questions:
1. "How do I create a streaming agent in Pydantic AI?"
2. "What output types does Pydantic AI support?"
3. "How do I add custom tools to an agent?"
4. "What is the difference between run() and run_stream()?"
5. "How do I handle retries in Pydantic AI?"

Return a dict mapping each question to its answer.
""".lstrip()

# Simpler prompt variant for zero-setup mode (no API keys needed)
PROMPT_ZERO_SETUP = """
Search Pinecone documentation for information about these topics and summarize findings:

1. "How to create a Pinecone index"
2. "How to upsert vectors"
3. "How to query vectors"
4. "What embedding models does Pinecone support"
5. "How to use Pinecone with Python"

For each topic, use search_docs to find relevant documentation, then summarize.
Return a dict mapping each topic to its summary.
""".lstrip()


# =============================================================================
# MCP Server Factories
# =============================================================================


def create_pinecone_mcp() -> MCPServerStdio:
    """Create Pinecone MCP server connection.

    Provides: search_records, upsert_records, rerank_documents
    Requires: PINECONE_API_KEY environment variable
    """
    api_key = os.environ.get('PINECONE_API_KEY')
    if not api_key:
        raise ValueError('PINECONE_API_KEY environment variable is required')

    return MCPServerStdio(
        'npx',
        args=['-y', '@pinecone-database/mcp'],
        env={'PINECONE_API_KEY': api_key, **os.environ},
    )


def create_tavily_mcp() -> MCPServerStreamableHTTP:
    """Create Tavily MCP server connection.

    Provides: tavily_search, tavily_extract
    Requires: TAVILY_API_KEY environment variable
    """
    api_key = os.environ.get('TAVILY_API_KEY')
    if not api_key:
        raise ValueError('TAVILY_API_KEY environment variable is required')

    return MCPServerStreamableHTTP(
        url=f'https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}',
    )


def create_pinecone_mcp_zero_setup() -> MCPServerStdio:
    """Create Pinecone MCP server for zero-setup mode.

    Uses search_docs tool which doesn't require API key.
    """
    return MCPServerStdio(
        'npx',
        args=['-y', '@pinecone-database/mcp'],
    )


# =============================================================================
# Agent Factories
# =============================================================================


def create_traditional_agent(
    pinecone: MCPServerStdio,
    tavily: MCPServerStreamableHTTP,
    model: str = DEFAULT_MODEL,
) -> Agent[None, str]:
    """Create agent with traditional tool calling."""
    combined: CombinedToolset[None] = CombinedToolset([pinecone, tavily])
    guarded: PineconeGuardToolset[None] = PineconeGuardToolset(wrapped=combined)
    agent: Agent[None, str] = Agent(
        model,
        toolsets=[guarded],
        system_prompt='You are a knowledge base assistant. Use the available tools to search, upsert, and rerank results.',
    )
    return agent


def create_code_mode_agent(
    pinecone: MCPServerStdio,
    tavily: MCPServerStreamableHTTP,
    model: str = DEFAULT_MODEL,
) -> Agent[None, str]:
    """Create agent with code mode (tools as Python functions)."""
    combined: CombinedToolset[None] = CombinedToolset([pinecone, tavily])
    guarded: PineconeGuardToolset[None] = PineconeGuardToolset(wrapped=combined)
    code_toolset: CodeModeToolset[None] = CodeModeToolset(wrapped=guarded, max_retries=MAX_RETRIES)
    agent: Agent[None, str] = Agent(
        model,
        toolsets=[code_toolset],
        system_prompt='You are a knowledge base assistant. Write Python code to efficiently search, upsert, and rerank results.',
    )
    return agent


def create_traditional_agent_zero_setup(
    pinecone: MCPServerStdio,
    model: str = DEFAULT_MODEL,
) -> Agent[None, str]:
    """Create agent with traditional tool calling (zero-setup mode)."""
    agent: Agent[None, str] = Agent(
        model,
        toolsets=[pinecone],
        system_prompt='You are a documentation assistant. Use search_docs to find information in Pinecone documentation.',
    )
    return agent


def create_code_mode_agent_zero_setup(
    pinecone: MCPServerStdio,
    model: str = DEFAULT_MODEL,
) -> Agent[None, str]:
    """Create agent with code mode (zero-setup mode)."""
    code_toolset: CodeModeToolset[None] = CodeModeToolset(wrapped=pinecone, max_retries=MAX_RETRIES)
    agent: Agent[None, str] = Agent(
        model,
        toolsets=[code_toolset],
        system_prompt='You are a documentation assistant. Write Python code to efficiently search Pinecone documentation.',
    )
    return agent
