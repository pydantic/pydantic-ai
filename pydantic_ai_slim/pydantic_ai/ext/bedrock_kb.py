# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false
"""Amazon Bedrock Knowledge Base integration for Pydantic AI.

Provides a reusable retrieval tool that queries Amazon Bedrock Managed Knowledge Bases.

Usage:
    from pydantic_ai import Agent
    from pydantic_ai.ext.bedrock_kb import create_bedrock_kb_tool

    agent = Agent('aws:us.anthropic.claude-sonnet-4-5-20250929-v1:0')
    bedrock_kb_tool = create_bedrock_kb_tool(knowledge_base_id="ABCDEFGHIJ")
    agent.tool_plain(bedrock_kb_tool)
"""

import os
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


def get_source_uri(result: dict[str, Any]) -> str:
    """Extract source URI from a retrieval result, handling all location types."""
    location = result.get('location', {})
    loc_type = location.get('type', '')
    if loc_type == 'S3' or 's3Location' in location:
        return location.get('s3Location', {}).get('uri', '')
    elif loc_type == 'WEB' or 'webLocation' in location:
        return location.get('webLocation', {}).get('url', '')
    elif 'confluenceLocation' in location:
        return location.get('confluenceLocation', {}).get('url', '')
    elif 'salesforceLocation' in location:
        return location.get('salesforceLocation', {}).get('url', '')
    elif 'sharePointLocation' in location:
        return location.get('sharePointLocation', {}).get('url', '')
    elif 'customDocumentLocation' in location:
        return location.get('customDocumentLocation', {}).get('id', '')
    # Fallback to metadata._source_uri (for agentic results)
    return result.get('metadata', {}).get('_source_uri', '')


class RetrievalResult(BaseModel):
    """A single result from a Bedrock Knowledge Base query."""

    content: str
    source: str = ''
    score: float = 0.0


@dataclass
class BedrockKBConfig:
    """Configuration for Bedrock Knowledge Base retrieval."""

    knowledge_base_id: str = field(default_factory=lambda: os.environ.get('KNOWLEDGE_BASE_ID', ''))
    region_name: str = field(default_factory=lambda: os.environ.get('AWS_REGION', 'us-east-1'))
    number_of_results: int = 5
    knowledge_base_type: str = 'MANAGED'
    use_agentic_retrieval: bool = field(
        default_factory=lambda: os.environ.get('USE_AGENTIC_RETRIEVAL', 'true').lower() != 'false'
    )


def create_bedrock_kb_tool(
    knowledge_base_id: str | None = None,
    region_name: str | None = None,
    number_of_results: int = 5,
    knowledge_base_type: str = 'MANAGED',
    use_agentic_retrieval: bool | None = None,
    client_override: Any = None,
):
    """Create a Bedrock Knowledge Base retrieval tool for use with a Pydantic AI agent.

    Args:
        knowledge_base_id: The KB ID. Falls back to KNOWLEDGE_BASE_ID env var.
        region_name: AWS region. Falls back to AWS_REGION env var or us-east-1.
        number_of_results: Max results to return.
        knowledge_base_type: "MANAGED" (recommended) or "VECTOR".
        use_agentic_retrieval: Use AgenticRetrieveStream for complex queries with
            query decomposition and managed reranking. Falls back to plain Retrieve
            on failure. Defaults to True.
        client_override: Pre-configured boto3 client (for testing). If None, creates one lazily.

    Returns:
        An async function suitable for use with agent.tool_plain().
    """
    config = BedrockKBConfig(
        knowledge_base_id=knowledge_base_id or os.environ.get('KNOWLEDGE_BASE_ID', ''),
        region_name=region_name or os.environ.get('AWS_REGION', 'us-east-1'),
        number_of_results=number_of_results,
        knowledge_base_type=knowledge_base_type,
        use_agentic_retrieval=use_agentic_retrieval
        if use_agentic_retrieval is not None
        else os.environ.get('USE_AGENTIC_RETRIEVAL', 'true').lower() != 'false',
    )

    _client: Any = client_override

    def _get_client():
        nonlocal _client
        if _client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    'boto3 is required for Bedrock KB integration. Install with: pip install boto3>=1.41.0'
                )
            _client = boto3.client('bedrock-agent-runtime', region_name=config.region_name)
        return _client

    def _managed_retrieve(query: str) -> list[RetrievalResult]:
        """Retrieve using plain managed Retrieve API."""
        client = _get_client()

        if config.knowledge_base_type == 'MANAGED':
            retrieval_config = {'managedSearchConfiguration': {'numberOfResults': config.number_of_results}}
        else:
            retrieval_config = {'vectorSearchConfiguration': {'numberOfResults': config.number_of_results}}

        response = client.retrieve(
            knowledgeBaseId=config.knowledge_base_id,
            retrievalQuery={'text': query},
            retrievalConfiguration=retrieval_config,
        )

        results = []
        for result in response.get('retrievalResults', []):
            content = result.get('content', {}).get('text', '')
            source = get_source_uri(result)
            score = result.get('score', 0.0)
            results.append(RetrievalResult(content=content, source=source, score=score))

        return results

    def _agentic_retrieve(query: str) -> list[RetrievalResult]:
        """Retrieve using AgenticRetrieveStream with fallback to plain Retrieve."""
        try:
            client = _get_client()
            response = client.agentic_retrieve_stream(
                knowledgeBaseId=config.knowledge_base_id,
                messages=[{'content': {'text': query}, 'role': 'user'}],
                retrievers=[
                    {
                        'configuration': {
                            'knowledgeBase': {
                                'knowledgeBaseId': config.knowledge_base_id,
                                'retrievalOverrides': {'maxNumberOfResults': config.number_of_results},
                            }
                        }
                    }
                ],
                agenticRetrieveConfiguration={
                    'foundationModelType': 'MANAGED',
                    'rerankingModelType': 'MANAGED',
                },
            )
            # Process streaming response
            results = []
            for event in response.get('stream', []):
                if 'result' in event and 'results' in event['result']:
                    for result in event['result']['results']:
                        content = result.get('content', {}).get('text', '')
                        source = get_source_uri(result)
                        score = result.get('score', 0.0)
                        results.append(RetrievalResult(content=content, source=source, score=score))
            return results
        except Exception:
            # Fall back to plain managed retrieve
            return _managed_retrieve(query)

    async def bedrock_kb_retrieve(query: str) -> list[RetrievalResult]:
        """Retrieve relevant documents from an Amazon Bedrock Knowledge Base.

        Args:
            query: The search query to find relevant documents.

        Returns:
            A list of retrieval results with content, source, and relevance score.
        """
        if config.use_agentic_retrieval:
            return _agentic_retrieve(query)
        return _managed_retrieve(query)

    return bedrock_kb_retrieve
