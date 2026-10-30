"""Knowledge base capability: retrieval from a document corpus the application provides, over a pluggable backend.

`InMemoryKnowledgeBase` needs nothing extra. `BedrockKnowledgeBase` needs `boto3` and `OpenAIVectorStore`
needs `openai`; they load on first access, so importing this package pulls in neither.
"""

from typing import TYPE_CHECKING

from pydantic_ai_harness.knowledge_base._backend import (
    ComparisonOperator,
    Document,
    FilterCondition,
    FilterNode,
    FilterScalar,
    FilterValue,
    KnowledgeBaseBackend,
    KnowledgeBaseResult,
    LogicalFilter,
    LogicalOperator,
    MetadataFilter,
    NativeFileSearchBackend,
    Passage,
    ReadableKnowledgeBase,
    SearchMode,
    allows_read,
    is_readable,
    matches,
    normalize_filter,
)
from pydantic_ai_harness.knowledge_base._capability import KnowledgeBase
from pydantic_ai_harness.knowledge_base._in_memory import InMemoryKnowledgeBase
from pydantic_ai_harness.knowledge_base._toolset import FilterFunc, KnowledgeBaseToolset

if TYPE_CHECKING:
    from pydantic_ai_harness.knowledge_base._bedrock import (
        BedrockAgentRuntimeClient,
        BedrockKnowledgeBase,
        S3Client,
        SourceConverter,
    )
    from pydantic_ai_harness.knowledge_base._openai import OpenAIVectorStore
    from pydantic_ai_harness.knowledge_base._pgvector import PgVectorKnowledgeBase
    from pydantic_ai_harness.knowledge_base._qdrant import QdrantKnowledgeBase

__all__ = [
    'BedrockAgentRuntimeClient',
    'BedrockKnowledgeBase',
    'ComparisonOperator',
    'Document',
    'FilterCondition',
    'FilterFunc',
    'FilterNode',
    'FilterScalar',
    'FilterValue',
    'InMemoryKnowledgeBase',
    'KnowledgeBase',
    'KnowledgeBaseBackend',
    'KnowledgeBaseResult',
    'KnowledgeBaseToolset',
    'LogicalFilter',
    'LogicalOperator',
    'MetadataFilter',
    'NativeFileSearchBackend',
    'OpenAIVectorStore',
    'S3Client',
    'SourceConverter',
    'Passage',
    'PgVectorKnowledgeBase',
    'QdrantKnowledgeBase',
    'ReadableKnowledgeBase',
    'SearchMode',
    'allows_read',
    'is_readable',
    'matches',
    'normalize_filter',
]

_BACKENDS = {
    'BedrockAgentRuntimeClient': '_bedrock',
    'BedrockKnowledgeBase': '_bedrock',
    'S3Client': '_bedrock',
    'SourceConverter': '_bedrock',
    'OpenAIVectorStore': '_openai',
    'QdrantKnowledgeBase': '_qdrant',
    'PgVectorKnowledgeBase': '_pgvector',
}


def __getattr__(name: str) -> object:
    module_name = _BACKENDS.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    from importlib import import_module

    return getattr(import_module(f'{__name__}.{module_name}'), name)
