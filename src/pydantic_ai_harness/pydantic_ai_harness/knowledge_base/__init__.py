"""Knowledge base capability: retrieval from a document corpus the application provides, over a pluggable backend.

`InMemoryKnowledgeBase` needs nothing extra. `BedrockKnowledgeBase` needs `boto3` and `OpenAIVectorStore`
needs `openai`; they load on first access, so importing this package pulls in neither.
"""

from typing import TYPE_CHECKING

from pydantic_ai_harness.knowledge_base._backend import (
    Document,
    FilterScalar,
    KnowledgeBaseBackend,
    KnowledgeBaseResult,
    MetadataFilter,
    NativeFileSearchBackend,
    Passage,
    ReadableKnowledgeBase,
    matches,
)
from pydantic_ai_harness.knowledge_base._capability import KnowledgeBase
from pydantic_ai_harness.knowledge_base._in_memory import InMemoryKnowledgeBase
from pydantic_ai_harness.knowledge_base._toolset import FilterFunc, KnowledgeBaseToolset

if TYPE_CHECKING:
    from pydantic_ai_harness.knowledge_base._bedrock import BedrockAgentRuntimeClient, BedrockKnowledgeBase
    from pydantic_ai_harness.knowledge_base._openai import OpenAIVectorStore

__all__ = [
    'BedrockAgentRuntimeClient',
    'BedrockKnowledgeBase',
    'Document',
    'FilterFunc',
    'FilterScalar',
    'InMemoryKnowledgeBase',
    'KnowledgeBase',
    'KnowledgeBaseBackend',
    'KnowledgeBaseResult',
    'KnowledgeBaseToolset',
    'MetadataFilter',
    'NativeFileSearchBackend',
    'OpenAIVectorStore',
    'Passage',
    'ReadableKnowledgeBase',
    'matches',
]

_BACKENDS = {
    'BedrockAgentRuntimeClient': '_bedrock',
    'BedrockKnowledgeBase': '_bedrock',
    'OpenAIVectorStore': '_openai',
}


def __getattr__(name: str) -> object:
    module_name = _BACKENDS.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    from importlib import import_module

    return getattr(import_module(f'{__name__}.{module_name}'), name)
