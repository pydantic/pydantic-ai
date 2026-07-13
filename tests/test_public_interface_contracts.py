"""Enforcement meta-tests for standards the maintainers currently uphold by review alone.

Each test here pins a convention that reviewers repeatedly ask contributors to follow, so that a
violation shows up as a failing test (and, where relevant, an explicit snapshot/allowlist diff)
instead of relying on a human to catch it in review.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
from collections.abc import Callable, Iterator

import pytest
from inline_snapshot import snapshot

import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.agent.abstract import AbstractAgent


def test_public_all_is_sorted_and_pinned():
    """`pydantic_ai.__all__` is pinned so widening the top-level surface is an explicit snapshot diff.

    DouweM has left many "this should not be public" review comments across PRs; freezing the
    exported names turns any addition into a deliberate, reviewable change rather than a silent one.
    """
    assert sorted(pydantic_ai.__all__) == snapshot(
        [
            'AbstractConcurrencyLimiter',
            'AbstractToolset',
            'Agent',
            'AgentCapability',
            'AgentModelSettings',
            'AgentNativeTool',
            'AgentRetries',
            'AgentRun',
            'AgentRunError',
            'AgentRunResult',
            'AgentRunResultEvent',
            'AgentSpec',
            'AgentStreamEvent',
            'AgentToolset',
            'AnyConcurrencyLimit',
            'ApprovalRequired',
            'ApprovalRequiredToolset',
            'AudioFormat',
            'AudioMediaType',
            'AudioUrl',
            'BaseToolCallPart',
            'BaseToolReturnPart',
            'BinaryContent',
            'BinaryImage',
            'CachePoint',
            'CallDeferred',
            'CallToolsNode',
            'CapabilityFunc',
            'CodeExecutionTool',
            'CombinedToolset',
            'CompactionPart',
            'ConcurrencyLimit',
            'ConcurrencyLimitExceeded',
            'ConcurrencyLimitedModel',
            'ConcurrencyLimiter',
            'DEFAULT_PROFILE',
            'DeferredLoadingToolset',
            'DeferredToolRequests',
            'DeferredToolResults',
            'DocumentFormat',
            'DocumentMediaType',
            'DocumentUrl',
            'Embedder',
            'EmbeddingModel',
            'EmbeddingResult',
            'EmbeddingSettings',
            'EndStrategy',
            'ExternalToolset',
            'FallbackExceptionGroup',
            'FilePart',
            'FileSearchTool',
            'FileUrl',
            'FilteredToolset',
            'FinalResultEvent',
            'FinishReason',
            'FunctionToolCallEvent',
            'FunctionToolResultEvent',
            'FunctionToolset',
            'HandleResponseEvent',
            'ImageFormat',
            'ImageGenerationTool',
            'ImageMediaType',
            'ImageUrl',
            'IncludeReturnSchemasToolset',
            'IncompleteToolCall',
            'InlineDefsJsonSchemaTransformer',
            'InstructionPart',
            'InstrumentationSettings',
            'JsonSchemaTransformer',
            'MCPServerTool',
            'MemoryTool',
            'ModelAPIError',
            'ModelHTTPError',
            'ModelMessage',
            'ModelMessagesTypeAdapter',
            'ModelProfile',
            'ModelProfileSpec',
            'ModelRequest',
            'ModelRequestContext',
            'ModelRequestNode',
            'ModelRequestPart',
            'ModelRequestState',
            'ModelResponse',
            'ModelResponsePart',
            'ModelResponsePartDelta',
            'ModelResponseState',
            'ModelResponseStreamEvent',
            'ModelRetry',
            'ModelSettings',
            'MultiModalContent',
            'NativeOutput',
            'NativeToolCallPart',
            'NativeToolReturnPart',
            'OutputToolCallEvent',
            'OutputToolResultEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'PartStartEvent',
            'PrefixedToolset',
            'PreparedToolset',
            'PromptedOutput',
            'RenamedToolset',
            'RequestUsage',
            'RetryPromptPart',
            'RunContext',
            'RunUsage',
            'SetMetadataToolset',
            'SkipModelRequest',
            'SkipToolExecution',
            'SkipToolValidation',
            'StructuredDict',
            'SystemPromptPart',
            'TemplateStr',
            'TextContent',
            'TextOutput',
            'TextPart',
            'TextPartDelta',
            'ThinkingPart',
            'ThinkingPartDelta',
            'Tool',
            'ToolApproved',
            'ToolCallEvent',
            'ToolCallPart',
            'ToolCallPartDelta',
            'ToolChoice',
            'ToolDefinition',
            'ToolDenied',
            'ToolOrOutput',
            'ToolOutput',
            'ToolResultEvent',
            'ToolReturn',
            'ToolReturnPart',
            'ToolsetFunc',
            'ToolsetTool',
            'UndrainedPendingMessagesError',
            'UnexpectedModelBehavior',
            'UploadedFile',
            'UsageLimitExceeded',
            'UsageLimits',
            'UserContent',
            'UserError',
            'UserPromptNode',
            'UserPromptPart',
            'VideoFormat',
            'VideoMediaType',
            'VideoUrl',
            'WebFetchTool',
            'WebSearchTool',
            'WebSearchUserLocation',
            'WrapperToolset',
            'XSearchTool',
            '__version__',
            'capture_run_messages',
            'format_as_xml',
            'limit_model_concurrency',
        ]
    )


def test_public_all_entries_are_importable():
    """Every name in `pydantic_ai.__all__` resolves as an attribute of the package.

    A dangling `__all__` entry (a name that was renamed or removed but left behind) otherwise ships
    green and only surfaces when a user hits the `ImportError`.
    """
    missing = sorted(set(pydantic_ai.__all__).difference(dir(pydantic_ai)))
    assert missing == []


# Frozen snapshot of public dataclasses that predate the keyword-only convention and take two or
# more positional (non-`kw_only`) `__init__` fields. It grandfathers today's offenders so the test
# below only gates NEW public dataclasses: a new one must use `_: KW_ONLY` (or `kw_only=True`) so
# adding fields later can't break positional callers. Do NOT add existing dataclasses here to make
# them positional, and do NOT add a new entry without maintainer sign-off — a new entry means a new
# public dataclass is shipping with a fragile positional signature on purpose.
_KW_ONLY_ALLOWLIST: frozenset[str] = frozenset(
    {
        'pydantic_ai.agent.Agent',
        'pydantic_ai.capabilities.abstract.CapabilityOrdering',
        'pydantic_ai.capabilities.image_generation.ImageGeneration',
        'pydantic_ai.capabilities.mcp.MCP',
        'pydantic_ai.capabilities.native_or_local.NativeOrLocalTool',
        'pydantic_ai.capabilities.prefix_tools.PrefixTools',
        'pydantic_ai.capabilities.web_fetch.WebFetch',
        'pydantic_ai.capabilities.web_search.WebSearch',
        'pydantic_ai.capabilities.x_search.XSearch',
        'pydantic_ai.common_tools.exa.ExaFindSimilarTool',
        'pydantic_ai.common_tools.exa.ExaSearchTool',
        'pydantic_ai.concurrency.ConcurrencyLimit',
        'pydantic_ai.embeddings.bedrock.BedrockEmbeddingModel',
        'pydantic_ai.embeddings.cohere.CohereEmbeddingModel',
        'pydantic_ai.embeddings.google.GoogleEmbeddingModel',
        'pydantic_ai.embeddings.instrumented.InstrumentedEmbeddingModel',
        'pydantic_ai.embeddings.openai.OpenAIEmbeddingModel',
        'pydantic_ai.embeddings.sentence_transformers.SentenceTransformerEmbeddingModel',
        'pydantic_ai.embeddings.test.TestEmbeddingModel',
        'pydantic_ai.embeddings.voyageai.VoyageAIEmbeddingModel',
        'pydantic_ai.function_signature.GenericTypeExpr',
        'pydantic_ai.function_signature.LiteralTypeExpr',
        'pydantic_ai.function_signature.SimpleTypeExpr',
        'pydantic_ai.function_signature.UnionTypeExpr',
        'pydantic_ai.mcp.MCPToolset',
        'pydantic_ai.messages.BaseToolCallPart',
        'pydantic_ai.messages.BaseToolReturnPart',
        'pydantic_ai.messages.CachePoint',
        'pydantic_ai.messages.FunctionToolCallEvent',
        'pydantic_ai.messages.NativeToolCallPart',
        'pydantic_ai.messages.NativeToolReturnPart',
        'pydantic_ai.messages.OutputToolCallEvent',
        'pydantic_ai.messages.OutputToolResultEvent',
        'pydantic_ai.messages.ToolCallPart',
        'pydantic_ai.messages.ToolReturnPart',
        'pydantic_ai.messages.UploadedFile',
        'pydantic_ai.models.anthropic.AnthropicModel',
        'pydantic_ai.models.anthropic.AnthropicStreamedResponse',
        'pydantic_ai.models.bedrock.BedrockConverseModel',
        'pydantic_ai.models.bedrock.BedrockStreamedResponse',
        'pydantic_ai.models.cerebras.CerebrasModel',
        'pydantic_ai.models.cohere.CohereModel',
        'pydantic_ai.models.concurrency.ConcurrencyLimitedModel',
        'pydantic_ai.models.fallback.FallbackModel',
        'pydantic_ai.models.function.DeltaToolCall',
        'pydantic_ai.models.function.FunctionModel',
        'pydantic_ai.models.function.FunctionStreamedResponse',
        'pydantic_ai.models.google.GeminiStreamedResponse',
        'pydantic_ai.models.google.GoogleModel',
        'pydantic_ai.models.groq.GroqModel',
        'pydantic_ai.models.groq.GroqStreamedResponse',
        'pydantic_ai.models.huggingface.HuggingFaceModel',
        'pydantic_ai.models.huggingface.HuggingFaceStreamedResponse',
        'pydantic_ai.models.instrumented.InstrumentationSettings',
        'pydantic_ai.models.instrumented.InstrumentedModel',
        'pydantic_ai.models.mistral.MistralModel',
        'pydantic_ai.models.mistral.MistralStreamedResponse',
        'pydantic_ai.models.ollama.OllamaModel',
        'pydantic_ai.models.openai.OpenAIChatModel',
        'pydantic_ai.models.openai.OpenAIResponsesModel',
        'pydantic_ai.models.openai.OpenAIResponsesStreamedResponse',
        'pydantic_ai.models.openai.OpenAIStreamedResponse',
        'pydantic_ai.models.openrouter.OpenRouterModel',
        'pydantic_ai.models.openrouter.OpenRouterStreamedResponse',
        'pydantic_ai.models.test.TestModel',
        'pydantic_ai.models.test.TestStreamedResponse',
        'pydantic_ai.models.xai.XaiStreamedResponse',
        'pydantic_ai.models.zai.ZaiModel',
        'pydantic_ai.output.NativeOutput',
        'pydantic_ai.output.OutputContext',
        'pydantic_ai.output.OutputObjectDefinition',
        'pydantic_ai.output.PromptedOutput',
        'pydantic_ai.output.ToolOutput',
        'pydantic_ai.result.FinalResult',
        'pydantic_ai.result.StreamedRunResult',
        'pydantic_ai.run.AgentRunResult',
        'pydantic_ai.tool_manager.ToolManager',
        'pydantic_ai.tool_manager.ValidatedToolCall',
        'pydantic_ai.tools.Tool',
        'pydantic_ai.toolsets.approval_required.ApprovalRequiredToolset',
        'pydantic_ai.toolsets.deferred_loading.DeferredLoadingToolset',
        'pydantic_ai.toolsets.filtered.FilteredToolset',
        'pydantic_ai.toolsets.prefixed.PrefixedToolset',
        'pydantic_ai.toolsets.prepared.PreparedToolset',
        'pydantic_ai.toolsets.renamed.RenamedToolset',
        'pydantic_ai.toolsets.set_metadata.SetMetadataToolset',
    }
)


def _is_public_dotted_path(dotted: str) -> bool:
    return not any(part.startswith('_') for part in dotted.split('.'))


def _iter_pydantic_ai_dataclasses() -> Iterator[tuple[str, type]]:
    for module_info in pkgutil.walk_packages(pydantic_ai.__path__, f'{pydantic_ai.__name__}.'):
        try:
            module = importlib.import_module(module_info.name)
        except ImportError:  # pragma: lax no cover
            continue
        for obj in vars(module).values():
            if isinstance(obj, type) and dataclasses.is_dataclass(obj) and obj.__module__ == module_info.name:
                yield f'{obj.__module__}.{obj.__qualname__}', obj


def test_new_public_dataclasses_are_keyword_only():
    """New public dataclasses must not add a second positional `__init__` field.

    "Pretty much all plain dataclasses need `_: KW_ONLY`" is DouweM's most-repeated unenforced
    review nit. Existing offenders are grandfathered in `_KW_ONLY_ALLOWLIST` (changing them to
    keyword-only would break positional callers); this test only fails when a NEW public dataclass
    ships with two or more positional fields, which is where the "add a field, break callers" trap
    lives. Make the new dataclass keyword-only, or add it to the allowlist with maintainer sign-off.
    """
    offenders: set[str] = set()
    for name, cls in _iter_pydantic_ai_dataclasses():
        if not _is_public_dotted_path(name):
            continue
        positional_init_fields = [field for field in dataclasses.fields(cls) if field.init and not field.kw_only]
        if len(positional_init_fields) >= 2:
            offenders.add(name)

    unexpected_offenders = offenders - _KW_ONLY_ALLOWLIST
    assert unexpected_offenders == set()


_RUN_FAMILY_METHODS = ('run', 'run_sync', 'run_stream', 'run_stream_events', 'iter', 'override')


def _load_wrapper(module_path: str, class_name: str) -> type[AbstractAgent[object, object]] | None:
    try:
        module = importlib.import_module(module_path)
    except ImportError:  # pragma: lax no cover
        return None
    return getattr(module, class_name)


_DURABLE_WRAPPERS = {
    'TemporalAgent': _load_wrapper('pydantic_ai.durable_exec.temporal', 'TemporalAgent'),
    'DBOSAgent': _load_wrapper('pydantic_ai.durable_exec.dbos', 'DBOSAgent'),
    'PrefectAgent': _load_wrapper('pydantic_ai.durable_exec.prefect', 'PrefectAgent'),
}


def _parameter_kinds(method: Callable[..., object]) -> dict[str, str]:
    return {name: parameter.kind.name for name, parameter in inspect.signature(method).parameters.items()}


@pytest.mark.parametrize('method_name', _RUN_FAMILY_METHODS)
@pytest.mark.parametrize(
    'wrapper',
    [
        pytest.param(cls, id=name, marks=pytest.mark.skipif(cls is None, reason=f'{name} dependency not installed'))
        for name, cls in _DURABLE_WRAPPERS.items()
    ],
)
def test_durable_wrapper_run_family_signature_parity(wrapper: type[AbstractAgent[object, object]], method_name: str):
    """Durable-execution agent wrappers hand-mirror `Agent`'s run-family signatures.

    `TemporalAgent`, `DBOSAgent`, and `PrefectAgent` re-declare each of `run`, `run_sync`,
    `run_stream`, `run_stream_events`, `iter`, and `override` so they can wrap the run in a
    workflow/activity. Nothing forces a new keyword added to `Agent` to be copied into each wrapper,
    so a wrapper silently drops support for it. This asserts every wrapper method accepts (at least)
    the same parameters, by name and kind, as the corresponding `Agent` method.
    """
    base_parameter_kinds = _parameter_kinds(getattr(Agent, method_name))
    wrapper_parameter_kinds = _parameter_kinds(getattr(wrapper, method_name))
    assert base_parameter_kinds.items() <= wrapper_parameter_kinds.items()
