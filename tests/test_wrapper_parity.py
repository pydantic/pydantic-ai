"""Each model wrapper forwards every member of the class it wraps, and each wrapper subclass handles every entry point.

A wrapper (`WrapperModel`, `WrapperEmbeddingModel`, `WrapperImageGenerationModel`) holds a model of its base class.
A member the wrapper doesn't define resolves to the base class's own implementation rather than the wrapped model's,
since `__getattr__` forwards only names the class lacks. So:

- each wrapper defines every public method and property of its base;
- each subclass of a wrapper overrides every entry point (`ENTRY_POINTS` in `tests/model_adapter_cases.py`). A subclass
  adds behavior around the wrapped model's calls, like a concurrency limit, a durable step or a span, and an entry
  point it leaves to the wrapper's plain forwarding goes without that behavior.

A member left unforwarded is listed in `DELEGATES_BY_DESIGN` with why that's right, or in `KNOWN_GAPS` while it isn't;
a gap that's a defect is pinned by a strict `xfail` test below until it's fixed. Not VCR tests: they check the classes
themselves.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import warnings
from collections.abc import AsyncGenerator, Iterator, Sequence
from contextlib import asynccontextmanager

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai.embeddings import EmbeddingModel, InstrumentedEmbeddingModel, TestEmbeddingModel, WrapperEmbeddingModel
from pydantic_ai.images import ImageGenerationModel
from pydantic_ai.images.wrapper import WrapperImageGenerationModel
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.batch import Batch, BatchRequest, BatchResult, EncodedBatchRequest
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from .capability_cases import caused_by_missing_dependency
from .manifests import Exempt, assert_classified
from .model_adapter_cases import ADAPTER_PACKAGES, ENTRY_POINTS, invoke

WRAPPERS: dict[
    type[Model | EmbeddingModel | ImageGenerationModel], type[Model | EmbeddingModel | ImageGenerationModel]
] = {
    WrapperModel: Model,
    WrapperEmbeddingModel: EmbeddingModel,
    WrapperImageGenerationModel: ImageGenerationModel,
}
"""Each wrapper, and the base class of the models it wraps."""

WRAPPER_PACKAGES = (*ADAPTER_PACKAGES, 'pydantic_ai.durable_exec')
"""Where wrapper subclasses ship: the model packages, and the durable execution engines' model wrappers."""

_BATCH = ('batch_encode', 'batch_create', 'batch_status', 'batch_results', 'batch_cancel')
_DURABLE_BATCHES = Exempt(
    'a batched request is made durable by the batch capability, which runs each submit and check as its own durable '
    'unit (see `docs/batch.md`), not by this wrapper'
)
_DEPRECATED_DURABLE = Exempt(
    'deprecated; keeps the token count and compaction in workflow code, as '
    'https://github.com/pydantic/pydantic-ai/pull/13460 decided'
)
_BATCHED = Exempt('batches only `request`; the wrapped model answers its other entry points itself')

DELEGATES_BY_DESIGN: dict[tuple[str, str], Exempt] = {
    ('WrapperModel', 'label'): Exempt('derived from `model_name` and `system`, which it forwards'),
    ('WrapperModel', 'resolve_prompt_cache_retention'): Exempt(
        'deprecated; calls `resolve_cache_retention`, which it forwards'
    ),
    ('WrapperModel', 'supported_native_tools'): Exempt(
        'a classmethod the wrapped model reads while building its own `profile`, which the wrapper forwards'
    ),
    ('WrapperImageGenerationModel', 'prepare_generate'): Exempt('reads `settings`, which it forwards'),
    **{('BatchedModel', name): _BATCHED for name in (*_BATCH, 'count_tokens', 'compact_messages')},
    **{('DurableModel', name): _DURABLE_BATCHES for name in _BATCH},
    **{
        (engine, name): _DURABLE_BATCHES for engine in ('DBOSModel', 'PrefectModel', 'TemporalModel') for name in _BATCH
    },
    **{
        (engine, name): _DEPRECATED_DURABLE
        for engine in ('DBOSModel', 'PrefectModel', 'TemporalModel')
        for name in ('count_tokens', 'compact_messages')
    },
    ('InstrumentedEmbeddingModel', 'max_input_tokens'): Exempt('answers from a table, with no call to trace'),
}

_TOOL_MODES = Exempt(
    "reports `None`, since it intersects the profile with the wrapper's own empty `supported_tool_*_modes`; "
    'pinned by `test_wrapper_reports_the_wrapped_model_tool_modes`'
)
_UNTRACED = Exempt(
    'sends a request the wrapper records no span for; pinned by `test_instrumented_model_traces_count_tokens`'
)
_UNLIMITED = Exempt(
    'takes no concurrency slot, unlike `request` and `count_tokens`; open question for the maintainers: should a '
    'batch API call or `cancel_suspended_response` take one?'
)

KNOWN_GAPS: dict[tuple[str, str], Exempt] = {
    ('WrapperModel', 'tool_deferral_mode'): _TOOL_MODES,
    ('WrapperModel', 'tool_addition_mode'): _TOOL_MODES,
    ('WrapperEmbeddingModel', 'prepare_embed'): Exempt(
        "merges the wrapper's own `_settings`, not the wrapped model's `settings`; pinned by "
        '`test_instrumented_embedding_span_records_the_wrapped_model_settings`'
    ),
    **{
        ('InstrumentedModel', name): _UNTRACED
        for name in (*_BATCH, 'count_tokens', 'compact_messages', 'cancel_suspended_response')
    },
    ('InstrumentedEmbeddingModel', 'count_tokens'): _UNTRACED,
    **{('ConcurrencyLimitedModel', name): _UNLIMITED for name in (*_BATCH, 'cancel_suspended_response')},
}


def _members(base: type) -> set[str]:
    """`base`'s public methods and properties, and its async context manager methods, as its subclasses see them."""
    return {
        name
        for cls in base.__mro__
        if cls is not object
        for name, member in vars(cls).items()
        if (not name.startswith('_') or name in ('__aenter__', '__aexit__'))
        and (isinstance(member, property | classmethod | staticmethod) or inspect.isfunction(member))
    }


def _defining_class(cls: type, name: str) -> type:
    return next(owner for owner in cls.__mro__ if name in vars(owner))


def _subclasses(cls: type) -> Iterator[type]:
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _subclasses(subclass)


def _shipped_wrapper_subclasses() -> tuple[dict[type, list[type]], bool]:
    """Each wrapper's subclasses in `WRAPPER_PACKAGES`, and whether the walk imported every module."""
    complete = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for package_name in WRAPPER_PACKAGES:
            package = importlib.import_module(package_name)
            for module_info in pkgutil.walk_packages(package.__path__, f'{package_name}.'):
                try:
                    importlib.import_module(module_info.name)
                except Exception as exc:
                    if not caused_by_missing_dependency(exc):
                        raise
                    complete = False
    shipped = {
        wrapper: sorted(
            {cls for cls in _subclasses(wrapper) if cls.__module__.startswith('pydantic_ai.')}, key=lambda c: c.__name__
        )
        for wrapper in WRAPPERS
    }
    return shipped, complete


def test_each_wrapper_member_is_forwarded_or_classified():
    """A member a wrapper leaves to its base class, or a subclass to its wrapper's forwarding, needs a reason."""
    subclasses, complete = _shipped_wrapper_subclasses()
    unforwarded: set[tuple[str, str]] = set()
    for wrapper, base in WRAPPERS.items():
        unforwarded |= {
            (wrapper.__name__, name) for name in _members(base) if _defining_class(wrapper, name) is not wrapper
        }
        for subclass in subclasses[wrapper]:
            unforwarded |= {
                (subclass.__name__, name) for name in ENTRY_POINTS[base] if _defining_class(subclass, name) is wrapper
            }
    assert not DELEGATES_BY_DESIGN.keys() & KNOWN_GAPS.keys()
    assert_classified(
        unforwarded,
        {**DELEGATES_BY_DESIGN, **KNOWN_GAPS},
        complete_walk=complete,
        table_name='DELEGATES_BY_DESIGN | KNOWN_GAPS',
    )


class _Reached(Exception):
    """Raised by `_RecordingModel` from the entry point a wrapper forwarded to it."""


class _RecordingModel(Model):
    """A model whose every entry point records that it was reached, and stops there."""

    def __init__(self) -> None:
        super().__init__()
        self.reached: list[str] = []

    @property
    def model_name(self) -> str:
        return 'recording'

    @property
    def system(self) -> str:
        return 'recording'

    def _reach(self, entry_point: str) -> _Reached:
        self.reached.append(entry_point)
        return _Reached(entry_point)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        raise self._reach('request')

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: object = None,
    ) -> AsyncGenerator[StreamedResponse]:
        raise self._reach('request_stream')
        yield  # pragma: no cover

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        raise self._reach('count_tokens')

    async def compact_messages(
        self, request_context: ModelRequestContext, *, instructions: str | None = None
    ) -> ModelResponse:
        raise self._reach('compact_messages')

    async def batch_encode(self, request: BatchRequest) -> EncodedBatchRequest:
        self.reached.append('batch_encode')
        return EncodedBatchRequest(custom_id=request.custom_id, group='recording', body='{}')

    async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
        raise self._reach('batch_create')

    async def batch_status(self, batch: Batch) -> Batch:
        raise self._reach('batch_status')

    async def batch_results(self, batch: Batch) -> list[BatchResult]:
        raise self._reach('batch_results')

    async def batch_cancel(self, batch: Batch) -> Batch:
        raise self._reach('batch_cancel')

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        raise self._reach('cancel_suspended_response')


@pytest.mark.parametrize('entry_point', ENTRY_POINTS[Model])
async def test_wrapper_model_forwards_each_entry_point(entry_point: str):
    """A `WrapperModel` hands each entry point to the model it wraps."""
    wrapped = _RecordingModel()
    try:
        await invoke(WrapperModel(wrapped), entry_point)
    except _Reached:
        pass
    assert wrapped.reached[-1] == entry_point


def test_wrapper_model_forwards_its_provider():
    wrapped = FunctionModel(lambda messages, info: ModelResponse(parts=[]))
    assert WrapperModel(wrapped).provider is wrapped.provider


def _deferring(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise NotImplementedError


@pytest.mark.xfail(strict=True, reason='see `KNOWN_GAPS`: a wrapper reads its own empty `supported_tool_*_modes`')
def test_wrapper_reports_the_wrapped_model_tool_modes():
    wrapped = FunctionModel(
        _deferring, profile=ModelProfile(tool_deferral_mode='standalone', tool_addition_mode='by_reference')
    )
    assert (wrapped.tool_deferral_mode, wrapped.tool_addition_mode) == ('standalone', 'by_reference')
    wrapper = WrapperModel(wrapped)
    assert (wrapper.tool_deferral_mode, wrapper.tool_addition_mode) == ('standalone', 'by_reference')


@pytest.mark.xfail(strict=True, reason="see `KNOWN_GAPS`: `prepare_embed` merges the wrapper's own `_settings`")
async def test_instrumented_embedding_span_records_the_wrapped_model_settings():
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    model = InstrumentedEmbeddingModel(
        TestEmbeddingModel(settings={'dimensions': 3}), InstrumentationSettings(tracer_provider=tracer_provider)
    )
    await model.embed('hello', input_type='query')
    (span,) = exporter.get_finished_spans()
    settings = span.attributes and span.attributes.get('embedding_settings')
    assert isinstance(settings, str)
    assert json.loads(settings) == {'dimensions': 3}


class _CountingModel(_RecordingModel):
    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        return RequestUsage(input_tokens=1)


@pytest.mark.xfail(strict=True, reason='see `KNOWN_GAPS`: `InstrumentedModel` records no span for a token count')
async def test_instrumented_model_traces_count_tokens():
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    model = InstrumentedModel(_CountingModel(), InstrumentationSettings(tracer_provider=tracer_provider))
    await model.count_tokens([ModelRequest.user_text_prompt('hello')], None, ModelRequestParameters())
    assert exporter.get_finished_spans()
