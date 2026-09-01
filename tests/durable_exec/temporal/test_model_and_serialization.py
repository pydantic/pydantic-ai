from __future__ import annotations

import re
import sys
import uuid
import warnings
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, Any, Literal
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, TypeAdapter

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    BinaryContent,
    BinaryImage,
    CodeExecutionTool,
    DocumentUrl,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    MultiModalContent,
    RequestUsage,
    RunContext,
    RunUsage,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    WebSearchTool,
    WebSearchUserLocation,
)
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import (
    Instrumentation,
    NativeTool,
)
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import (
    UserError,
)
from pydantic_ai.messages import UploadedFile
from pydantic_ai.models import (
    CompletedStreamedResponse,
    ModelRequestParameters,
    infer_model,
    infer_model_profile,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import DEFAULT_PROFILE, ModelProfile
from pydantic_ai.tools import ToolDefinition

from ..._inline_snapshot import snapshot
from ...model_lifecycle_utils import LifecycleTrackingModel

try:
    import temporalio.api.common.v1
    from temporalio import workflow
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]
    from temporalio.client import Client
    from temporalio.contrib.opentelemetry import TracingInterceptor
    from temporalio.contrib.pydantic import PydanticPayloadConverter, pydantic_data_converter
    from temporalio.converter import (
        DataConverter,
        DefaultPayloadConverter,
        ExternalStorage,
        PayloadCodec,
        StorageDriver,
    )
    from temporalio.testing import ActivityEnvironment
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec._toolset import CallToolResult
    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        LogfirePlugin,
        PydanticAIPayloadConverter,
        PydanticAIPlugin,
        PydanticAIWorkflow,
        TemporalAgent,  # pyright: ignore[reportDeprecated]
        TemporalDurability,
        _logfire as temporal_logfire,  # pyright: ignore[reportPrivateUsage]
        _payload_converter as temporal_payload_converter,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.durable_exec.temporal._activity_execution import (
        execute_activity as execute_temporal_activity,
    )
    from pydantic_ai.durable_exec.temporal._durability import _RequestParams  # pyright: ignore[reportPrivateUsage]
    from pydantic_ai.durable_exec.temporal._model import (
        TemporalModel,
        _CancelParams as _ModelCancelParams,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.durable_exec.temporal._run_context import TemporalRunContext

    from .sandbox_workflow import PydanticAIPluginSandboxWorkflow
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)


# The 3.14 durable-exec CI leg takes this skip; every other leg falls through. `lax` rather than
# plain because which of the two arms a run measures depends on its Python version.
if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14: '
        'sandbox module state accumulates across validation cycles causing import failures after ~22 workflows '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

try:
    import logfire
    from logfire import Logfire
    from logfire._internal.config import LogfireConfig
    from logfire._internal.tracer import _ProxyTracer  # pyright: ignore[reportPrivateUsage]
    from opentelemetry.trace import ProxyTracer
except ImportError:  # pragma: lax no cover
    pytest.skip('logfire not installed', allow_module_level=True)

try:
    from pydantic_ai.mcp import MCPToolset  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError:  # pragma: lax no cover
    pytest.skip('mcp not installed', allow_module_level=True)

try:
    from pydantic_ai.models.openai import OpenAIChatModel  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError:  # pragma: lax no cover
    pytest.skip('openai not installed', allow_module_level=True)


with workflow.unsafe.imports_passed_through():
    # Workaround for a race condition when running `logfire.info` inside an activity with attributes to serialize and pandas importable:
    # AttributeError: partially initialized module 'pandas' has no attribute '_pandas_parser_CAPI' (most likely due to a circular import)
    try:
        import pandas  # pyright: ignore[reportUnusedImport] # noqa: F401
    except ImportError:  # pragma: lax no cover
        pass

    # https://github.com/temporalio/sdk-python/blob/3244f8bffebee05e0e7efefb1240a75039903dda/tests/test_client.py#L112C1-L113C1

    from ..._inline_snapshot import snapshot

    # Loads `vcr`, which Temporal doesn't like without passing through the import
    from ...conftest import IsDatetime, IsStr, try_import

    # `_shared` loads the same sandbox-sensitive modules, so import it passed-through as well.
    from ._shared import (
        BASE_ACTIVITY_CONFIG,
        TASK_QUEUE,
        DynamicToolsetDeps,
        _BuiltinToolModel,  # pyright: ignore[reportPrivateUsage]
        _durability_fn_model,  # pyright: ignore[reportPrivateUsage]
        _select_builtin_tool,  # pyright: ignore[reportPrivateUsage]
        _WebSearchOnlyModel,  # pyright: ignore[reportPrivateUsage]
        code_execution_builtin_model,
        complex_temporal_agent,
        dynamic_toolset_temporal_agent,
        model,
        payload_limit_detail,
        simple_temporal_agent,
        test_model_error_1,
        test_model_error_2,
        web_search_builtin_model,
        web_search_model,
        workflow_raises,
    )

with try_import() as anthropic_imports_successful:
    import anthropic

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

# `TemporalAgent` is deprecated in favor of `capabilities=[TemporalDurability(...)]`.
# These tests exercise the wrapper-agent path on purpose; suppress the warning here
# rather than globally in `pyproject.toml`. The `pytestmark` entry below covers warnings
# emitted *inside* test functions; the `filterwarnings` call below covers warnings emitted
# at module import time (e.g. module-level construction of `TemporalAgent`).
warnings.filterwarnings('ignore', message='`TemporalAgent` is deprecated', category=PydanticAIDeprecationWarning)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.xdist_group(name='temporal-model'),
    pytest.mark.filterwarnings(
        'ignore:`TemporalAgent` is deprecated:pydantic_ai._warnings.PydanticAIDeprecationWarning'
    ),
]


# Regression test for the workflow-sandbox passthrough list (`_workflow_runner` in
# `durable_exec/temporal/__init__.py`). A `gateway/` model named by string is constructed lazily via
# `infer_model` *inside* the workflow, so the provider's SDK is imported and its client built under
# the `SandboxedWorkflowRunner`. Provider SDKs touch the filesystem/env at construction time, which
# the sandbox forbids unless the SDK module is passed through. Every other test builds its model at
# module scope (outside the sandbox), so this seam was previously uncovered. Construction-only (no
# model request) keeps it deterministic.
@workflow.defn
class ConstructModelInWorkflow:
    @workflow.run
    async def run(self, model_name: str) -> str:
        # We assert only that construction succeeds — no request is made.
        return type(infer_model(model_name)).__name__


@pytest.mark.parametrize(
    ('model_name', 'expected_model_class'),
    [
        # Only `gateway/` providers exercise the sandbox: they import their SDK lazily inside
        # `gateway_provider()`, so the import and client construction run *inside* the workflow. Direct
        # providers (e.g. `anthropic:`) import their SDK at module level, which rides Temporal's
        # transitive passthrough of `pydantic_ai` and never trips — so they give no regression coverage.
        #
        # The reported regression: `gateway/anthropic:` in-workflow tripped the `anthropic` SDK's
        # `Path.home()` access.
        pytest.param('gateway/anthropic:claude-sonnet-4-6', 'AnthropicModel', id='gateway-anthropic'),
        # Canary: OpenAI needs no passthrough today; turns red here (not in a user's workflow) if a
        # future SDK release makes a restricted call (e.g. reads `~/...`) during construction.
        pytest.param('gateway/openai-chat:gpt-5', 'OpenAIChatModel', id='gateway-openai'),
        # Positive coverage of the `google.auth` (+`certifi`) passthrough: `google-genai` lazily
        # imports `google.auth` during construction, which the sandbox flags without it.
        pytest.param('gateway/google-cloud:gemini-2.5-pro', 'GoogleModel', id='gateway-google'),
    ],
)
async def test_model_construction_in_workflow_passes_sandbox(
    model_name: str,
    expected_model_class: str,
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
):
    # Dummy credentials suffice since no request is made. The gateway key must encode a region
    # (`pylf_v<n>_<region>_...`) so the base URL can be inferred.
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_API_KEY', 'pylf_v1_us_0123456789abcdef')

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ConstructModelInWorkflow],
        # A sandbox violation surfaces as a workflow *task* failure, which Temporal retries forever
        # by default — so a regression would hang rather than fail. Promote any in-workflow exception
        # (e.g. `RestrictedWorkflowAccessError`) to a workflow failure so it surfaces immediately.
        workflow_failure_exception_types=[Exception],
    ):
        # Without the SDK passed through this fails with a `WorkflowFailureError`: under the suite's
        # warnings-as-errors, Temporal's "imported after initial workflow load" becomes a hard error;
        # in production the SDK's restricted `Path.home()`/env access raises `RestrictedWorkflowAccessError`.
        result = await client.execute_workflow(
            ConstructModelInWorkflow.run,
            args=[model_name],
            id=f'construct_model_{re.sub(r"[^a-zA-Z0-9]", "_", model_name)}',
            task_queue=TASK_QUEUE,
        )
    assert result == expected_model_class


# Regression test for the `httpx2` stack passthrough entries in `_workflow_runner`.
# `ModelResponse.cost()` lazily imports genai-prices on first call; inside a workflow that trips the
# sandbox unless those modules are passed through (see #6215).
@workflow.defn
class CalculateCostInWorkflow:
    @workflow.run
    async def run(self) -> float:
        response = ModelResponse(
            parts=[TextPart('ok')],
            usage=RequestUsage(input_tokens=100, output_tokens=10),
            model_name='claude-sonnet-4-5',
            provider_name='anthropic',
        )
        return float(response.cost().total_price)


async def test_response_cost_in_workflow_passes_sandbox(client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CalculateCostInWorkflow],
        workflow_failure_exception_types=[Exception],
    ):
        result = await client.execute_workflow(
            CalculateCostInWorkflow.run,
            id='calculate_cost_in_workflow',
            task_queue=TASK_QUEUE,
        )
    assert result > 0


def test_temporal_model_request_activities_capture_deps_type():
    """Both model-request activities must capture the real `deps_type` as the `deps` argument type.

    `temporalio`'s `@activity.defn` freezes a function's type hints into `arg_types` at decoration time for
    payload conversion, so `deps`'s annotation has to be set before decorating. If it's set afterwards (as the
    non-streaming activity used to do), the patch is cosmetic and the activity deserializes `deps` as a raw
    dict instead of the declared deps type.
    """
    model = dynamic_toolset_temporal_agent.model
    assert isinstance(model, TemporalModel)

    # `arg_types[1]` is the `deps` argument's captured type, which drives Temporal's payload conversion.
    deps_type = DynamicToolsetDeps | None
    request_arg_types = ActivityDefinition.must_from_callable(model.request_activity).arg_types  # pyright: ignore[reportUnknownMemberType]
    stream_arg_types = ActivityDefinition.must_from_callable(model.request_stream_activity).arg_types  # pyright: ignore[reportUnknownMemberType]
    assert request_arg_types is not None and request_arg_types[1] == deps_type
    assert stream_arg_types is not None and stream_arg_types[1] == deps_type


@workflow.defn
class SimpleAgentWorkflowWithOverrideBuiltinTools:
    @workflow.run
    async def run(self, prompt: str) -> None:
        with simple_temporal_agent.override(native_tools=[WebSearchTool()]):
            pass


async def test_temporal_agent_override_builtin_tools_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideBuiltinTools],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Native tools cannot be contextually overridden inside a Temporal workflow, they must be set at agent creation time.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithOverrideBuiltinTools.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithOverrideBuiltinTools.__name__,
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class DirectStreamWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        messages: list[ModelMessage] = [ModelRequest.user_text_prompt(prompt)]
        async with model_request_stream(complex_temporal_agent.model, messages) as stream:
            async for _ in stream:
                pass
        return 'done'  # pragma: no cover


async def test_temporal_model_stream_direct(client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DirectStreamWorkflow],
        plugins=[AgentPlugin(complex_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'A Temporal model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            ),
        ):
            await client.execute_workflow(
                DirectStreamWorkflow.run,
                args=['What is the capital of Mexico?'],
                id=DirectStreamWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )


async def test_logfire_plugin(client: Client):
    def setup_logfire(send_to_logfire: bool = True, metrics: Literal[False] | None = None) -> Logfire:
        instance = logfire.configure(local=True, metrics=metrics)
        instance.config.token = 'test'
        instance.config.send_to_logfire = send_to_logfire
        return instance

    plugin = LogfirePlugin(setup_logfire)

    config = client.config()
    config['plugins'] = [plugin]
    new_client = Client(**config)

    interceptor = new_client.config(active_config=True)['interceptors'][0]
    assert isinstance(interceptor, TracingInterceptor)
    if isinstance(interceptor.tracer, ProxyTracer):
        assert interceptor.tracer._instrumenting_module_name == 'temporalio'  # pyright: ignore[reportPrivateUsage] # pragma: lax no cover
    elif isinstance(interceptor.tracer, _ProxyTracer):
        assert interceptor.tracer.instrumenting_module_name == 'temporalio'  # pragma: lax no cover
    else:
        assert False, f'Unexpected tracer type: {type(interceptor.tracer)}'  # pragma: no cover

    with patch.object(
        temporal_logfire, 'OpenTelemetryConfig', wraps=temporal_logfire.OpenTelemetryConfig
    ) as open_telemetry_config:
        new_client = await Client.connect(client.service_client.config.target_host, plugins=[plugin])
    assert new_client.service_client.config.runtime is not None
    assert open_telemetry_config.call_args.kwargs['metric_periodicity'] == timedelta(seconds=60)

    plugin = LogfirePlugin(setup_logfire, metric_periodicity=timedelta(minutes=5))
    with patch.object(
        temporal_logfire, 'OpenTelemetryConfig', wraps=temporal_logfire.OpenTelemetryConfig
    ) as open_telemetry_config:
        await Client.connect(client.service_client.config.target_host, plugins=[plugin])
    assert open_telemetry_config.call_args.kwargs['metric_periodicity'] == timedelta(minutes=5)

    plugin = LogfirePlugin(setup_logfire, metrics=False)
    custom_runtime = temporal_logfire.Runtime(telemetry=temporal_logfire.TelemetryConfig())
    new_client = await Client.connect(
        client.service_client.config.target_host, plugins=[plugin], runtime=custom_runtime
    )
    assert new_client.service_client.config.runtime is custom_runtime

    plugin = LogfirePlugin(lambda: setup_logfire(send_to_logfire=False))
    new_client = await Client.connect(client.service_client.config.target_host, plugins=[plugin])
    assert new_client.service_client.config.runtime is None

    plugin = LogfirePlugin(lambda: setup_logfire(metrics=False))
    new_client = await Client.connect(client.service_client.config.target_host, plugins=[plugin])
    assert new_client.service_client.config.runtime is None


@pytest.mark.parametrize('already_configured', [True, False])
async def test_logfire_plugin_default_setup(client: Client, monkeypatch: pytest.MonkeyPatch, already_configured: bool):
    """The default setup only calls `logfire.configure()` when Logfire isn't configured yet.

    `logfire.configure()` is a reset rather than an additive call: it re-derives every unspecified
    argument from the environment and shuts down the existing tracer provider. Calling it on every
    `Client.connect()` silently discarded a host's own scrubbing patterns, additional span processors,
    console settings, and service name. Pydantic AI is instrumented either way.

    `logfire.DEFAULT_LOGFIRE_INSTANCE` is swapped for a stand-in so the assertions don't depend on
    (or disturb) whatever configuration the rest of the test session has installed globally.
    """
    instance = (
        logfire.configure(local=True, send_to_logfire=False) if already_configured else Logfire(config=LogfireConfig())
    )
    assert instance.config._initialized is already_configured  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(logfire, 'DEFAULT_LOGFIRE_INSTANCE', instance)

    configure_calls: list[dict[str, Any]] = []
    instrumented: list[Logfire] = []

    def configure(**kwargs: Any) -> Logfire:
        configure_calls.append(kwargs)
        return instance

    def instrument_pydantic_ai(self: Logfire, *args: Any, **kwargs: Any) -> None:
        instrumented.append(self)

    monkeypatch.setattr(logfire, 'configure', configure)
    monkeypatch.setattr(Logfire, 'instrument_pydantic_ai', instrument_pydantic_ai)

    await Client.connect(client.service_client.config.target_host, plugins=[LogfirePlugin()])

    assert configure_calls == ([] if already_configured else [{}])
    assert instrumented == [instance]


@pytest.mark.parametrize('already_instrumented', [True, False])
def test_logfire_plugin_default_setup_preserves_instrumentation(
    monkeypatch: pytest.MonkeyPatch, already_instrumented: bool
):
    """The default setup leaves a host's own Pydantic AI instrumentation settings alone.

    `instrument_pydantic_ai()` replaces rather than merges `Agent._instrument_default`, so calling it
    unconditionally turned a deliberate `include_content=False` back on, putting prompts, completions
    and tool call results on exported spans. A host that hasn't instrumented is still instrumented.

    As in `test_logfire_plugin_default_setup` above, `logfire.DEFAULT_LOGFIRE_INSTANCE`, `configure`
    and `instrument_pydantic_ai` are swapped for stand-ins so the assertions neither depend on nor
    disturb whatever configuration the rest of the test session has installed globally.
    """
    instance = Logfire(config=LogfireConfig())
    monkeypatch.setattr(logfire, 'DEFAULT_LOGFIRE_INSTANCE', instance)

    instrumented: list[Logfire] = []

    def configure(**kwargs: Any) -> Logfire:
        return instance

    def instrument_pydantic_ai(self: Logfire, *args: Any, **kwargs: Any) -> None:
        instrumented.append(self)

    monkeypatch.setattr(logfire, 'configure', configure)
    monkeypatch.setattr(Logfire, 'instrument_pydantic_ai', instrument_pydantic_ai)

    settings = InstrumentationSettings(include_content=False, include_binary_content=False)
    monkeypatch.setattr(Agent, '_instrument_default', settings if already_instrumented else False)

    temporal_logfire._default_setup_logfire()  # pyright: ignore[reportPrivateUsage]

    # With a stand-in in place, whether the plugin instruments at all is the observable: the stand-in
    # deliberately doesn't assign `_instrument_default`, so asserting on it here would prove nothing.
    assert instrumented == ([] if already_instrumented else [instance])
    if already_instrumented:
        assert Agent._instrument_default is settings  # pyright: ignore[reportPrivateUsage]


image_agent = Agent(model, name='image_agent', output_type=BinaryImage)


# This needs to be done before the `TemporalAgent` is bound to the workflow.
image_temporal_agent = TemporalAgent(image_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class ImageAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> BinaryImage:
        result = await image_temporal_agent.run(prompt)
        return result.output  # pragma: no cover


async def test_image_agent(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ImageAgentWorkflow],
        plugins=[AgentPlugin(image_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Image output is not supported with Temporal because the image would ride the activity payload, '
                'which is capped by the server blob-size limit (2MB by default, leaving about 1.5MB of raw image '
                'bytes once base64-encoded).'
            ),
        ):
            await client.execute_workflow(
                ImageAgentWorkflow.run,
                args=['Generate an image of an axolotl.'],
                id=ImageAgentWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )


async def _respond_with_oversized_image(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # A native image-generation tool puts the image on the response like this, so it rides the
    # model-request activity payload rather than a tool-call one.
    return ModelResponse(
        parts=[
            TextPart('here is your image'),
            FilePart(content=BinaryImage(data=b'\x00' * 1_600_000, media_type='image/png')),
        ]
    )


oversized_model_response_agent = Agent(
    FunctionModel(_respond_with_oversized_image, model_name='oversized-response-model'),
    name='oversized_model_response_agent',
    deps_type=type(None),
    capabilities=[TemporalDurability(activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=60)))],
)


@workflow.defn
class OversizedModelResponseWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await oversized_model_response_agent.run(prompt)
        return result.output  # pragma: no cover


async def test_oversized_model_response_payload(client: Client):
    """A model response carrying binary content over Temporal's payload limit points at the cause (#7110).

    The `allow_image_output` guard doesn't cover this: it fires on the agent's `output_type`, while a
    native image-generation tool returns the image as a `FilePart` on the model response instead.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[OversizedModelResponseWorkflow],
        plugins=[AgentPlugin(oversized_model_response_agent)],
    ):
        with workflow_raises(
            UserError,
            f"The response from model 'function:oversized-response-model' is too large for Temporal. {payload_limit_detail(2134150)}. Binary content like an image is base64-encoded into the activity payload, so if that is the cause, the raw-byte budget is about three quarters of the limit — roughly 1.5MB at the 2MB default. A generated image is the usual cause, so ask the model for a smaller one through the model settings; a streamed segment can also overflow on its buffered events alone. To keep large payloads out of the workflow history without changing what your tools or models return, configure Temporal external storage (or a claim-check `payload_codec`) on your `DataConverter` — `PydanticAIPlugin` preserves it, and it covers every payload in both directions. See https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/#large-payloads",
        ):
            await client.execute_workflow(
                OversizedModelResponseWorkflow.run,
                args=['Draw me something.'],
                id=OversizedModelResponseWorkflow.__name__,
                task_queue=TASK_QUEUE,
                execution_timeout=timedelta(seconds=30),
            )


# ============================================================================
# DocumentUrl Serialization Test - Verifies that DocumentUrl with custom
# media_type is properly serialized through Temporal activities
# ============================================================================

document_url_agent = Agent(
    TestModel(custom_output_args={'url': 'https://example.com/doc/12345', 'media_type': 'application/pdf'}),
    name='document_url_agent',
    output_type=DocumentUrl,
)


document_url_temporal_agent = TemporalAgent(document_url_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class DocumentUrlAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> DocumentUrl:
        result = await document_url_temporal_agent.run(prompt)
        return result.output


async def test_document_url_serialization_preserves_media_type(allow_model_requests: None, client: Client):
    """Test that `DocumentUrl` with custom `media_type` is preserved through Temporal serialization.

    This is a regression test for https://github.com/pydantic/pydantic-ai/issues/3949
    where `DocumentUrl.media_type` (a computed field) was lost during Temporal activity
    serialization because the backing field `_media_type` was excluded from serialization.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DocumentUrlAgentWorkflow],
        plugins=[AgentPlugin(document_url_temporal_agent)],
    ):
        output = await client.execute_workflow(
            DocumentUrlAgentWorkflow.run,
            args=['Return a document'],
            id=DocumentUrlAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            DocumentUrl(url='https://example.com/doc/12345', _media_type='application/pdf', _identifier='eb8998')
        )


# ============================================================================
# UploadedFile Serialization Test - Verifies that UploadedFile with custom
# media_type is properly serialized through Temporal activities
# ============================================================================

uploaded_file_agent = Agent(
    TestModel(
        custom_output_args={
            'file_id': 'file-abc123',
            'provider_name': 'openai',
            'media_type': 'image/png',
            'identifier': 'file-1',
        }
    ),
    name='uploaded_file_agent',
    output_type=UploadedFile,
)


uploaded_file_temporal_agent = TemporalAgent(uploaded_file_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class UploadedFileAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> UploadedFile:
        result = await uploaded_file_temporal_agent.run(prompt)
        return result.output


async def test_uploaded_file_serialization_preserves_media_type(allow_model_requests: None, client: Client):
    """Test that `UploadedFile` with custom `media_type` is preserved through Temporal serialization."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[UploadedFileAgentWorkflow],
        plugins=[AgentPlugin(uploaded_file_temporal_agent)],
    ):
        output = await client.execute_workflow(
            UploadedFileAgentWorkflow.run,
            args=['Return a file reference'],
            id=UploadedFileAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            UploadedFile(file_id='file-abc123', provider_name='openai', _media_type='image/png', _identifier='file-1')
        )


web_search_agent = Agent(
    web_search_model,
    name='web_search_agent',
    capabilities=[NativeTool(WebSearchTool(user_location=WebSearchUserLocation(city='Mexico City', country='MX')))],
)


# This needs to be done before the `TemporalAgent` is bound to the workflow.
web_search_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    web_search_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
    model_activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=300)),
)


@workflow.defn
class WebSearchAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await web_search_temporal_agent.run(prompt)
        return result.output


async def test_web_search_agent_run_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WebSearchAgentWorkflow],
        plugins=[AgentPlugin(web_search_temporal_agent)],
    ):
        output = await client.execute_workflow(
            WebSearchAgentWorkflow.run,
            args=['In one sentence, what is the top news story in my country today?'],
            id=WebSearchAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            'Severe floods and landslides across Veracruz, Hidalgo, and Puebla have cut off hundreds of communities and left dozens dead and many missing, prompting a major federal emergency response. ([apnews.com](https://apnews.com/article/5d036e18057361281e984b44402d3b1b?utm_source=openai))'
        )


def test_temporal_run_context_serialization_is_exhaustive():
    """Every `RunContext` field must be consciously categorized for Temporal serialization.

    Guards against silent drift: when a `RunContext` field is added, this test fails until
    the author either includes it in `TemporalRunContext.serialize_run_context` or lists it
    in `intentionally_unserialized` below with a reason. Without that decision a new field
    silently becomes unavailable inside a Temporal activity (the `__getattribute__` guard
    raises `UserError` on access), which is how the deferred-capability fields were missed.
    """
    # Fields deliberately NOT carried across the activity boundary, each with its reason.
    intentionally_unserialized = {
        'deps',  # passed separately to deserialize_run_context
        'agent',  # reattached after deserialize by deserialize_run_context
        'model',  # live Model instance, not serializable
        '_model_id',  # carried separately by operations that rebuild ctx.model worker-side
        'tracer',  # live tracer, not serializable
        'tool_manager',  # live ToolManager, not serializable (documented on the field)
        'capabilities',  # live capability objects (toolsets/hooks/callables), not serializable
        'root_capability',  # live capability chain, not serializable; reattached from the bound agent by deserialize_run_context
        'pending_messages',  # live run queue, meaningless outside the running agent; replaced by an EnqueueGuard
        'messages',  # full history would be duplicated into every activity payload, against Temporal's 2MB limit
        'prompt',  # multi-modal BinaryContent would ride in every payload, against Temporal's 2MB limit; text-only subclasses can opt in
        'validation_context',  # arbitrary user object with no serialization contract
        'model_settings',  # only set for model requests, which receive it as their own typed activity param
        '_mcp_tool_defs_cache',  # run-local cache read/written in workflow code; never needed inside an activity
        '_event_stream_buffer',  # run-local event buffer drained in workflow code; a public emit surface for activities is a follow-up
        'realtime_session',  # live RealtimeSession, not serializable; realtime sessions don't run inside Temporal activities
        '_cancellation',  # runtime-only controller holding a live asyncio task reference; cannot cross the activity boundary
        '_durable_operations',  # workflow-side callables cannot cross the activity boundary; worker dispatch is pre-registered
        '_run_capabilities_by_id',  # live per-run capability instances are recovered from the worker agent instead
    }
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    serialized = set(TemporalRunContext.serialize_run_context(ctx))
    all_fields = set(RunContext.__dataclass_fields__)

    overlap = serialized & intentionally_unserialized
    assert not overlap, f'Fields both serialized and excluded: {overlap}'

    uncategorized = all_fields - (serialized | intentionally_unserialized)
    assert not uncategorized, (
        f'Uncategorized `RunContext` fields: {uncategorized}. Add each to '
        '`TemporalRunContext.serialize_run_context` or to `intentionally_unserialized` (with a reason).'
    )


# Multi-Model Support Tests

# Module-level test models for multi-model selection test
test_model_selection_1 = TestModel(custom_output_text='Response from model 1')

test_model_selection_2 = TestModel(custom_output_text='Response from model 2')

test_model_selection_3 = TestModel(custom_output_text='Response from model 3')

test_model_error_unregistered = TestModel()


# Module-level temporal agents
agent_selection = Agent(test_model_selection_1, name='multi_model_workflow_test')

multi_model_selection_test_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    agent_selection,
    name='multi_model_workflow_test',
    models={
        'model_2': test_model_selection_2,
        'model_3': test_model_selection_3,
    },
    activity_config=BASE_ACTIVITY_CONFIG,
)


agent_error = Agent(test_model_error_1, name='error_test')

multi_model_error_test_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    agent_error,
    name='error_test',
    models={'other': test_model_error_2},
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class MultiModelWorkflow:
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await multi_model_selection_test_agent.run(prompt, model=model_id)
        return result.output


builtin_tool_agent = Agent(
    web_search_builtin_model,
    name='builtin_tool_dynamic_agent',
    capabilities=[NativeTool(_select_builtin_tool)],
)


builtin_tool_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    builtin_tool_agent,
    name='builtin_tool_dynamic_agent',
    models={'code': code_execution_builtin_model},
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class BuiltinToolWorkflow:
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await builtin_tool_temporal_agent.run(prompt, model=model_id)
        return result.output


# Model that does NOT support any builtin tools (used as default)
no_builtin_support_model = _BuiltinToolModel(custom_output_text='no builtin support', model_name='no-builtin-test')


# Model that DOES support WebSearchTool (registered as alternate model)
web_search_builtin_override_model = _WebSearchOnlyModel(
    custom_output_text='web search response',
    model_name='web-search-override',
)


# Agent initialized with model that doesn't support builtins, but has builtin tools configured
builtins_in_workflow_agent = Agent(
    no_builtin_support_model,
    capabilities=[NativeTool(WebSearchTool()), Instrumentation(settings=InstrumentationSettings())],
    name='builtins_in_workflow',
)


# TemporalAgent registers an alternate model that DOES support builtins
builtins_in_workflow_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    builtins_in_workflow_agent,
    name='builtins_in_workflow',
    models={'web_search': web_search_builtin_override_model},
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class BuiltinsInWorkflow(PydanticAIWorkflow):
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await builtins_in_workflow_temporal_agent.run(prompt, model=model_id)
        return result.output


@workflow.defn
class MultiModelWorkflowUnregistered:
    @workflow.run
    async def run(self, prompt: str) -> str:
        # Try to use an unregistered model
        result = await multi_model_error_test_agent.run(prompt, model=test_model_error_unregistered)
        return result.output  # pragma: no cover


async def test_temporal_agent_multi_model_reserved_id():
    """Test that reserved model IDs raise helpful errors."""
    test_model1 = TestModel()
    test_model2 = TestModel()

    agent = Agent(test_model1, name='reserved_id_test')
    with pytest.raises(UserError, match="Model ID 'default' is reserved"):
        TemporalAgent(  # pyright: ignore[reportDeprecated]
            agent,
            name='reserved_id_test',
            models={'default': test_model2},
        )


async def test_temporal_agent_multi_model_selection_in_workflow(allow_model_requests: None, client: Client):
    """Test selecting different models in a workflow using the model parameter."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModelWorkflow],
        plugins=[AgentPlugin(multi_model_selection_test_agent)],
    ):
        # Test using default model (model_id=None)
        output = await client.execute_workflow(
            MultiModelWorkflow.run,
            args=['Hello', None],
            id='MultiModelWorkflow_default',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 1'

        # Test selecting second model by ID
        output = await client.execute_workflow(
            MultiModelWorkflow.run,
            args=['Hello', 'model_2'],
            id='MultiModelWorkflow_model2',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 2'

        # Test selecting third model by ID
        output = await client.execute_workflow(
            MultiModelWorkflow.run,
            args=['Hello', 'model_3'],
            id='MultiModelWorkflow_model3',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 3'


async def test_temporal_dynamic_builtin_tools_select_by_model(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[BuiltinToolWorkflow],
        plugins=[AgentPlugin(builtin_tool_temporal_agent)],
    ):
        output = await client.execute_workflow(
            BuiltinToolWorkflow.run,
            args=['Hello', None],
            id='BuiltinToolWorkflow_default',
            task_queue=TASK_QUEUE,
        )
        assert output == 'search model'
        assert isinstance(web_search_builtin_model.last_model_request_parameters, ModelRequestParameters)
        assert web_search_builtin_model.last_model_request_parameters.native_tools
        assert isinstance(web_search_builtin_model.last_model_request_parameters.native_tools[0], WebSearchTool)

        output = await client.execute_workflow(
            BuiltinToolWorkflow.run,
            args=['Hello', 'code'],
            id='BuiltinToolWorkflow_code',
            task_queue=TASK_QUEUE,
        )
        assert output == 'code model'
        assert isinstance(code_execution_builtin_model.last_model_request_parameters, ModelRequestParameters)
        assert code_execution_builtin_model.last_model_request_parameters.native_tools
        assert isinstance(
            code_execution_builtin_model.last_model_request_parameters.native_tools[0],
            CodeExecutionTool,
        )


async def test_builtins_in_workflow_with_runtime_model_override(allow_model_requests: None, client: Client):
    """Test that builtin tools work when agent is initialized with a non-supporting model
    but run with a model that does support builtins."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[BuiltinsInWorkflow],
        plugins=[AgentPlugin(builtins_in_workflow_temporal_agent)],
    ):
        # Run with the model that supports WebSearchTool
        result = await client.execute_workflow(
            BuiltinsInWorkflow.run,
            args=['search for something', 'web_search'],
            id='BuiltinsInWorkflow',
            task_queue=TASK_QUEUE,
        )
        assert result == 'web search response'

    # Verify the web search model received the WebSearchTool in its request parameters
    assert isinstance(web_search_builtin_override_model.last_model_request_parameters, ModelRequestParameters)
    assert web_search_builtin_override_model.last_model_request_parameters.native_tools
    assert isinstance(
        web_search_builtin_override_model.last_model_request_parameters.native_tools[0],
        WebSearchTool,
    )


async def test_temporal_agent_multi_model_unregistered_error(allow_model_requests: None, client: Client):
    """Test that using an unregistered model raises a helpful error."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModelWorkflowUnregistered],
        plugins=[AgentPlugin(multi_model_error_test_agent)],
    ):
        with workflow_raises(
            UserError,
            'Arbitrary model instances cannot be used at runtime inside a Temporal workflow. Register the model via `models` or reference a registered model by id.',
        ):
            await client.execute_workflow(
                MultiModelWorkflowUnregistered.run,
                args=['Hello'],
                id='MultiModelWorkflowUnregistered',
                task_queue=TASK_QUEUE,
            )


async def test_temporal_agent_multi_model_outside_workflow():
    """Test that multi-model agents work outside workflows (using wrapped agent behavior).

    Outside a workflow, a TemporalAgent should behave like a regular Agent.
    This includes supporting model selection by registered ID or instance.
    """
    test_model1 = TestModel(custom_output_text='Model 1 response')
    test_model2 = TestModel(custom_output_text='Model 2 response')
    test_model_unregistered = TestModel(custom_output_text='Unregistered model response')

    agent = Agent(test_model1, name='outside_workflow_test')
    temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
        agent,
        name='outside_workflow_test',
        models={'secondary': test_model2},
    )

    # Outside workflow, should use default model
    result = await temporal_agent.run('Hello')
    assert result.output == 'Model 1 response'

    # Outside workflow, passing a registered model ID should also work
    result = await temporal_agent.run('Hello', model='secondary')
    assert result.output == 'Model 2 response'

    # Passing a registered model instance should also work
    result = await temporal_agent.run('Hello', model=test_model2)
    assert result.output == 'Model 2 response'

    # Passing an unregistered model instance should also work outside workflow
    result = await temporal_agent.run('Hello', model=test_model_unregistered)
    assert result.output == 'Unregistered model response'


async def test_temporal_agent_without_default_model():
    """Test that a TemporalAgent can be created without a default model if models is provided.

    When no model is provided to run(), the first registered model should be used.
    """
    test_model1 = TestModel(custom_output_text='Model 1 response')
    test_model2 = TestModel(custom_output_text='Model 2 response')

    # Agent without a model
    agent = Agent(name='no_default_model_test')
    temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
        agent,
        name='no_default_model_test',
        models={
            'primary': test_model1,
            'secondary': test_model2,
        },
    )

    # Without a model, should use the first registered model
    result = await temporal_agent.run('Hello')
    assert result.output == 'Model 1 response'

    # Outside workflow, can use registered models by id
    result = await temporal_agent.run('Hello', model='primary')
    assert result.output == 'Model 1 response'

    result = await temporal_agent.run('Hello', model='secondary')
    assert result.output == 'Model 2 response'


# Workflow for testing passing model instances (can't be workflow args, so map by key)
_model_instance_map = {
    'default_instance': test_model_selection_1,
    'model_2_instance': test_model_selection_2,
}


@workflow.defn
class MultiModelWorkflowInstance:
    @workflow.run
    async def run(self, prompt: str, instance_key: str) -> str:
        model_instance = _model_instance_map[instance_key]
        result = await multi_model_selection_test_agent.run(prompt, model=model_instance)
        return result.output


@pytest.mark.parametrize(
    ('model_id', 'expected_output'),
    [
        pytest.param('default', 'Response from model 1', id='default_explicit'),
    ],
)
async def test_temporal_agent_model_selection_by_id(
    allow_model_requests: None, client: Client, model_id: str, expected_output: str
):
    """Test model selection by passing model ID strings."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModelWorkflow],
        plugins=[AgentPlugin(multi_model_selection_test_agent)],
    ):
        output = await client.execute_workflow(
            MultiModelWorkflow.run,
            args=['Hello', model_id],
            id=f'MultiModelWorkflow_{model_id}',
            task_queue=TASK_QUEUE,
        )
        assert output == expected_output


@pytest.mark.parametrize(
    ('instance_key', 'expected_output'),
    [
        pytest.param('default_instance', 'Response from model 1', id='default_instance'),
        pytest.param('model_2_instance', 'Response from model 2', id='registered_instance'),
    ],
)
async def test_temporal_agent_model_selection_by_instance(
    allow_model_requests: None, client: Client, instance_key: str, expected_output: str
):
    """Test model selection by passing model instances."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModelWorkflowInstance],
        plugins=[AgentPlugin(multi_model_selection_test_agent)],
    ):
        output = await client.execute_workflow(
            MultiModelWorkflowInstance.run,
            args=['Hello', instance_key],
            id=f'MultiModelWorkflowInstance_{instance_key}',
            task_queue=TASK_QUEUE,
        )
        assert output == expected_output


def test_temporal_model_profile_for_raw_strings():
    """Test TemporalModel infers model_name, system, and profile from raw strings without constructing providers."""

    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__profile_inference',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # Without using_model, properties come from default
    assert temporal_model.profile == default_model.profile
    assert temporal_model.model_name == default_model.model_name
    assert temporal_model.system == default_model.system

    # With raw string, all properties are inferred correctly
    with temporal_model.using_model('openai:gpt-5'):
        assert temporal_model.model_name == 'gpt-5'
        assert temporal_model.system == 'openai'
        assert temporal_model.profile == infer_model_profile('openai:gpt-5')

    # Anthropic profile inference includes WebSearchTool support
    with temporal_model.using_model('anthropic:claude-sonnet-4-5'):
        assert temporal_model.model_name == 'claude-sonnet-4-5'
        assert temporal_model.system == 'anthropic'
        assert temporal_model.profile == infer_model_profile('anthropic:claude-sonnet-4-5')

    # Registered models work correctly for all properties
    alt_model = TestModel(custom_output_text='alt', model_name='alt-model')
    temporal_model_with_registry = TemporalModel(
        default_model,
        activity_name_prefix='test__profile_registry',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
        models={'alt': alt_model},
    )
    with temporal_model_with_registry.using_model('alt'):
        assert temporal_model_with_registry.model_name == 'alt-model'
        assert temporal_model_with_registry.system == alt_model.system
        assert temporal_model_with_registry.profile == alt_model.profile


class DefaultHostModel(TestModel):
    @property
    def base_url(self) -> str:
        return 'https://default.example.com:1111/v1'


class AltHostModel(TestModel):
    @property
    def base_url(self) -> str:
        return 'https://alt.example.com:2222/v1'


def test_temporal_model_base_url_follows_active_model():
    """`base_url` resolves through `using_model()` like the other identity properties.

    Without this it would report the wrapped default's URL, so a request span would name the active
    model in `gen_ai.request.model` while pointing `server.address` at a different model's host.
    """
    temporal_model = TemporalModel(
        DefaultHostModel(model_name='default-model'),
        activity_name_prefix='test__base_url',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
        models={'alt': AltHostModel(model_name='alt-model')},
    )

    assert temporal_model.base_url == snapshot('https://default.example.com:1111/v1')

    with temporal_model.using_model('alt'):
        assert temporal_model.base_url == snapshot('https://alt.example.com:2222/v1')

    with temporal_model.using_model('openai:gpt-5'):
        assert temporal_model.base_url is None


def test_temporal_model_model_id_follows_active_model():
    """`model_id` resolves through `using_model()` rather than reporting the wrapped default's.

    `WrapperModel` forwards `model_id` so a wrapped `FallbackModel` keeps its own composed ID, which
    would otherwise pin this to the default model. The ID names the activity a request runs under, so
    a swapped-in model has to be the one it reports.
    """
    temporal_model = TemporalModel(
        TestModel(model_name='default-model'),
        activity_name_prefix='test__model_id',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
        models={'alt': FallbackModel(TestModel(model_name='alt-model'), TestModel(model_name='spare-model'))},
    )

    assert temporal_model.model_id == snapshot('test:default-model')

    with temporal_model.using_model('alt'):
        assert temporal_model.model_id == snapshot('fallback:test:alt-model,test:spare-model')

    with temporal_model.using_model('openai:gpt-5'):
        assert temporal_model.model_id == snapshot('openai:gpt-5')

    with temporal_model.using_model('gpt-5'):
        assert temporal_model.model_id == snapshot('test:gpt-5')


async def test_temporal_model_request_outside_workflow():
    """Test that TemporalModel.request() falls back to wrapped model outside a workflow.

    When TemporalModel.request() is called directly (not through TemporalAgent.run())
    and not inside a Temporal workflow, it should delegate to the wrapped model's request method.
    """
    test_model = TestModel(custom_output_text='Direct model response')

    temporal_model = TemporalModel(
        test_model,
        activity_name_prefix='test__direct_request',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # Call request() directly - outside a workflow, this should fall back to super().request()
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
    response = await temporal_model.request(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            native_tools=[],
            output_mode='text',
            allow_text_output=True,
            output_tools=[],
            output_object=None,
        ),
    )

    # Verify response comes from the wrapped TestModel
    assert any(isinstance(part, TextPart) and part.content == 'Direct model response' for part in response.parts)


async def test_temporal_activities_manage_inferred_model_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class TrackingModel(LifecycleTrackingModel):
        def request_stream(self, *args: Any, **kwargs: Any) -> Any:
            events.append('stream')
            return super().request_stream(*args, **kwargs)

        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            events.append('cancel')

    async def handle_stream(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        events.append('handler')

    temporal_model = TemporalModel(
        TestModel(),
        activity_name_prefix='test__inferred_lifecycle',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=object,
        event_stream_handler=handle_stream,
    )

    def infer_tracking_model(model_id: str, ctx: RunContext[object] | None) -> TrackingModel:
        return TrackingModel(events, include_exit_exception=False)

    monkeypatch.setattr(temporal_model, '_infer_model', infer_tracking_model)
    deps = object()
    ctx = RunContext[object](deps=deps, model=TestModel(), usage=RunUsage(), run_id='inferred-lifecycle')
    params = _RequestParams(
        messages=[ModelRequest.user_text_prompt('hello')],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
        model_id='rebuilt',
    )

    await ActivityEnvironment().run(temporal_model.request_activity, params, deps)
    await ActivityEnvironment().run(
        temporal_model.request_stream_activity,
        params,
        deps,  # pyright: ignore[reportArgumentType]
    )
    await ActivityEnvironment().run(
        temporal_model.cancel_suspended_response_activity,
        _ModelCancelParams(
            response=ModelResponse(parts=[TextPart('paused')], state='suspended'),
            model_id='rebuilt',
            serialized_run_context=params.serialized_run_context,
            deps=deps,
        ),
    )

    assert events == [
        'enter',
        'request',
        'exit',
        'enter',
        'stream',
        'handler',
        'exit',
        'enter',
        'cancel',
        'exit',
    ]


async def test_temporal_activity_does_not_manage_registered_models() -> None:
    class TrackingModel(LifecycleTrackingModel):
        def request_stream(self, *args: Any, **kwargs: Any) -> Any:
            self.events.append('stream')
            return super().request_stream(*args, **kwargs)

        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            self.events.append('cancel')

    async def handle_stream(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        pass

    events: list[str] = []
    default = TrackingModel(events, include_exit_exception=False, custom_output_text='default')
    registered = TrackingModel(events, include_exit_exception=False, custom_output_text='registered')
    temporal_model = TemporalModel(
        default,
        activity_name_prefix='test__registered_lifecycle',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=object,
        models={'registered': registered},
        event_stream_handler=handle_stream,
    )
    deps = object()
    ctx = RunContext[object](deps=deps, model=default, usage=RunUsage(), run_id='registered-lifecycle')
    params = _RequestParams(
        messages=[ModelRequest.user_text_prompt('hello')],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
    )

    # A `model_id` of `None` resolves to the agent default, which is registered too, so it must not
    # be entered either.
    await ActivityEnvironment().run(temporal_model.request_activity, params, deps)
    assert events == ['request']

    events = []
    default.events = events
    registered.events = events
    params.model_id = 'registered'
    await ActivityEnvironment().run(temporal_model.request_activity, params, deps)
    assert events == ['request']

    events = []
    registered.events = events
    await ActivityEnvironment().run(
        temporal_model.request_stream_activity,
        params,
        deps,  # pyright: ignore[reportArgumentType]
    )
    assert events == ['stream']

    events = []
    registered.events = events
    await ActivityEnvironment().run(
        temporal_model.cancel_suspended_response_activity,
        _ModelCancelParams(
            response=ModelResponse(parts=[TextPart('paused')], state='suspended'),
            model_id='registered',
            serialized_run_context=params.serialized_run_context,
            deps=deps,
        ),
    )
    assert events == ['cancel']


async def test_temporal_model_cancel_suspended_response_outside_workflow():
    """`TemporalModel.cancel_suspended_response()` falls back to the wrapped model outside a workflow.

    Inside a workflow it runs the provider teardown in the `model_cancel_suspended_response` activity
    (registered in `temporal_activities`) so the raw HTTP call never runs in the workflow sandbox;
    outside a workflow it delegates straight to the wrapped model.
    """
    cancelled: list[ModelResponse] = []

    class RecordingModel(TestModel):
        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append(response)

    temporal_model = TemporalModel(
        RecordingModel(),
        activity_name_prefix='test__direct_cancel',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # The cancel activity is registered alongside the request activities.
    assert [
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in temporal_model.temporal_activities
    ] == snapshot(
        [
            'test__direct_cancel__model_request',
            'test__direct_cancel__model_request_stream',
            'test__direct_cancel__model_cancel_suspended_response',
        ]
    )

    response = ModelResponse(parts=[TextPart('paused')], state='suspended')
    await temporal_model.cancel_suspended_response(response)
    assert cancelled == [response]


@dataclass
class CancelTenantDeps:
    tenant_id: str


factory_cancel_calls: list[tuple[CancelTenantDeps, str]] = []

factory_cancelled_responses: list[ModelResponse] = []


def cancel_provider_factory(ctx: RunContext[object], provider_name: str) -> Any:
    assert isinstance(ctx.deps, CancelTenantDeps)
    factory_cancel_calls.append((ctx.deps, provider_name))
    return object()


class FactoryCancelRecordingModel(TestModel):
    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        factory_cancelled_responses.append(response)


factory_cancel_temporal_model = TemporalModel(
    TestModel(),
    activity_name_prefix='test__factory_cancel',
    activity_config=BASE_ACTIVITY_CONFIG,
    deps_type=CancelTenantDeps,
    provider_factory=cancel_provider_factory,
)


@workflow.defn
class FactoryCancelWorkflow:
    @workflow.run
    async def run(self, response: ModelResponse) -> None:
        deps = CancelTenantDeps(tenant_id='tenant-a')
        ctx = RunContext[CancelTenantDeps](deps=deps, model=TestModel(), usage=RunUsage(), run_id='factory-cancel')
        await execute_temporal_activity(
            activity=factory_cancel_temporal_model.cancel_suspended_response_activity,
            args=[
                _ModelCancelParams(
                    response=response,
                    model_id='runtime-provider:model',
                    serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
                    deps=deps,
                )
            ],
            **BASE_ACTIVITY_CONFIG,
        )


async def test_temporal_model_cancel_suspended_response_uses_provider_factory(client: Client) -> None:
    """A real worker preserves structured deps used to select the cancellation client."""
    factory_cancel_calls.clear()
    factory_cancelled_responses.clear()
    factory_model = FactoryCancelRecordingModel()

    def infer_runtime_model(model_id: str, provider_factory: Callable[[str], object] | None = None):
        assert provider_factory is not None
        provider_factory('runtime-provider')
        return factory_model

    response = ModelResponse(parts=[TextPart('paused')], state='suspended')
    with patch('pydantic_ai.durable_exec.temporal._model.models.infer_model', side_effect=infer_runtime_model):
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[FactoryCancelWorkflow],
            activities=factory_cancel_temporal_model.temporal_activities,
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            await client.execute_workflow(
                FactoryCancelWorkflow.run,
                args=[response],
                id=f'{FactoryCancelWorkflow.__name__}-{uuid.uuid4()}',
                task_queue=TASK_QUEUE,
            )

    assert factory_cancel_calls == [(CancelTenantDeps(tenant_id='tenant-a'), 'runtime-provider')]
    assert factory_cancelled_responses == [response]


async def test_temporal_model_cancel_suspended_response_accepts_legacy_payload() -> None:
    """An old cancel payload keeps the environment-inference behavior."""
    response = ModelResponse(parts=[TextPart('paused')], state='suspended')
    params = TypeAdapter(_ModelCancelParams).validate_python(
        {'response': response, 'model_id': 'runtime-provider:model'}
    )
    assert params.serialized_run_context is None
    assert params.deps is None

    cancelled: list[ModelResponse] = []

    class RecordingModel(TestModel):
        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append(response)

    environment_model = RecordingModel()

    temporal_model = TemporalModel(
        TestModel(),
        activity_name_prefix='test__legacy_cancel',
        activity_config=BASE_ACTIVITY_CONFIG,
        deps_type=str,
        provider_factory=Mock(),
    )
    with patch('pydantic_ai.durable_exec.temporal._model.models.infer_model', return_value=environment_model) as infer:
        await ActivityEnvironment().run(temporal_model.cancel_suspended_response_activity, params)

    infer.assert_called_once_with('runtime-provider:model')
    assert cancelled == [response]


# Module-level so the `@workflow.defn` below can bind to it (mirrors `simple_temporal_agent`). The
# activity records into this list; since activities always run outside the workflow sandbox in the
# worker process, the workflow can dispatch the teardown while the assertion still observes it here.
model_cancel_calls: list[ModelResponse] = []


class CancelRecordingModel(TestModel):
    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        model_cancel_calls.append(response)


cancel_temporal_model = TemporalModel(
    CancelRecordingModel(),
    activity_name_prefix='cancel_suspended',
    activity_config=BASE_ACTIVITY_CONFIG,
    deps_type=type(None),
)


@workflow.defn
class CancelSuspendedResponseWorkflow:
    @workflow.run
    async def run(self, response: ModelResponse) -> None:
        # In-workflow, `cancel_suspended_response` must dispatch the provider teardown to the
        # `model_cancel_suspended_response` activity rather than make the raw HTTP call in the sandbox.
        await cancel_temporal_model.cancel_suspended_response(response)


replay_legacy_cancel_payload = True


@workflow.defn
class CancelSuspendedResponseReplayWorkflow:
    @workflow.run
    async def run(self, response: ModelResponse) -> None:
        params = _ModelCancelParams(response=response)
        if not replay_legacy_cancel_payload:
            ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), run_id='cancel-replay')
            params.serialized_run_context = TemporalRunContext.serialize_run_context(ctx)
            params.deps = ctx.deps
        await execute_temporal_activity(
            activity=cancel_temporal_model.cancel_suspended_response_activity,
            args=[params],
            **BASE_ACTIVITY_CONFIG,
        )


async def test_temporal_model_cancel_suspended_response_in_workflow(client: Client):
    """Inside a workflow, `cancel_suspended_response` tears the server-side job down via an activity.

    Counterpart to `test_temporal_model_cancel_suspended_response_outside_workflow`: it drives the
    in-workflow override -> `workflow.execute_activity` -> activity-body path end to end, proving the
    wrapped model's cancel actually runs and that the `ModelResponse` argument survives serialization
    across both the workflow and activity boundaries.
    """
    model_cancel_calls.clear()
    response = ModelResponse(parts=[TextPart('paused')], state='suspended')
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CancelSuspendedResponseWorkflow],
        activities=cancel_temporal_model.temporal_activities,
    ):
        await client.execute_workflow(
            CancelSuspendedResponseWorkflow.run,
            args=[response],
            id=CancelSuspendedResponseWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    # The teardown ran in the activity worker against the wrapped model, with the response faithfully
    # round-tripped through both serialization boundaries.
    assert model_cancel_calls == [response]


async def test_temporal_model_cancel_suspended_response_replays_legacy_history(client: Client) -> None:
    """A history recorded with the old cancel payload replays with the one-argument command intact."""
    global replay_legacy_cancel_payload

    replay_legacy_cancel_payload = True
    response = ModelResponse(parts=[TextPart('paused')], state='suspended')
    workflow_id = f'{CancelSuspendedResponseReplayWorkflow.__name__}-{uuid.uuid4()}'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CancelSuspendedResponseReplayWorkflow],
        activities=cancel_temporal_model.temporal_activities,
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        await client.execute_workflow(
            CancelSuspendedResponseReplayWorkflow.run,
            args=[response],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    replay_legacy_cancel_payload = False
    try:
        await Replayer(
            workflows=[CancelSuspendedResponseReplayWorkflow],
            workflow_runner=UnsandboxedWorkflowRunner(),
            data_converter=pydantic_data_converter,
        ).replay_workflow(history)
    finally:
        replay_legacy_cancel_payload = True


async def test_temporal_model_request_stream_outside_workflow():
    """Test that TemporalModel.request_stream() falls back to wrapped model outside a workflow.

    When TemporalModel.request_stream() is called directly (not through TemporalAgent.run())
    and not inside a Temporal workflow, it should delegate to the wrapped model's request_stream method.
    """
    test_model = TestModel(custom_output_text='Direct stream response')

    temporal_model = TemporalModel(
        test_model,
        activity_name_prefix='test__direct_stream',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # Call request_stream() directly - outside a workflow, this should fall back to super().request_stream()
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
    async with temporal_model.request_stream(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            native_tools=[],
            output_mode='text',
            allow_text_output=True,
            output_tools=[],
            output_object=None,
        ),
    ) as stream:
        # Consume the stream
        async for _ in stream:
            pass

        # Get the final response
        response = stream.get()

    # Verify response comes from the wrapped TestModel
    assert any(isinstance(part, TextPart) and part.content == 'Direct stream response' for part in response.parts)


class CustomPydanticPayloadConverter(PydanticPayloadConverter):
    """A custom payload converter that inherits from PydanticPayloadConverter."""

    pass


class CustomPayloadConverter(DefaultPayloadConverter):
    """A custom payload converter that does not inherit from PydanticPayloadConverter."""

    pass


class MockPayloadCodec(PayloadCodec):
    """A mock payload codec for testing (simulates encryption codec)."""

    async def encode(
        self, payloads: Sequence[temporalio.api.common.v1.Payload]
    ) -> list[temporalio.api.common.v1.Payload]:  # pragma: no cover
        return list(payloads)

    async def decode(
        self, payloads: Sequence[temporalio.api.common.v1.Payload]
    ) -> list[temporalio.api.common.v1.Payload]:  # pragma: no cover
        return list(payloads)


async def test_pydantic_ai_payload_converter_builds_type_adapter_once() -> None:
    """Repeated decoding reuses one adapter instead of rebuilding it for every payload."""
    temporal_payload_converter._type_adapter.cache_clear()  # pyright: ignore[reportPrivateUsage]
    converter = DataConverter(payload_converter_class=PydanticAIPayloadConverter)
    payloads = await converter.encode(['result'])

    with patch.object(
        temporal_payload_converter, 'TypeAdapter', wraps=temporal_payload_converter.TypeAdapter
    ) as type_adapter:
        for _ in range(5):
            assert await converter.decode(payloads, [str]) == ['result']

    assert type_adapter.call_count == 1


async def test_pydantic_ai_payload_converter_reuses_more_than_128_type_adapters() -> None:
    """Cyclic access over 129 distinct hints does not rebuild adapters after warmup."""
    temporal_payload_converter._type_adapter.cache_clear()  # pyright: ignore[reportPrivateUsage]
    hints = [type(f'Result{i}', (BaseModel,), {'__annotations__': {'v': int}}) for i in range(129)]

    for hint in hints:
        temporal_payload_converter._type_adapter(hint)  # pyright: ignore[reportPrivateUsage]

    misses_after_warmup = temporal_payload_converter._type_adapter.cache_info().misses  # pyright: ignore[reportPrivateUsage]
    for _ in range(3):
        for hint in hints:
            temporal_payload_converter._type_adapter(hint)  # pyright: ignore[reportPrivateUsage]

    assert temporal_payload_converter._type_adapter.cache_info().misses == misses_after_warmup  # pyright: ignore[reportPrivateUsage]


async def test_pydantic_ai_payload_converter_separates_type_hints() -> None:
    """Different hints use distinct adapters and preserve their respective output types."""
    temporal_payload_converter._type_adapter.cache_clear()  # pyright: ignore[reportPrivateUsage]
    converter = DataConverter(payload_converter_class=PydanticAIPayloadConverter)
    str_payloads = await converter.encode(['1'])
    int_payloads = await converter.encode([1])

    with patch.object(
        temporal_payload_converter, 'TypeAdapter', wraps=temporal_payload_converter.TypeAdapter
    ) as type_adapter:
        assert await converter.decode(str_payloads, [str]) == ['1']
        assert await converter.decode(int_payloads, [int]) == [1]

    assert type_adapter.call_count == 2


async def test_pydantic_ai_payload_converter_accepts_unhashable_type_hint() -> None:
    """Unhashable Pydantic-compatible hints are built uncached rather than rejected."""
    converter = DataConverter(payload_converter_class=PydanticAIPayloadConverter)
    payloads = await converter.encode([1])
    unhashable_hint = Annotated[int, []]

    with patch.object(
        temporal_payload_converter, 'TypeAdapter', wraps=temporal_payload_converter.TypeAdapter
    ) as type_adapter:
        assert await converter.decode(payloads, [unhashable_hint]) == [1]  # pyright: ignore[reportArgumentType]
        assert await converter.decode(payloads, [unhashable_hint]) == [1]  # pyright: ignore[reportArgumentType]

    assert type_adapter.call_count == 2


@pytest.mark.parametrize(
    'value',
    [
        {'metadata': {'reason': 'review'}, 'kind': 'approval_required'},
        {'metadata': {'reason': 'later'}, 'kind': 'call_deferred'},
        {'message': 'retry this', 'kind': 'model_retry'},
        {'result': 'result', 'kind': 'tool_return'},
        {'result': {'kind': 'tool-return', 'value': 1}, 'kind': 'tool_content_result'},
        {'message': 'failed', 'kind': 'tool_failed'},
    ],
)
async def test_pydantic_ai_payload_converter_matches_stock_for_call_tool_result(value: dict[str, Any]) -> None:
    """Every `CallToolResult` variant round-trips identically through stock and memoized converters."""
    stock_payloads = await pydantic_data_converter.encode([value])
    stock_result = await pydantic_data_converter.decode(stock_payloads, [CallToolResult])  # pyright: ignore[reportArgumentType]

    converter = DataConverter(payload_converter_class=PydanticAIPayloadConverter)
    memoized_payloads = await converter.encode([value])
    memoized_result = await converter.decode(memoized_payloads, [CallToolResult])  # pyright: ignore[reportArgumentType]

    assert memoized_payloads == stock_payloads
    assert memoized_result == stock_result


def test_pydantic_ai_plugin_no_converter_uses_memoizing_converter() -> None:
    """When no converter is provided, `PydanticAIPlugin` uses its memoizing converter."""
    plugin = PydanticAIPlugin()
    # Create a minimal config without data_converter
    config: dict[str, Any] = {}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter


def test_pydantic_ai_plugin_passes_pydantic_monty_through_sandbox() -> None:
    runner = SandboxedWorkflowRunner()
    config: dict[str, Any] = {'workflow_runner': runner}

    result = PydanticAIPlugin().configure_worker(config)  # type: ignore[arg-type]

    assert 'workflow_runner' in result
    configured_runner = result['workflow_runner']
    assert isinstance(configured_runner, SandboxedWorkflowRunner)
    assert 'pydantic_monty' in configured_runner.restrictions.passthrough_modules


async def test_pydantic_ai_plugin_runs_workflow_in_sandbox(temporal_target: str) -> None:
    client = await Client.connect(temporal_target)
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[PydanticAIPluginSandboxWorkflow],
        plugins=[PydanticAIPlugin()],
        workflow_runner=SandboxedWorkflowRunner(),
    ):
        result = await client.execute_workflow(
            PydanticAIPluginSandboxWorkflow.run,
            id=f'{PydanticAIPluginSandboxWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )

    assert result == 'sandboxed'


def test_pydantic_ai_plugin_with_stock_pydantic_payload_converter_upgraded() -> None:
    """The exact stock `PydanticPayloadConverter` is upgraded to the memoizing converter."""
    plugin = PydanticAIPlugin()
    codec = MockPayloadCodec()
    converter = DataConverter(payload_converter_class=PydanticPayloadConverter, payload_codec=codec)
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'] is not converter
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter
    assert result['data_converter'].payload_codec is codec
    assert result['data_converter'].failure_converter_class is converter.failure_converter_class


def test_pydantic_ai_plugin_with_custom_pydantic_subclass_unchanged() -> None:
    """When converter uses a subclass of PydanticPayloadConverter, return it unchanged (no warning)."""
    plugin = PydanticAIPlugin()
    converter = DataConverter(payload_converter_class=CustomPydanticPayloadConverter)
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'] is converter
    assert result['data_converter'].payload_converter_class is CustomPydanticPayloadConverter


def test_pydantic_ai_plugin_with_default_payload_converter_replaced() -> None:
    """When converter uses DefaultPayloadConverter, replace payload_converter_class without warning."""
    plugin = PydanticAIPlugin()
    converter = DataConverter(payload_converter_class=DefaultPayloadConverter)
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'] is not converter
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter


def test_pydantic_ai_plugin_preserves_custom_payload_codec() -> None:
    """When converter has a custom payload_codec, preserve it while replacing payload_converter_class."""
    plugin = PydanticAIPlugin()
    codec = MockPayloadCodec()
    converter = DataConverter(
        payload_converter_class=DefaultPayloadConverter,
        payload_codec=codec,
    )
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'] is not converter
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter
    assert result['data_converter'].payload_codec is codec
    assert result['data_converter'].failure_converter_class is converter.failure_converter_class


def test_pydantic_ai_plugin_preserves_external_storage() -> None:
    """A user's Temporal external storage config survives the payload converter swap.

    The Temporal docs point large-payload users at `external_storage`, so this has to keep working.
    """

    class MockStorageDriver(StorageDriver):
        def name(self) -> str:
            return 'mock'

        async def store(self, context: Any, payloads: Any) -> Any:
            raise NotImplementedError

        async def retrieve(self, context: Any, claims: Any) -> Any:
            raise NotImplementedError

    external_storage = ExternalStorage(drivers=[MockStorageDriver()])
    plugin = PydanticAIPlugin()
    converter = DataConverter(
        payload_converter_class=DefaultPayloadConverter,
        external_storage=external_storage,
    )
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter
    assert result['data_converter'].external_storage is external_storage


def test_pydantic_ai_plugin_with_non_pydantic_converter_warns() -> None:
    """When converter uses a non-Pydantic payload converter, warn and replace."""
    plugin = PydanticAIPlugin()
    converter = DataConverter(payload_converter_class=CustomPayloadConverter)
    config: dict[str, Any] = {'data_converter': converter}
    with pytest.warns(
        UserWarning,
        match='A non-Pydantic Temporal payload converter was used which has been replaced with '
        '`PydanticAIPayloadConverter`',
    ):
        result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter


def test_pydantic_ai_plugin_with_non_pydantic_converter_preserves_codec() -> None:
    """When converter uses a non-Pydantic payload converter with custom codec, warn but preserve codec."""
    plugin = PydanticAIPlugin()
    codec = MockPayloadCodec()
    converter = DataConverter(
        payload_converter_class=CustomPayloadConverter,
        payload_codec=codec,
    )
    config: dict[str, Any] = {'data_converter': converter}
    with pytest.warns(UserWarning):
        result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter
    assert result['data_converter'].payload_codec is codec


def test_temporal_model_profile_with_no_provider_prefix() -> None:
    """Test TemporalModel uses DEFAULT_PROFILE when model string has no inferable provider."""

    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__no_provider_prefix',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # A model string without a provider prefix that can't be inferred returns DEFAULT_PROFILE
    with temporal_model.using_model('some-random-model'):
        assert temporal_model.profile is DEFAULT_PROFILE


def test_temporal_model_profile_with_unknown_provider() -> None:
    """Test TemporalModel uses DEFAULT_PROFILE when provider is unknown."""

    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__unknown_provider',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # An unknown provider should return DEFAULT_PROFILE
    with temporal_model.using_model('unknown-provider:some-model'):
        assert temporal_model.profile is DEFAULT_PROFILE


@pytest.mark.parametrize(
    'model_id',
    [
        'openai:gpt-5',
        'gateway/openai:gpt-5',
    ],
)
def test_temporal_model_prepare_request_with_unregistered_model_string(model_id: str) -> None:
    """Test prepare_request uses inferred profile for unregistered model strings.

    Verifies that the OpenAI json_schema_transformer is applied to function tool
    schemas (adding additionalProperties: false) when using an OpenAI model string,
    both directly and via gateway/.
    """
    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__prepare_request_unregistered',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    tool_def = ToolDefinition(
        name='my_tool',
        description='A test tool',
        parameters_json_schema={
            'type': 'object',
            'properties': {'x': {'type': 'integer'}},
            'required': ['x'],
        },
    )

    model_request_params = ModelRequestParameters(
        function_tools=[tool_def],
        native_tools=[],
        output_mode='text',
        allow_text_output=True,
        output_tools=[],
        output_object=None,
    )

    # With an unregistered model string, prepare_request should use the inferred
    # profile's json_schema_transformer (OpenAI adds additionalProperties: false)
    with temporal_model.using_model(model_id):
        _, params = temporal_model.prepare_request(None, model_request_params)
        assert params.output_mode == 'text'
        assert len(params.function_tools) == 1
        assert params.function_tools[0].parameters_json_schema['additionalProperties'] is False


def test_temporal_model_prepare_messages_with_unregistered_model_string() -> None:
    """`prepare_messages` defers preparation for unregistered model strings.

    The temporal wrapper has no concrete `Model` instance to delegate to in the workflow,
    so the activity performs the single authoritative pass after resolving it.
    """
    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__prepare_messages_unregistered',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hi')])]
    with temporal_model.using_model('openai:gpt-5'):
        prepared = temporal_model.prepare_messages(messages)
    assert prepared == messages


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_temporal_model_runtime_provider_prepares_messages_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistered model string is prepared only after its concrete profile is known."""

    def provider_factory(_ctx: RunContext[object], _provider_name: str) -> AnthropicProvider:
        return AnthropicProvider(api_key='test-api-key')

    temporal_model = TemporalModel(
        TestModel(),
        activity_name_prefix='test__runtime_provider_prepare_messages_once',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=object,
        provider_factory=provider_factory,
    )
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart('leading'), UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('answer')]),
        ModelRequest(parts=[SystemPromptPart('mid'), UserPromptPart('second')]),
    ]

    def infer_unsupported_profile(_model_id: str) -> ModelProfile:
        return DEFAULT_PROFILE

    monkeypatch.setattr('pydantic_ai.durable_exec.temporal._model.infer_model_profile', infer_unsupported_profile)
    with temporal_model.using_model('anthropic:claude-opus-5'):
        prepared_messages = temporal_model.prepare_messages(messages)

    received_messages: list[list[ModelMessage]] = []

    async def request(
        _model: AnthropicModel,
        activity_messages: list[ModelMessage],
        _model_settings: ModelSettings | None,
        _model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        received_messages.append(activity_messages)
        return ModelResponse(parts=[TextPart('done')])

    monkeypatch.setattr(AnthropicModel, 'request', request)
    deps = object()
    ctx = RunContext[object](deps=deps, model=TestModel(), usage=RunUsage(), run_id='runtime-provider')
    params = _RequestParams(
        messages=prepared_messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
        model_id='anthropic:claude-opus-5',
    )
    await ActivityEnvironment().run(temporal_model.request_activity, params, deps)

    assert received_messages == [messages]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_temporal_model_runtime_provider_reprepares_messages(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    """The activity applies the concrete transport profile before sending serialized history."""
    foundry_client = anthropic.AsyncAnthropicFoundry(
        resource='test-resource',
        api_key='test-api-key',
    )

    def provider_factory(_ctx: RunContext[object], _provider_name: str) -> AnthropicProvider:
        return AnthropicProvider(anthropic_client=foundry_client)

    async def event_stream_handler(
        _ctx: RunContext[object], _streamed_response: AsyncIterable[AgentStreamEvent]
    ) -> None:
        pass

    temporal_model = TemporalModel(
        TestModel(),
        activity_name_prefix=f'test__runtime_provider_reprepare_{stream}',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=object,
        provider_factory=provider_factory,
        event_stream_handler=event_stream_handler,
    )
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart('leading'), UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('answer')]),
        ModelRequest(parts=[SystemPromptPart('mid'), UserPromptPart('second')]),
    ]
    with temporal_model.using_model('anthropic:claude-opus-5'):
        prepared_messages = temporal_model.prepare_messages(messages)
    assert prepared_messages == messages

    rendered_requests: list[dict[str, Any]] = []

    async def render(
        model: AnthropicModel,
        activity_messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        assert model_settings is None
        anthropic_settings: AnthropicModelSettings = {}
        system_prompt, anthropic_messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
            activity_messages,
            model_request_parameters,
            anthropic_settings,
        )
        rendered_requests.append({'system': system_prompt, 'messages': anthropic_messages})
        return ModelResponse(parts=[TextPart('done')])

    if stream:

        @asynccontextmanager
        async def request_stream(
            model: AnthropicModel,
            activity_messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[object] | None = None,
        ) -> AsyncGenerator[CompletedStreamedResponse]:
            del run_context
            response = await render(model, activity_messages, model_settings, model_request_parameters)
            yield CompletedStreamedResponse(response, model_request_parameters=model_request_parameters)

        monkeypatch.setattr(AnthropicModel, 'request_stream', request_stream)
    else:
        monkeypatch.setattr(AnthropicModel, 'request', render)

    deps = object()
    ctx = RunContext[object](deps=deps, model=TestModel(), usage=RunUsage(), run_id='runtime-provider')
    params = _RequestParams(
        messages=prepared_messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
        model_id='anthropic:claude-opus-5',
    )
    if stream:
        await ActivityEnvironment().run(
            temporal_model.request_stream_activity,
            params,
            deps,  # pyright: ignore[reportArgumentType]
        )
    else:
        await ActivityEnvironment().run(temporal_model.request_activity, params, deps)

    assert rendered_requests == snapshot(
        [
            {
                'system': 'leading',
                'messages': [
                    {'role': 'user', 'content': [{'text': 'first', 'type': 'text'}]},
                    {'role': 'assistant', 'content': [{'text': 'answer', 'type': 'text'}]},
                    {
                        'role': 'user',
                        'content': [
                            {'text': '<system>mid</system>', 'type': 'text'},
                            {'text': 'second', 'type': 'text'},
                        ],
                    },
                ],
            }
        ]
    )


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_temporal_model_runtime_provider_preserves_unmodified_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The activity forwards history unchanged when the concrete model has nothing to rewrite."""

    def provider_factory(_ctx: RunContext[object], _provider_name: str) -> AnthropicProvider:
        return AnthropicProvider(api_key='test-api-key')

    temporal_model = TemporalModel(
        TestModel(),
        activity_name_prefix='test__runtime_provider_preserve_messages',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=object,
        provider_factory=provider_factory,
    )
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('hello')])]
    received_messages: list[list[ModelMessage]] = []

    async def request(
        _model: AnthropicModel,
        activity_messages: list[ModelMessage],
        _model_settings: ModelSettings | None,
        _model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        received_messages.append(activity_messages)
        return ModelResponse(parts=[TextPart('done')])

    monkeypatch.setattr(AnthropicModel, 'request', request)

    deps = object()
    ctx = RunContext[object](deps=deps, model=TestModel(), usage=RunUsage(), run_id='runtime-provider')
    params = _RequestParams(
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
        model_id='anthropic:claude-opus-5',
    )
    await ActivityEnvironment().run(temporal_model.request_activity, params, deps)
    assert received_messages
    assert received_messages[0] is messages


def test_temporal_model_customize_request_parameters_with_registered_model() -> None:
    """Test customize_request_parameters delegates to the currently active registered model."""

    class _CustomizingTestModel(TestModel):
        def customize_request_parameters(
            self, model_request_parameters: ModelRequestParameters
        ) -> ModelRequestParameters:
            return ModelRequestParameters(output_mode='tool', allow_text_output=False)

    default_model = TestModel(custom_output_text='default')
    alternate_model = _CustomizingTestModel(custom_output_text='alternate')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__customize_registered',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
        models={'alternate': alternate_model},
    )

    with temporal_model.using_model('alternate'):
        customized = temporal_model.customize_request_parameters(ModelRequestParameters())

    assert customized.output_mode == 'tool'
    assert customized.allow_text_output is False


# Tests for BinaryContent and DocumentUrl serialization in Temporal
# This is a regression test for #3702 (BinaryContent) and verifies that FileUrl
# instances (like DocumentUrl) with explicit media_type are properly preserved.


multimodal_content_agent = Agent(TestModel(), name='multimodal_content_agent')


@multimodal_content_agent.tool
def get_multimodal_content(ctx: RunContext) -> list[str | MultiModalContent]:
    """Return a list with text, BinaryContent, and DocumentUrl."""
    return [
        'test',
        BinaryImage(data=b'\x89PNG', media_type='image/png'),
        # URL doesn't hint at media type, so media_type must be specified explicitly
        DocumentUrl(url='https://example.com/doc/12345', media_type='application/pdf'),
    ]


multimodal_content_temporal_agent = TemporalAgent(multimodal_content_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class MultiModalContentWorkflow:
    @workflow.run
    async def run(self, prompt: list[UserContent]) -> list[ModelMessage]:
        result = await multimodal_content_temporal_agent.run(prompt)
        return result.all_messages()


async def test_multimodal_content_serialization_in_workflow(client: Client):
    """Test that BinaryContent and DocumentUrl survive Temporal serialization.

    This tests both:
    1. Passing BinaryContent and DocumentUrl as input to agent.run (workflow→activity)
    2. Returning BinaryContent and DocumentUrl from a tool (activity→workflow)

    BinaryContent is serialized with base64 encoding. DocumentUrl requires explicit
    media_type since it cannot be inferred from the URL.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModalContentWorkflow],
        plugins=[AgentPlugin(multimodal_content_temporal_agent)],
    ):
        # Pass both BinaryContent and DocumentUrl as input
        prompt: list[str | MultiModalContent] = [
            'Process these files and call the tool',
            BinaryImage(data=b'\x89PNG', media_type='image/png'),
            DocumentUrl(url='https://example.com/doc/12345', media_type='application/pdf'),
        ]
        messages = await client.execute_workflow(
            MultiModalContentWorkflow.run,
            args=[prompt],
            id='test_multimodal_content_serialization',
            task_queue=TASK_QUEUE,
        )
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                'Process these files and call the tool',
                                BinaryImage(data=b'\x89PNG', media_type='image/png', identifier='4effda'),
                                DocumentUrl(
                                    url='https://example.com/doc/12345',
                                    _media_type='application/pdf',
                                    _identifier='eb8998',
                                ),
                            ],
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_multimodal_content',
                            args={},
                            tool_call_id='pyd_ai_tool_call_id__get_multimodal_content',
                        )
                    ],
                    usage=RequestUsage(input_tokens=61, output_tokens=2),
                    model_name='test',
                    timestamp=IsDatetime(),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_multimodal_content',
                            content=[
                                'test',
                                BinaryImage(data=b'\x89PNG', media_type='image/png', identifier='4effda'),
                                DocumentUrl(
                                    url='https://example.com/doc/12345',
                                    _media_type='application/pdf',
                                    _identifier='eb8998',
                                ),
                            ],
                            tool_call_id='pyd_ai_tool_call_id__get_multimodal_content',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='{"get_multimodal_content":["test",{"data":"iVBORw==","media_type":"image/png","vendor_metadata":null,"kind":"binary","identifier":"4effda"},{"url":"https://example.com/doc/12345","force_download":false,"vendor_metadata":null,"kind":"document-url","media_type":"application/pdf","identifier":"eb8998"}]}'
                        )
                    ],
                    usage=RequestUsage(input_tokens=62, output_tokens=34),
                    model_name='test',
                    timestamp=IsDatetime(),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

        # Explicitly verify that media_type is preserved through serialization for both
        # BinaryContent and DocumentUrl. This is important because _media_type has compare=False
        # on DocumentUrl, so the snapshot comparison doesn't actually verify it. The media_type
        # cannot be inferred from the URL, so if serialization loses it, accessing media_type
        # would raise an error.
        media_types: list[tuple[str, str]] = []
        for message in messages:
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    for content in part.content:
                        if isinstance(content, (BinaryContent, DocumentUrl)):
                            media_types.append((type(content).__name__, content.media_type))
                elif isinstance(part, ToolReturnPart):
                    for content in part.content_items():
                        if isinstance(content, (BinaryContent, DocumentUrl)):
                            media_types.append((type(content).__name__, content.media_type))
        # Should have 4 items: 2 from user input, 2 from tool return.
        # The image `BinaryContent` round-trips as `BinaryImage`: narrowing is applied during
        # `MultiModalContent` validation, so it now survives the Temporal serialization boundary too.
        assert media_types == [
            ('BinaryImage', 'image/png'),
            ('DocumentUrl', 'application/pdf'),
            ('BinaryImage', 'image/png'),
            ('DocumentUrl', 'application/pdf'),
        ]


async def test_text_content_serialization_in_workflow(client: Client):
    """Test that TextContent is properly serialized in Temporal."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModalContentWorkflow],
        plugins=[AgentPlugin(multimodal_content_temporal_agent)],
    ):
        prompt = [
            'This is a text content test',
            TextContent(content='This should be preserved as TextContent', metadata={'preserved': True}),
        ]
        messages = await client.execute_workflow(
            MultiModalContentWorkflow.run,
            args=[prompt],
            id='test_text_content_serialization',
            task_queue=TASK_QUEUE,
        )
        assert messages[0] == snapshot(
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'This is a text content test',
                            TextContent(
                                content='This should be preserved as TextContent', metadata={'preserved': True}
                            ),
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        )


_pydantic_ai_agents_durable = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)

_pydantic_ai_agents_agent = Agent(
    _durability_fn_model,
    name='pydantic_ai_agents_attr_test',
    capabilities=[_pydantic_ai_agents_durable],
)


@workflow.defn
class _BareAgentWorkflowViaAttribute:
    __pydantic_ai_agents__ = [_pydantic_ai_agents_agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _pydantic_ai_agents_agent.run(prompt)
        return result.output


async def test_pydantic_ai_plugin_discovers_bare_agent_with_durability(client: Client):
    """`PydanticAIPlugin` registers activities from a bare `AbstractAgent` listed in `__pydantic_ai_agents__`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[_BareAgentWorkflowViaAttribute],
    ):
        output = await client.execute_workflow(
            _BareAgentWorkflowViaAttribute.run,
            args=['Discovered'],
            id=_BareAgentWorkflowViaAttribute.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == 'Echo: Discovered'


_missing_cap_agent = Agent(_durability_fn_model, name='no_cap_in_attr')


@workflow.defn
class _MissingCapWorkflow:
    __pydantic_ai_agents__ = [_missing_cap_agent]

    # `configure_worker` rejects before this can execute.
    @workflow.run
    async def run(self, prompt: str) -> str:  # pragma: no cover
        result = await _missing_cap_agent.run(prompt)
        return result.output


async def test_pydantic_ai_plugin_rejects_bare_agent_without_durability(client: Client):
    """`PydanticAIPlugin` raises a clear error when an agent in `__pydantic_ai_agents__` lacks `TemporalDurability`."""
    with pytest.raises(UserError, match='no `TemporalDurability` capability'):
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[_MissingCapWorkflow],
        ):
            # The error is raised before reaching here.
            pass  # pragma: no cover
