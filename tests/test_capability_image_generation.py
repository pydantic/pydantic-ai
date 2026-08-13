from __future__ import annotations

import inspect
import re
import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

import pydantic_ai.images as images_module
from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import (
    ImageGeneration,
    Instrumentation,
)
from pydantic_ai.exceptions import (
    ContentFilterError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.images import (
    ImageGenerationInput,
    ImageGenerationResult,
    ImageGenerationSettings,
    ImageGenerator,
    TestImageGenerationModel,
)
from pydantic_ai.messages import (
    BinaryImage,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.native_tools import (
    ImageGenerationTool,
)
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsInstance, IsStr, iter_message_parts, try_import

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire


class TestImageGenerationCapability:
    def test_image_gen_init_params_cover_builtin_tool_and_direct_geometry(self):
        """ImageGeneration adds direct geometry without dropping existing native-tool fields."""
        import dataclasses

        # partial_images is excluded — not useful for subagent fallback (no streaming).
        # optional is excluded — applies to wire-side dropping, not local-fallback config.
        builtin_fields = {
            f.name
            for f in dataclasses.fields(ImageGenerationTool)
            if f.name not in ('kind', 'optional', 'partial_images')
        }
        builtin_fields.remove('model')
        builtin_fields.add('image_model')
        # Subtract framework-inherited kw-only params from `AbstractCapability`
        # (forwarded so `dataclasses.replace` round-trips through the custom `__init__`).
        init_params = set(inspect.signature(ImageGeneration.__init__).parameters.keys()) - {
            'self',
            'native',
            'local',
            'fallback_model',
            'id',
            'defer_loading',
            'description',
        }
        assert init_params == builtin_fields | {'dimensions'}

        # The spec-only constructor deliberately narrows runtime-only values, but must keep the
        # same parameter names so YAML/JSON configuration cannot drift from the Python API.
        spec_params = set(inspect.signature(ImageGeneration.from_spec).parameters)
        runtime_params = set(inspect.signature(ImageGeneration.__init__).parameters) - {'self'}
        assert spec_params == runtime_params

    def test_image_generation_default(self):
        """ImageGeneration() provides only builtin, no local fallback."""
        cap = ImageGeneration()
        builtins = cap.get_native_tools()
        assert len(builtins) == 1
        assert isinstance(builtins[0], ImageGenerationTool)
        # No default local
        assert cap.local is None
        assert cap.get_toolset() is None

    def test_image_generation_with_custom_local(self):
        """ImageGeneration(local=custom) → provides custom local fallback."""
        from pydantic_ai.tools import Tool

        def my_gen(prompt: str) -> str:
            return 'image_url'  # pragma: no cover

        cap = ImageGeneration(local=my_gen)
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    def test_image_generation_accepts_direct_image_model(self):
        """A direct image model is normalized to the local capability tool."""
        from pydantic_ai.tools import Tool

        cap = ImageGeneration(local=TestImageGenerationModel())

        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    def test_image_generation_accepts_direct_model_name(self):
        """A direct image model name is resolved as a local capability strategy."""
        from pydantic_ai.tools import Tool

        cap = ImageGeneration(native=False, local='openai:gpt-image-1.5')

        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    async def test_image_generation_direct_fallback(self, allow_model_requests: None):
        """The direct fallback applies portable settings and warns for native-only settings."""
        image_model = TestImageGenerationModel(
            settings={
                'extra_headers': {'x-test': 'preserved'},
            }
        )
        generator = ImageGenerator(image_model)

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args={'prompt': 'tiny robot'})])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        with pytest.warns(UserWarning, match='ignored native-tool setting'):
            capability = ImageGeneration(
                native=False,
                local=generator,
                background='opaque',
                input_fidelity='high',
                moderation='low',
                output_compression=80,
                output_format='jpeg',
                quality='low',
                size='1024x1024',
                dimensions=(1024, 1024),
            )
        agent = Agent(
            outer_model,
            capabilities=[capability],
        )

        result = await agent.run('Generate an image')

        assert result.output == 'done'
        assert image_model.last_settings == {
            'dimensions': (1024, 1024),
            'extra_headers': {'x-test': 'preserved'},
        }
        tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
        assert len(tool_returns) == 1
        assert isinstance(tool_returns[0].content, BinaryImage)
        assert tool_returns[0].content.media_type == 'image/png'

        aspect_ratio_model = TestImageGenerationModel()
        aspect_ratio_agent = Agent(
            outer_model,
            capabilities=[ImageGeneration(native=False, local=aspect_ratio_model, aspect_ratio='1:1')],
        )
        await aspect_ratio_agent.run('Generate an image')
        assert aspect_ratio_model.last_settings == {'aspect_ratio': '1:1'}

    async def test_image_generation_spec_string_local_resolves_deferred_model_during_run(
        self, allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
    ):
        """A spec-loaded string `local` stays unresolved until the tool generates, then applies the JSON dimensions."""
        image_model = TestImageGenerationModel()
        inferred_models: list[object] = []

        def infer_model(model: object) -> TestImageGenerationModel:
            inferred_models.append(model)
            return image_model

        monkeypatch.setattr(images_module, 'infer_image_generation_model', infer_model)

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args={'prompt': 'tiny robot'})])

        agent = Agent.from_spec(
            {
                'capabilities': [
                    {
                        'ImageGeneration': {
                            'native': False,
                            'local': 'openai:gpt-image-1.5',
                            'dimensions': [1280, 720],
                        }
                    }
                ]
            },
            model=FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset())),
        )
        assert inferred_models == snapshot([])

        result = await agent.run('Generate an image')

        assert inferred_models == snapshot(['openai:gpt-image-1.5'])
        assert image_model.last_settings == snapshot({'dimensions': (1280, 720)})
        tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
        assert tool_returns == snapshot(
            [
                ToolReturnPart(
                    tool_name='generate_image',
                    content=IsInstance(BinaryImage),
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                )
            ]
        )

    @pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
    async def test_image_generation_direct_fallback_instrumentation_omits_binary_tool_result(
        self, allow_model_requests: None, capfire: CaptureLogfire
    ):
        """Tool span redaction does not change the binary result delivered to the outer model.

        Scoped to the tool span's own result attribute. `include_binary_content=False` is not a
        library-wide redaction guarantee: the run span's `final_result` and the serialized message
        history carry binary content regardless.
        """

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args={'prompt': 'tiny robot'})])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(
            outer_model,
            capabilities=[
                ImageGeneration(native=False, local=TestImageGenerationModel()),
                Instrumentation(settings=InstrumentationSettings(include_binary_content=False)),
            ],
        )

        result = await agent.run('Generate an image')

        tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
        assert len(tool_returns) == 1
        assert isinstance(tool_returns[0].content, BinaryImage)

        spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
        tool_span = next(span for span in spans if span['name'] == 'execute_tool generate_image')
        tool_result = tool_span['attributes']['gen_ai.tool.call.result']
        assert tool_result['media_type'] == 'image/png'
        assert 'data' not in tool_result

    async def test_image_generation_direct_fallback_rejects_edit_action(self, allow_model_requests: None):
        """The prompt-only capability tool cannot silently turn a requested edit into generation."""

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args={'prompt': 'tiny robot'})])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(
            outer_model,
            capabilities=[ImageGeneration(native=False, local=TestImageGenerationModel(), action='edit')],
        )

        with pytest.raises(UserError, match='cannot honor `action="edit"`'):
            await agent.run('Edit an image')

    async def test_image_generation_direct_fallback_warns_for_image_model(self, allow_model_requests: None):
        """The direct model is selected by `local`, so the legacy model override is explicit about being ignored."""

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args={'prompt': 'tiny robot'})])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(
            outer_model,
            capabilities=[
                ImageGeneration(
                    native=False,
                    local=TestImageGenerationModel(),
                    image_model='gpt-image-1',
                )
            ],
        )

        with pytest.warns(UserWarning, match=r'ignored `image_model`'):
            result = await agent.run('Generate an image')

        assert result.output == 'done'

    def test_image_generation_rejects_local_true(self):
        """Unlike named image models, `local=True` is not an image generation strategy."""
        with pytest.raises(UserError, match=r'`local=True` is not supported'):
            ImageGeneration(local=True)  # type: ignore[arg-type]

    async def test_image_generation_direct_fallback_rejects_multiple_images(self, allow_model_requests: None):
        """The single-image capability contract never silently discards direct API outputs."""

        class MultipleImageGenerationModel(TestImageGenerationModel):
            async def generate(
                self,
                prompt: str,
                *,
                images: Sequence[ImageGenerationInput] | None = None,
                settings: ImageGenerationSettings | None = None,
            ) -> ImageGenerationResult:
                result = await super().generate(prompt, images=images, settings=settings)
                return replace(result, images=[*result.images, *result.images])

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args={'prompt': 'tiny robot'})])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(
            outer_model,
            capabilities=[ImageGeneration(native=False, local=MultipleImageGenerationModel())],
            retries=0,
        )

        with pytest.raises(UnexpectedModelBehavior, match='returned 2 images; expected exactly one'):
            await agent.run('Generate an image')

    async def test_image_generation_prefers_native_over_direct_fallback(self, allow_model_requests: None):
        """The existing native-or-local routing suppresses the direct fallback when native is supported."""
        image_model = TestImageGenerationModel()

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.function_tools == []
            return ModelResponse(parts=[TextPart(content='native path')])

        outer_model = FunctionModel(
            outer_model_fn,
            profile=ModelProfile(supported_native_tools=frozenset({ImageGenerationTool})),
        )
        agent = Agent(outer_model, capabilities=[ImageGeneration(local=ImageGenerator(image_model))])

        result = await agent.run('Generate an image')

        assert result.output == 'native path'
        assert image_model.last_settings is None

    async def test_image_generation_warns_when_native_supersedes_direct_only_geometry(self, allow_model_requests: None):
        """Dropping the direct generator drops `dimensions` with it, which only the request knows.

        Whether the drop happens depends on the model's `supported_native_tools`, so the capability
        can't tell at construction time; the warning has to come from the per-request prepare
        function, and the model that has no native image generation must stay silent.
        """
        image_model = TestImageGenerationModel()

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.function_tools == []
            return ModelResponse(parts=[TextPart(content='native path')])

        outer_model = FunctionModel(
            outer_model_fn,
            profile=ModelProfile(supported_native_tools=frozenset({ImageGenerationTool})),
        )
        agent = Agent(
            outer_model,
            capabilities=[ImageGeneration(local=ImageGenerator(image_model), dimensions=(1280, 720))],
        )

        with pytest.warns(UserWarning, match=r'direct-only setting\(s\) go unapplied: dimensions'):
            result = await agent.run('Generate an image')

        assert result.output == 'native path'
        assert image_model.last_settings is None

    async def test_image_generation_callable_native_does_not_warn_about_direct_only_geometry(
        self, allow_model_requests: None
    ):
        """A callable `native` that yields no tool leaves the generator in place, so nothing is unapplied.

        The framework resolves the callable per request, so the capability cannot anticipate the
        result without calling it a second time; warning on the mere possibility would be wrong here.
        """
        image_model = TestImageGenerationModel()

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert [tool.name for tool in info.function_tools] == ['generate_image']
            return ModelResponse(parts=[TextPart(content='local path')])

        outer_model = FunctionModel(
            outer_model_fn,
            profile=ModelProfile(supported_native_tools=frozenset({ImageGenerationTool})),
        )
        agent = Agent(
            outer_model,
            capabilities=[
                ImageGeneration(native=lambda ctx: None, local=ImageGenerator(image_model), dimensions=(1280, 720))
            ],
        )

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = await agent.run('Generate an image')

        assert result.output == 'local path'

    async def test_image_generation_applies_direct_only_geometry_without_native_support(
        self, allow_model_requests: None
    ):
        """The same configuration on a model without native image generation applies and stays silent."""
        image_model = TestImageGenerationModel()

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args={'prompt': 'tiny robot'})])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(
            outer_model,
            capabilities=[ImageGeneration(local=ImageGenerator(image_model), dimensions=(1280, 720))],
        )

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = await agent.run('Generate an image')

        assert result.output == 'done'
        assert image_model.last_settings == {'dimensions': (1280, 720)}

    async def test_image_generation_warns_when_native_supersedes_non_native_aspect_ratio(
        self, allow_model_requests: None
    ):
        """`'2:1'` is outside the native tool's vocabulary, so only the dropped generator could apply it."""
        image_model = TestImageGenerationModel()

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='native path')])

        outer_model = FunctionModel(
            outer_model_fn,
            profile=ModelProfile(supported_native_tools=frozenset({ImageGenerationTool})),
        )
        agent = Agent(
            outer_model,
            capabilities=[ImageGeneration(local=ImageGenerator(image_model), aspect_ratio='2:1')],
        )

        with pytest.warns(UserWarning, match=r'direct-only setting\(s\) go unapplied: aspect_ratio'):
            result = await agent.run('Generate an image')

        assert result.output == 'native path'
        assert image_model.last_settings is None

    async def test_image_generation_native_vocabulary_aspect_ratio_does_not_warn(self, allow_model_requests: None):
        """`'16:9'` is forwarded to the native tool, so the native path applies it rather than dropping it."""
        image_model = TestImageGenerationModel()

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='native path')])

        outer_model = FunctionModel(
            outer_model_fn,
            profile=ModelProfile(supported_native_tools=frozenset({ImageGenerationTool})),
        )
        capability = ImageGeneration(local=ImageGenerator(image_model), aspect_ratio='16:9')
        agent = Agent(outer_model, capabilities=[capability])

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = await agent.run('Generate an image')

        assert result.output == 'native path'
        native_tool = capability.get_native_tools()[0]
        assert isinstance(native_tool, ImageGenerationTool)
        assert native_tool.aspect_ratio == '16:9'

    def test_image_generation_with_fallback_model(self):
        """ImageGeneration(fallback_model=...) creates a local fallback tool."""
        from pydantic_ai.tools import Tool

        cap = ImageGeneration(fallback_model='openai-responses:gpt-5.4')
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None
        builtins = cap.get_native_tools()
        assert len(builtins) == 1
        assert isinstance(builtins[0], ImageGenerationTool)

    def test_image_generation_fallback_model_warns_for_direct_only_geometry(self):
        with pytest.warns(UserWarning, match='ignored direct-only setting.*dimensions'):
            cap = ImageGeneration(
                fallback_model='openai-responses:gpt-5.4',
                dimensions=(2048, 1152),
            )

        assert cap.get_toolset() is not None

    def test_image_generation_forwards_config_to_builtin(self):
        """ImageGeneration config fields are forwarded to the ImageGenerationTool builtin."""
        cap = ImageGeneration(
            action='generate',
            background='opaque',
            input_fidelity='high',
            moderation='low',
            image_model='gpt-image-2',
            output_compression=80,
            output_format='jpeg',
            quality='high',
            size='1024x1024',
            aspect_ratio='16:9',
        )
        builtins = cap.get_native_tools()
        assert len(builtins) == 1
        tool = builtins[0]
        assert isinstance(tool, ImageGenerationTool)
        assert tool.action == 'generate'
        assert tool.background == 'opaque'
        assert tool.input_fidelity == 'high'
        assert tool.moderation == 'low'
        assert tool.model == 'gpt-image-2'
        assert tool.output_compression == 80
        assert tool.output_format == 'jpeg'
        assert tool.quality == 'high'
        assert tool.size == '1024x1024'
        assert tool.aspect_ratio == '16:9'

    @pytest.mark.parametrize(
        ('kwargs', 'ignored'),
        [
            ({'dimensions': (2048, 1152)}, 'dimensions'),
            ({'size': '2048x1024', 'aspect_ratio': '2:1'}, 'size, aspect_ratio'),
        ],
    )
    def test_image_generation_legacy_ignores_direct_only_geometry(self, kwargs: dict[str, Any], ignored: str):
        with pytest.warns(UserWarning, match=f'ignored direct-only setting.*{ignored}'):
            cap = ImageGeneration(image_model='gpt-image-2', **kwargs)
            builtins = cap.get_native_tools()
        assert len(builtins) == 1
        tool = builtins[0]
        assert isinstance(tool, ImageGenerationTool)
        assert tool.size is None
        assert tool.aspect_ratio is None

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'dimensions': (1280, 720)},
            {'aspect_ratio': '19.5:9'},
        ],
    )
    def test_image_generation_direct_generator_suppresses_ignored_geometry_warning(self, kwargs: dict[str, Any]):
        """Geometry the native tool can't express is not "ignored" when a direct generator applies it.

        `native=True` is the default, so the native tool is still built and still drops these values —
        but warning about it is wrong when `local` is a direct generator that honors them. Both cases
        use a value outside the native vocabulary so they take the non-native branch.

        `size` is deliberately absent: `_direct_image_settings` forwards only `dimensions` and
        `aspect_ratio`, so a non-native `size` really is dropped by both paths and still warns.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            cap = ImageGeneration(local=ImageGenerator(TestImageGenerationModel()), **kwargs)
            builtins = cap.get_native_tools()

        assert len(builtins) == 1
        tool = builtins[0]
        assert isinstance(tool, ImageGenerationTool)
        assert tool.size is None
        assert tool.aspect_ratio is None

    def test_image_generation_fallback_merges_custom_native_with_overrides(self):
        """Custom native tool settings are merged with capability-level overrides for the fallback."""
        from pydantic_ai.tools import Tool

        custom_native = ImageGenerationTool(quality='high', size='1024x1024')
        cap = ImageGeneration(
            native=custom_native,
            fallback_model='openai-responses:gpt-5.4',
            output_format='jpeg',  # capability-level override
        )
        # The local fallback should exist and contain the merged config
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    def test_image_generation_callable_native_with_fallback(self):
        """When native is a callable, the fallback local tool still gets created."""
        from pydantic_ai.tools import Tool

        cap = ImageGeneration(
            native=lambda ctx: ImageGenerationTool(quality='high'),
            fallback_model='openai-responses:gpt-5.4',
        )
        # Callable native can't be resolved at init time, but local fallback is still created
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    def test_image_generation_fallback_model_and_local_conflict(self):
        """ImageGeneration(fallback_model=..., local=func) raises UserError."""

        def my_gen(prompt: str) -> str:
            return 'image_url'  # pragma: no cover

        with pytest.raises(UserError, match='cannot specify both `fallback_model` and `local`'):
            ImageGeneration(fallback_model='openai-responses:gpt-5.4', local=my_gen)

    def test_image_generation_fallback_model_with_local_false(self):
        """ImageGeneration(fallback_model=..., local=False) raises UserError."""
        with pytest.raises(UserError, match='cannot specify both `fallback_model` and `local`'):
            ImageGeneration(fallback_model='openai-responses:gpt-5.4', local=False)

    async def test_image_generation_callable_fallback_model(self, allow_model_requests: None):
        """ImageGeneration with async callable fallback_model resolves the model per-run."""
        from pydantic_ai.messages import BinaryImage

        image_data = b'\x89PNG\r\n\x1a\n'  # minimal PNG header

        def inner_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[FilePart(content=BinaryImage(data=image_data, media_type='image/png'))])

        inner_model = FunctionModel(inner_model_fn, profile=ModelProfile(supports_image_output=True))

        async def model_factory(ctx: RunContext) -> FunctionModel:
            return inner_model

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args='{"prompt": "test"}')])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(outer_model, capabilities=[ImageGeneration(fallback_model=model_factory)])
        result = await agent.run('Generate a test image')
        assert result.output == 'done'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Generate a test image', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='generate_image',
                            args='{"prompt": "test"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=54, output_tokens=5),
                    model_name='function:outer_model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='generate_image',
                            content=BinaryImage(data=b'\x89PNG\r\n\x1a\n', media_type='image/png'),
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='done')],
                    usage=RequestUsage(input_tokens=54, output_tokens=6),
                    model_name='function:outer_model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_image_generation_callable_returns_image_only_model(self, allow_model_requests: None):
        """Callable fallback_model returning an image-only model name is caught at call time."""

        def model_factory(ctx: RunContext) -> str:
            return 'openai-responses:gpt-image-1'

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args='{"prompt": "test"}')])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(outer_model, capabilities=[ImageGeneration(fallback_model=model_factory)])
        with pytest.raises(UserError, match="'gpt-image-1' is a dedicated image generation model"):
            await agent.run('Generate a test image')

    async def test_image_generation_subagent_content_filter_error_is_not_retried(self, allow_model_requests: None):
        """A moderation block aborts the run rather than becoming a retry prompt.

        `ContentFilterError` subclasses `UnexpectedModelBehavior`, so it would otherwise be swallowed
        by the retry mapping below; a provider refusal is deterministic and rephrasing is the caller's
        decision, matching how the agent loop treats a `content_filter` finish reason.
        """

        def blocked_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise ContentFilterError('image generation was blocked for content moderation')

        inner_model = FunctionModel(blocked_model_fn, profile=ModelProfile(supports_image_output=True))

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args='{"prompt": "test"}')])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(outer_model, capabilities=[ImageGeneration(fallback_model=inner_model)])

        with pytest.raises(ContentFilterError, match='blocked for content moderation'):
            await agent.run('Generate a test image')

    async def test_image_generation_subagent_error_becomes_model_retry(self, allow_model_requests: None):
        """UnexpectedModelBehavior from subagent becomes a retry prompt to the outer model."""

        # FunctionModel that returns text but no image — triggers UnexpectedModelBehavior
        def no_image_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='No image generated.')])

        inner_model = FunctionModel(no_image_model_fn, profile=ModelProfile(supports_image_output=True))

        call_count = 0

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args='{"prompt": "test"}')])
            return ModelResponse(parts=[TextPart(content='gave up')])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(outer_model, capabilities=[ImageGeneration(fallback_model=inner_model)])
        result = await agent.run('Generate a test image')
        assert result.output == 'gave up'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Generate a test image', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='generate_image',
                            args='{"prompt": "test"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=54, output_tokens=5),
                    model_name='function:outer_model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Exceeded maximum output retries (1)',
                            tool_name='generate_image',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='gave up')],
                    usage=RequestUsage(input_tokens=66, output_tokens=7),
                    model_name='function:outer_model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    @pytest.mark.parametrize(
        'provider, model_name, suggestion',
        [
            ('openai-responses', 'gpt-image-2', 'openai-responses:gpt-5.5'),
            ('openai-responses', 'gpt-image-1.5', 'openai-responses:gpt-5.5'),
            ('openai-responses', 'gpt-image-1', 'openai-responses:gpt-5.4'),
            ('openai-responses', 'gpt-image-1-mini', 'openai-responses:gpt-5.4'),
            ('google', 'imagen-3.0-generate-002', 'google:gemini-3-pro-image'),
            ('google', 'imagen-3.0-fast-generate-001', 'google:gemini-3-pro-image'),
        ],
    )
    def test_image_generation_rejects_image_only_model(self, provider: str, model_name: str, suggestion: str):
        """Using a dedicated image model raises a clear error with a conversational alternative."""
        with pytest.raises(
            UserError,
            match=re.escape(
                f'{model_name!r} is a dedicated image generation model that cannot be used as '
                f'`fallback_model` directly. Pass an `ImageGenerator` with a direct image model '
                f'to `local` instead, or use a conversational model with image generation support, '
                f'e.g. {suggestion!r}.'
            ),
        ):
            ImageGeneration(fallback_model=f'{provider}:{model_name}')

    @pytest.mark.vcr()
    async def test_image_generation_local_fallback(self, allow_model_requests: None, openai_api_key: str):
        """ImageGeneration(fallback_model=...) with non-supporting outer model uses subagent fallback."""
        from pydantic_ai.messages import BinaryImage
        from pydantic_ai.models.openai import OpenAIResponsesModel
        from pydantic_ai.providers.openai import OpenAIProvider

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # If we see a tool return, the image was generated — return final text
            if any(
                isinstance(part, ToolReturnPart)
                for msg in messages
                if isinstance(msg, ModelRequest)
                for part in msg.parts
            ):
                return ModelResponse(parts=[TextPart(content='Here is the generated image.')])

            # First call: invoke the generate_image tool
            assert info.function_tools, 'Expected generate_image tool to be available'
            tool = info.function_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args='{"prompt": "A cute baby sea otter"}')])

        inner_model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key=openai_api_key))
        outer_model = FunctionModel(model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(
            outer_model,
            capabilities=[
                ImageGeneration(fallback_model=inner_model),
            ],
        )
        result = await agent.run('Generate an image of a cute baby sea otter')
        assert result.output == 'Here is the generated image.'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(content='Generate an image of a cute baby sea otter', timestamp=IsDatetime())
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='generate_image',
                            args='{"prompt": "A cute baby sea otter"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=59, output_tokens=9),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='generate_image',
                            content=IsInstance(BinaryImage),
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Here is the generated image.')],
                    usage=RequestUsage(input_tokens=59, output_tokens=15),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    @pytest.mark.vcr()
    async def test_image_generation_local_fallback_google(self, allow_model_requests: None, gemini_api_key: str):
        """ImageGeneration fallback with Google image model."""
        pytest.importorskip('google.genai', reason='google extra not installed')
        from pydantic_ai.messages import BinaryImage
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(content='Here is the generated image.')])
            assert info.function_tools, 'Expected generate_image tool to be available'
            tool = info.function_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args='{"prompt": "A cute baby sea otter"}')])

        inner_model = GoogleModel('gemini-3-pro-image', provider=GoogleProvider(api_key=gemini_api_key))
        outer_model = FunctionModel(model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(outer_model, capabilities=[ImageGeneration(fallback_model=inner_model)])
        result = await agent.run('Generate an image of a cute baby sea otter')
        assert result.output == 'Here is the generated image.'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(content='Generate an image of a cute baby sea otter', timestamp=IsDatetime())
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='generate_image',
                            args='{"prompt": "A cute baby sea otter"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=59, output_tokens=9),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='generate_image',
                            content=IsInstance(BinaryImage),
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Here is the generated image.')],
                    usage=RequestUsage(input_tokens=59, output_tokens=15),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )
