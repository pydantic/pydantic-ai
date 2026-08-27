from __future__ import annotations

import pytest

from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.native_tools import ImageGenerationTool

pytestmark = pytest.mark.anyio


class TestImageGenerationCapability:
    def test_image_gen_init_params_match_builtin_tool(self):
        """ImageGeneration.__init__ accepts all ImageGenerationTool configurable fields."""
        import dataclasses
        import inspect

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
        assert init_params == builtin_fields

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

    def test_image_generation_with_fallback_model(self):
        """ImageGeneration(fallback_model=...) creates a local fallback tool."""
        from pydantic_ai.tools import Tool

        cap = ImageGeneration(fallback_model='openai-responses:gpt-5.4')
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None
        builtins = cap.get_native_tools()
        assert len(builtins) == 1
        assert isinstance(builtins[0], ImageGenerationTool)

    def test_image_generation_forwards_config_to_builtin(self):
        """ImageGeneration config fields are forwarded to the ImageGenerationTool builtin."""
        cap = ImageGeneration(
            provider_settings={'openai': {'quality': 'low', 'partial_images': 2}},
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
        assert tool.provider_settings == {
            'openai': {
                'action': 'generate',
                'background': 'opaque',
                'input_fidelity': 'high',
                'moderation': 'low',
                'model': 'gpt-image-2',
                'partial_images': 2,
                'quality': 'high',
            }
        }
        assert tool.output_compression == 80
        assert tool.output_format == 'jpeg'
        assert tool.size == '1024x1024'
        assert tool.aspect_ratio == '16:9'

    def test_image_generation_fallback_merges_custom_native_with_overrides(self):
        """Custom native tool settings are merged with capability-level overrides for the fallback."""
        from pydantic_ai.tools import Tool

        custom_native = ImageGenerationTool(provider_settings={'openai': {'quality': 'high'}}, size='1024x1024')
        cap = ImageGeneration(
            native=custom_native,
            fallback_model='openai-responses:gpt-5.4',
            action='generate',
            output_format='jpeg',  # capability-level override
        )
        resolved_native = cap._resolved_native()  # pyright: ignore[reportPrivateUsage]
        assert resolved_native.provider_settings == {'openai': {'quality': 'high', 'action': 'generate'}}
        assert resolved_native.output_format == 'jpeg'
        # The local fallback should exist and contain the merged config
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    def test_image_generation_callable_native_with_fallback(self):
        """When native is a callable, the fallback local tool still gets created."""
        from pydantic_ai.tools import Tool

        cap = ImageGeneration(
            native=lambda ctx: ImageGenerationTool(provider_settings={'openai': {'quality': 'high'}}),
            fallback_model='openai-responses:gpt-5.4',
        )
        # Callable native can't be resolved at init time, but local fallback is still created
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None
