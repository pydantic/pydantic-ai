from __future__ import annotations

import warnings

import pytest

from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.native_tools import ImageGenerationTool

pytestmark = pytest.mark.anyio

# Keep these focused configuration tests separate because `test_capabilities.py` is at the repository's file-size limit.


class TestImageGenerationConfiguration:
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
        assert cap.local is None
        assert cap.get_toolset() is None

    def test_image_generation_with_custom_local(self):
        """ImageGeneration(local=custom) provides a custom local fallback."""
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
        """Nested settings win over convenience fields without emitting a deprecation warning."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', PydanticAIDeprecationWarning)
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
                'quality': 'low',
            }
        }
        assert tool.output_compression == 80
        assert tool.output_format == 'jpeg'
        assert tool.size == '1024x1024'
        assert tool.aspect_ratio == '16:9'

    def test_image_generation_fallback_merges_custom_native_with_overrides(self):
        """Capability-level settings override a custom native tool for the fallback."""
        from pydantic_ai.tools import Tool

        custom_native = ImageGenerationTool(provider_settings={'openai': {'quality': 'high'}}, size='1024x1024')
        cap = ImageGeneration(
            native=custom_native,
            fallback_model='openai-responses:gpt-5.4',
            action='generate',
            output_format='jpeg',
            quality='low',
        )
        resolved_native = cap._resolved_native()  # pyright: ignore[reportPrivateUsage]
        assert resolved_native.provider_settings == {'openai': {'quality': 'low', 'action': 'generate'}}
        assert resolved_native.output_format == 'jpeg'
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    def test_image_generation_fallback_does_not_repeat_legacy_warning(self):
        with pytest.warns(PydanticAIDeprecationWarning, match=r'field `quality` is deprecated'):
            custom_native = ImageGenerationTool(quality='high')

        with warnings.catch_warnings():
            warnings.simplefilter('error', PydanticAIDeprecationWarning)
            cap = ImageGeneration(
                native=custom_native,
                fallback_model='openai-responses:gpt-5.4',
                action='generate',
            )
            resolved_native = cap._resolved_native()  # pyright: ignore[reportPrivateUsage]

        assert resolved_native.quality == 'high'
        assert resolved_native.provider_settings == {'openai': {'action': 'generate'}}

    def test_image_generation_resolves_portable_override(self):
        cap = ImageGeneration(native=ImageGenerationTool(size='1024x1024'), output_format='jpeg')

        resolved_native = cap._resolved_native()  # pyright: ignore[reportPrivateUsage]

        assert resolved_native.size == '1024x1024'
        assert resolved_native.output_format == 'jpeg'

    def test_image_generation_callable_native_with_fallback(self):
        """When native is a callable, the fallback local tool still gets created."""
        from pydantic_ai.tools import Tool

        cap = ImageGeneration(
            native=lambda ctx: ImageGenerationTool(provider_settings={'openai': {'quality': 'high'}}),
            fallback_model='openai-responses:gpt-5.4',
        )
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None
