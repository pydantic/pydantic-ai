from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, get_args

from typing_extensions import deprecated

from pydantic_ai._utils import replace_no_init
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.exceptions import ContentFilterError, ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.images import (
    ImageDimensions,
    ImageGenerationAspectRatio,
    ImageGenerationModel,
    ImageGenerationSettings,
    ImageGenerator,
)
from pydantic_ai.images._validation import DIMENSIONS_ASPECT_RATIO_CONFLICT
from pydantic_ai.messages import BinaryImage
from pydantic_ai.models import AbstractModel, KnownModelName, Model
from pydantic_ai.native_tools import (
    SUPPORTED_NATIVE_TOOLS,
    ImageAspectRatio,
    ImageGenerationModelName,
    ImageGenerationTool,
    ImageSize,
)
from pydantic_ai.tools import AgentDepsT, RunContext, Tool, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.prepared import PreparedToolset

from ._deprecated_fallback_model import resolve_fallback_subagent_model
from .abstract import AbstractCapability
from .native_or_local import NativeOrLocalTool

if TYPE_CHECKING:
    from pydantic_ai.common_tools.image_generation import ImageGenerationFallbackModel, ImageGenerationNativeTool

# Derived from the native tool's own aliases so widening either can't silently start dropping values.
_NATIVE_IMAGE_SIZES = frozenset(get_args(ImageSize))
_NATIVE_IMAGE_ASPECT_RATIOS = frozenset(get_args(ImageAspectRatio))

_EDIT_ACTION_UNSUPPORTED = (
    'The direct `ImageGeneration` fallback cannot honor `action="edit"` because the '
    '`generate_image` tool does not receive reference images. Use '
    '`ImageGenerator.generate(..., images=...)` directly for image editing.'
)

# Shared by the construction-time notice (`native=False`, where the direct generator is the only
# implementation) and the per-request one (a model with no native image generation drops the native
# tool), so both spellings of the same drop read identically.
_NATIVE_ONLY_SETTINGS_DROPPED = (
    'The direct `ImageGeneration` fallback ignored native-tool setting(s): {settings}. '
    'Configure provider-specific direct settings on the `ImageGenerator` or '
    '`ImageGenerationModel` instead.'
)


@dataclass(kw_only=True)
class _DirectImageGenerationTool:
    """Local capability tool backed directly by the image generation API."""

    generator: ImageGenerator | ImageGenerationModel
    settings: ImageGenerationSettings
    action: Literal['generate', 'edit', 'auto'] | None
    image_model: ImageGenerationModelName | None

    async def __call__(self, prompt: str) -> BinaryImage:
        if self.action == 'edit':
            # `ImageGeneration.__post_init__` already rejected this at construction when
            # `native=False`. With native enabled the native tool can honor the edit, so whether it
            # is unserviceable is only known once the model has dropped the native tool and called
            # this one instead.
            raise UserError(_EDIT_ACTION_UNSUPPORTED)
        if self.image_model is not None:
            warnings.warn(
                'Direct `ImageGeneration` fallback ignored `image_model`; `local` already selects the direct image model',
                UserWarning,
                stacklevel=2,
            )

        try:
            result = await self.generator.generate(prompt, settings=self.settings)
        except ContentFilterError as e:
            # Same conversion as the `fallback_subagent_model` path, so the capability fails the
            # same way on both fallbacks: the outer model gets to rephrase, and no exception escapes
            # the tool call for a durable engine to retry against an error class its non-retryable
            # list doesn't name. `ImageGenerator.generate` itself still raises `ContentFilterError`.
            raise ModelRetry(str(e)) from e
        if len(result.images) != 1:
            raise UnexpectedModelBehavior(
                f'Direct image generation fallback returned {len(result.images)} images; expected exactly one. '
                'If the generator asks for more than one image per call through a provider-specific '
                'image count setting, call `ImageGenerator.generate()` directly instead.'
            )
        return result.image


@dataclass(init=False)
class ImageGeneration(NativeOrLocalTool[AgentDepsT]):
    """Image generation capability.

    Uses the model's native image generation when available. When the model doesn't
    support it, pass an `ImageGenerator` or `ImageGenerationModel` to `local` to use
    the direct image generation API as a fallback.

    The `fallback_subagent_model` path is the other way to cover such a model: it runs an additional
    agent on an image-capable conversational model, so the image comes from that model's
    native `ImageGenerationTool`. Use it when you want those native tool semantics.

    Portable `dimensions` and `aspect_ratio` settings are applied to the direct fallback
    using `ImageGenerationSettings`. Other fields configure the native
    `ImageGenerationTool`; configure provider-specific direct settings on an explicit
    `ImageGenerator` or `ImageGenerationModel`.

    When passing a custom `native` instance or factory, its settings are also used for the
    `fallback_subagent_model` subagent; capability-level fields override any `native` settings. A
    static instance's `aspect_ratio` is also inherited by the direct fallback.
    """

    local: (
        str
        | Tool[AgentDepsT]
        | Callable[..., Any]
        | AbstractToolset[AgentDepsT]
        | ImageGenerator
        | ImageGenerationModel
        | bool
        | None
    ) = None
    """Configure the local fallback tool.

    Besides the `Tool`, toolset and callable shapes
    [`NativeOrLocalTool`][pydantic_ai.capabilities.NativeOrLocalTool] accepts, takes an
    [`ImageGenerator`][pydantic_ai.images.ImageGenerator] or an
    [`ImageGenerationModel`][pydantic_ai.images.ImageGenerationModel], which generate through the
    direct image API, or a `'provider:model'` string that resolves to an `ImageGenerator`. Any other
    string and `local=True` raise `UserError`: there is no named local strategy. The generator is
    kept as declared; the `generate_image` tool is derived from it and the capability's settings
    each time the toolset is requested.
    """

    fallback_subagent_model: ImageGenerationFallbackModel
    """Model for a subagent to run when the agent's model doesn't support image generation natively.

    Must be a model that supports image generation via the
    [`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool] native tool.
    This requires a conversational model with image generation support, not a dedicated
    image-only API — for one of those, pass it to `local` instead. Examples:

    * `'openai-responses:gpt-5.4'` — OpenAI model with image generation support
    * `'google:gemini-3-pro-image'` — Google image generation model

    Can be a model name string, `Model` instance, or a callable taking `RunContext`
    that returns a `Model` instance or model name string.
    """

    # Keep these fields in sync with ImageGenerationTool in native_tools.py.

    action: Literal['generate', 'edit', 'auto'] | None
    """Whether to generate a new image or edit an existing image.

    Supported by: OpenAI Responses. Default: `'auto'`.

    The direct `local=` generator receives no reference images, so `'edit'` raises `UserError`: at
    construction with `native=False`, and when the tool runs otherwise.
    """

    background: Literal['transparent', 'opaque', 'auto'] | None
    """Background type for the generated image.

    Supported by: OpenAI Responses.

    The direct `local=` generator ignores it; set the provider-prefixed equivalent on the generator instead.
    """

    input_fidelity: Literal['high', 'low'] | None
    """Input fidelity for matching style/features of input images.

    Supported by: OpenAI Responses. Default: `'low'`.

    The direct `local=` generator ignores it; set the provider-prefixed equivalent on the generator instead.
    """

    moderation: Literal['auto', 'low'] | None
    """Moderation level for the generated image.

    Supported by: OpenAI Responses.

    The direct `local=` generator ignores it; set the provider-prefixed equivalent on the generator instead.
    """

    image_model: ImageGenerationModelName | None
    """The image generation model to use.

    Supported by: OpenAI Responses.

    The direct `local=` generator ignores it with a warning, because `local` already names the model.
    """

    output_compression: int | None
    """Compression level for the output image.

    Supported by: OpenAI Responses (jpeg/webp, default: 100), Google Cloud (jpeg, default: 75).

    The direct `local=` generator ignores it; set the provider-prefixed equivalent on the generator instead.
    """

    output_format: Literal['png', 'webp', 'jpeg'] | None
    """Output format of the generated image.

    Supported by: OpenAI Responses (default: `'png'`), Google Cloud.

    The direct `local=` generator ignores it; set the provider-prefixed equivalent on the generator instead.
    """

    quality: Literal['low', 'medium', 'high', 'auto'] | None
    """Quality of the generated image.

    Supported by: OpenAI Responses.

    The direct `local=` generator ignores it; set the provider-prefixed equivalent on the generator instead.
    """

    size: ImageSize | None
    """Size of the generated image for the native tool.

    Supported by: OpenAI Responses (`'auto'`, `'1024x1024'`, `'1024x1536'`, `'1536x1024'`),
    Google Gemini 3 Pro Image and later (`'512'` on Gemini 3.1 Flash Image only, `'1K'`, `'2K'`, `'4K'`).

    Direct image APIs use provider-prefixed size or resolution settings.
    """

    dimensions: ImageDimensions | None
    """Exact direct-model output dimensions as `(width, height)` in pixels.

    This is mutually exclusive with `aspect_ratio`: passing both alongside a direct `local`
    generator raises `UserError` at construction. Only the direct `local` generator can apply it,
    so pass `native=False` to guarantee it takes effect: with the default `native=True` the direct
    generator is dropped whenever the conversational model generates images natively, and the
    native tool has no equivalent — that request warns. The `fallback_subagent_model` path ignores
    it with a warning. Supported shapes are model-specific; see the
    [Image Generation guide](../image-generation.md#supported-exact-dimensions).
    """

    aspect_ratio: ImageGenerationAspectRatio | None
    """Aspect ratio for generated images.

    Supported by: Google image-generation models (Gemini), OpenAI Responses (maps `'1:1'`, `'2:3'`,
    `'3:2'` to sizes).

    Direct adapters map this to a canonical geometry supported by the selected model. Ratios the
    native tool also accepts apply on either path; the rest need the direct generator, so pass
    `native=False` to guarantee them, as for `dimensions`, and a request that takes the native path
    instead warns. Ratios outside the native vocabulary are ignored by the `fallback_subagent_model`
    path with a warning. See the
    [ratio-to-dimensions matrix](../image-generation.md#canonical-dimensions-for-aspect_ratio).
    """

    id: str | None = 'image_generation'
    """One-off: an agent searches, fetches or generates one way, so the id is fixed.

    Declared here rather than only as an `__init__` default so the class states it where
    `_declares_default_id` -- and a reader -- can see it.
    """

    def __init__(
        self,
        *,
        native: ImageGenerationTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[ImageGenerationTool | None] | ImageGenerationTool | None]
        | bool = True,
        local: Tool[AgentDepsT]
        | Callable[..., Any]
        | ImageGenerator
        | ImageGenerationModel
        | str
        | Literal[False]
        | None = None,
        fallback_subagent_model: Model
        | KnownModelName
        | str
        | Callable[[RunContext[AgentDepsT]], Awaitable[Model | KnownModelName | str] | Model | KnownModelName | str]
        | None = None,
        action: Literal['generate', 'edit', 'auto'] | None = None,
        background: Literal['transparent', 'opaque', 'auto'] | None = None,
        input_fidelity: Literal['high', 'low'] | None = None,
        moderation: Literal['auto', 'low'] | None = None,
        image_model: ImageGenerationModelName | None = None,
        output_compression: int | None = None,
        output_format: Literal['png', 'webp', 'jpeg'] | None = None,
        quality: Literal['low', 'medium', 'high', 'auto'] | None = None,
        size: ImageSize | None = None,
        dimensions: ImageDimensions | None = None,
        aspect_ratio: ImageGenerationAspectRatio | None = None,
        id: str | None = 'image_generation',
        defer_loading: bool = False,
        description: str | None = None,
        # TODO(v3): remove `fallback_model`, the deprecated spelling of `fallback_subagent_model`.
        fallback_model: Model
        | KnownModelName
        | str
        | Callable[[RunContext[AgentDepsT]], Awaitable[Model | KnownModelName | str] | Model | KnownModelName | str]
        | None = None,
    ) -> None:
        self.id = id
        self.description = description
        self.defer_loading = defer_loading
        self.native = native
        self.fallback_subagent_model = resolve_fallback_subagent_model(
            type(self).__name__, fallback_subagent_model, fallback_model
        )
        self.action = action
        self.background = background
        self.input_fidelity = input_fidelity
        self.moderation = moderation
        self.image_model = image_model
        self.output_compression = output_compression
        self.output_format = output_format
        self.quality = quality
        self.size = size
        self.dimensions = dimensions
        self.aspect_ratio = aspect_ratio
        self.local = local
        self.__post_init__()

    def __post_init__(self) -> None:
        # Checked here rather than in `__init__` so a merge is held to it too: `combine` can pair
        # one instance's `fallback_subagent_model` with another's `local`, which no constructor
        # accepts, and the local tool would then take effect with `fallback_subagent_model` silently
        # ignored. Runs before the base resolves `local`, so it reads what was declared rather than
        # what was materialized.
        if self.fallback_subagent_model is not None and self.local is not None:
            raise UserError(
                'ImageGeneration: cannot specify both `fallback_subagent_model` and `local` — '
                'use `fallback_subagent_model` for the default subagent fallback, or `local` for a custom tool'
            )

        if isinstance(self.local, str):
            if ':' not in self.local:
                raise UserError(self._unsupported_local(self.local))
            # The provider prefix is the only part of the id resolvable without credentials, so
            # checking it here keeps the unsupported-strategy rejection at construction while the
            # model itself stays deferred to the first generate call. `local` keeps the generator
            # rather than a tool built from it: the tool is derived in `get_toolset`, so the settings
            # it carries can still change until then.
            #
            # The suppression covers the widened `local` field, which pyright reports at the field's
            # last assignment; `get_toolset` narrows the generator back to the `Tool` the base expects.
            self.local = ImageGenerator(self.local)  # pyright: ignore[reportIncompatibleVariableOverride]

        if self._has_direct_generator:
            # Reject at construction what only the direct generator could have served.
            #
            # The direct model rejects the geometry pair on every `generate` call, but a pair the
            # user set on the capability himself is already decided here, so it fails at construction
            # rather than at the first `generate_image` call. Ungated by `native`: whichever path a
            # request takes, the settings the generator would carry are contradictory.
            if self.dimensions is not None and self.aspect_ratio is not None:
                raise UserError(DIMENSIONS_ASPECT_RATIO_CONFLICT)
            # `native=False` is the one configuration whose routing is settled here: the direct
            # generator is the only implementation, so an `action='edit'` it cannot serve and the
            # native-only settings it cannot apply are both decidable now. Everywhere else the native
            # tool is built too, carries every one of those settings, and supersedes the generator per
            # request in `models.resolve_request_tools` — reporting them as dropped would be wrong for
            # exactly the configurations that apply them. The request that does drop them warns
            # instead, from the prepare function `get_toolset` installs.
            # `_DirectImageGenerationTool.__call__` still rejects the edit action, at the point where
            # the direct tool is provably the one running.
            if self.native is False:
                if self.action == 'edit':
                    raise UserError(_EDIT_ACTION_UNSUPPORTED)
                if native_only := self._native_only_settings():
                    # user → `__init__` → here → `warn`; `from_spec` adds a frame and so lands one short.
                    warnings.warn(
                        _NATIVE_ONLY_SETTINGS_DROPPED.format(settings=', '.join(native_only)),
                        UserWarning,
                        stacklevel=3,
                    )

        # The native tool's kwargs are collected once for the default native tool and again for the
        # `fallback_subagent_model` subagent's copy, so the notice lives here to fire exactly once.
        ignored: list[str] = []
        if self.native is not False or (self.local is None and self.fallback_subagent_model is not None):
            _, ignored = self._native_geometry()
        elif not self._has_direct_generator:
            # `native=False` with a local tool of the user's own: no native tool is built and the
            # tool the capability didn't build carries no settings, so the geometry the native tool
            # could never express has nothing left to apply it. `size` and the other native-only
            # settings are the direct generator's to report, from the block above.
            ignored = self._direct_only_geometry()
        super().__post_init__()
        if ignored:
            # user → `__init__` → here → `warn`; `from_spec` adds a frame and so lands one short.
            warnings.warn(
                f'`ImageGeneration` ignored direct-only setting(s): {", ".join(ignored)}. '
                "Only a direct generator applies them: use `native=False` with `local='provider:image-model'` "
                'or an `ImageGenerator`.',
                UserWarning,
                stacklevel=3,
            )

    # TODO(v3): remove the `fallback_model` property, the deprecated spelling of `fallback_subagent_model`.
    # The message is spelled out rather than shared with the helper that warns at construction:
    # a type checker only reports a deprecation whose message is a string literal.
    @property
    @deprecated(
        '`fallback_model` is deprecated; use `fallback_subagent_model` instead.', category=PydanticAIDeprecationWarning
    )
    def fallback_model(self) -> ImageGenerationFallbackModel:
        """Deprecated alias for [`fallback_subagent_model`][pydantic_ai.capabilities.ImageGeneration.fallback_subagent_model]."""
        return self.fallback_subagent_model

    @fallback_model.setter
    @deprecated(
        '`fallback_model` is deprecated; use `fallback_subagent_model` instead.', category=PydanticAIDeprecationWarning
    )
    def fallback_model(self, value: ImageGenerationFallbackModel) -> None:
        self.fallback_subagent_model = value

    @property
    def _has_direct_generator(self) -> bool:
        """Whether `local` generates through the direct image API.

        Derived rather than recorded, so every instance the framework builds from this
        configuration -- `dataclasses.replace`, `combine` -- reads the same answer as the
        constructed one.
        """
        return isinstance(self.local, (ImageGenerator, ImageGenerationModel))

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Merge like `NativeOrLocalTool`, except that `dimensions` is one value, not a collection.

        The default merge unions two sequences, and a `(width, height)` pair's entries are not
        independent: `(1024, 1024)` beside `(1536, 1024)` unions to `(1024, 1536)`, a flipped
        orientation neither instance asked for, and two disjoint pairs union to a three-element
        tuple that is no size at all. It takes the later stated value instead, the rule the scalar
        fields already get.

        Applied after the base merge because `__post_init__` reads only whether `dimensions` is
        set, never what it is, and a merge never turns a stated pair into `None`.
        """
        merged = super().combine(capabilities)
        assert isinstance(merged, cls)
        stated = [
            capability.dimensions
            for capability in capabilities
            if isinstance(capability, ImageGeneration) and capability.dimensions is not None
        ]
        return replace_no_init(merged, dimensions=stated[-1]) if stated else merged

    @classmethod
    def from_spec(
        cls,
        *,
        native: ImageGenerationTool | bool = True,
        local: str | Literal[False] | None = None,
        fallback_subagent_model: KnownModelName | str | None = None,
        action: Literal['generate', 'edit', 'auto'] | None = None,
        background: Literal['transparent', 'opaque', 'auto'] | None = None,
        input_fidelity: Literal['high', 'low'] | None = None,
        moderation: Literal['auto', 'low'] | None = None,
        image_model: ImageGenerationModelName | None = None,
        output_compression: int | None = None,
        output_format: Literal['png', 'webp', 'jpeg'] | None = None,
        quality: Literal['low', 'medium', 'high', 'auto'] | None = None,
        size: ImageSize | None = None,
        dimensions: ImageDimensions | None = None,
        aspect_ratio: ImageGenerationAspectRatio | None = None,
        id: str | None = 'image_generation',
        defer_loading: bool = False,
        description: str | None = None,
        # TODO(v3): remove `fallback_model`, the deprecated spelling of `fallback_subagent_model`. It stays in
        # the signature, and so in the published spec schema, because that schema forbids extra keys:
        # dropping it would stop a spec written against the old name from validating at all.
        fallback_model: KnownModelName | str | None = None,
    ) -> ImageGeneration[AgentDepsT]:
        """Construct from the JSON/YAML-serializable subset of the runtime API.

        Runtime objects accepted by `local`, such as `ImageGenerator`, `ImageGenerationModel`,
        `Tool`, and callables, can be passed to `ImageGeneration(...)` directly but cannot be
        represented in an agent spec. A direct image model name is serializable and can be passed
        as `local='provider:model'`.
        """
        # JSON and YAML have no tuple, so a spec always spells `dimensions` as a list, and spec
        # kwargs reach here unvalidated — the annotation above is what the published spec schema
        # advertises, not something that coerces or rejects on the way in.
        if isinstance(dimensions, list):
            if len(dimensions) != 2:
                raise UserError('Image generation `dimensions` must contain exactly two integers')
            dimensions = (dimensions[0], dimensions[1])

        return cls(
            native=native,
            local=local,
            fallback_subagent_model=fallback_subagent_model,
            fallback_model=fallback_model,
            action=action,
            background=background,
            input_fidelity=input_fidelity,
            moderation=moderation,
            image_model=image_model,
            output_compression=output_compression,
            output_format=output_format,
            quality=quality,
            size=size,
            dimensions=dimensions,
            aspect_ratio=aspect_ratio,
            id=id,
            defer_loading=defer_loading,
            description=description,
        )

    def _direct_only_geometry(self) -> list[str]:
        """Geometry settings the native tool has no way to express."""
        direct_only: list[str] = []
        if self.dimensions is not None:
            direct_only.append('dimensions')
        if self.aspect_ratio is not None and self.aspect_ratio not in _NATIVE_IMAGE_ASPECT_RATIOS:
            direct_only.append('aspect_ratio')
        return direct_only

    def _native_only_settings(self) -> list[str]:
        """Settings only the native tool can express, which a direct generator drops."""
        # Collected as a table rather than a chain of `if`s to keep the callers under the
        # complexity limit.
        return [
            name
            for name, value in (
                ('background', self.background),
                ('input_fidelity', self.input_fidelity),
                ('moderation', self.moderation),
                ('output_compression', self.output_compression),
                ('output_format', self.output_format),
                ('quality', self.quality),
                ('size', self.size),
            )
            if value is not None
        ]

    def _native_geometry(self) -> tuple[dict[str, Any], list[str]]:
        """The geometry settings the native tool can express, and the ones it can't.

        `dimensions` and `aspect_ratio` are only reported as ignored when no direct generator is
        configured, since the `generate_image` tool built for one forwards both. `size` has no
        direct counterpart, so it is dropped whichever path runs.

        That suppression is a construction-time approximation: this runs from `__post_init__`, before
        a model exists, while native-vs-local is decided per request in `models.resolve_request_tools`.
        Warning here regardless would fire on every configuration whose model has no native image
        generation, where the settings *are* applied. The case it misses — `native=True` plus a natively
        capable model, which drops the direct generator — is warned about per request instead, from the
        prepare function `get_toolset` installs.

        Split out of `_image_gen_kwargs` only to keep that method under the complexity limit.
        """
        kwargs: dict[str, Any] = {}
        ignored: list[str] = []
        if self.size is not None:
            if self.size in _NATIVE_IMAGE_SIZES:
                kwargs['size'] = self.size
            else:
                ignored.append('size')
        if self.aspect_ratio is not None and self.aspect_ratio in _NATIVE_IMAGE_ASPECT_RATIOS:
            kwargs['aspect_ratio'] = self.aspect_ratio
        if not self._has_direct_generator:
            ignored.extend(self._direct_only_geometry())
        return kwargs, ignored

    def _image_gen_kwargs(self) -> dict[str, Any]:
        """Collect settings supported by the native `ImageGenerationTool` path."""
        kwargs: dict[str, Any] = {}
        if self.background is not None:
            kwargs['background'] = self.background
        if self.input_fidelity is not None:
            kwargs['input_fidelity'] = self.input_fidelity
        if self.moderation is not None:
            kwargs['moderation'] = self.moderation
        if self.output_compression is not None:
            kwargs['output_compression'] = self.output_compression
        if self.output_format is not None:
            kwargs['output_format'] = self.output_format
        if self.quality is not None:
            kwargs['quality'] = self.quality

        geometry, _ = self._native_geometry()
        kwargs.update(geometry)

        if self.action is not None:
            kwargs['action'] = self.action
        if self.image_model is not None:
            kwargs['model'] = self.image_model
        return kwargs

    def _default_native(self) -> ImageGenerationTool:
        return ImageGenerationTool(**self._image_gen_kwargs())

    def _unsupported_local(self, value: str | bool) -> str:
        return (
            f'{type(self).__name__}: `local={value!r}` is not supported. Name a direct image '
            "model as `local='provider:model'`, or pass a `Tool`, `AbstractToolset`, or "
            'callable directly.'
        )

    def _resolve_local_strategy(self, name: str | bool) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]:
        # `__post_init__` turns a `'provider:model'` string into an `ImageGenerator` and rejects
        # every other string, so only `local=True` reaches the base's strategy hook.
        raise UserError(self._unsupported_local(name))

    def _direct_local_tool(self, generator: ImageGenerator | ImageGenerationModel) -> Tool[Any]:
        """Build the `generate_image` tool from the capability's current settings.

        Derived when the toolset is requested rather than stored at construction, so what the tool
        carries is always what the capability declares: `dataclasses.replace` and `combine` both
        produce an instance whose fields no longer match a tool built from an earlier one.
        """
        settings: ImageGenerationSettings = {}
        if self.dimensions is not None:
            settings['dimensions'] = self.dimensions
        # A custom `native` instance is the base and capability-level fields override it, the same
        # precedence `_resolved_native` gives the `fallback_subagent_model` subagent. `size` has no
        # counterpart on the other side of that merge; `dimensions` is the capability's own
        # spelling of the geometry the inherited `aspect_ratio` expresses, and the two are mutually
        # exclusive in `ImageGenerationSettings`, so inheriting alongside it would fail the generate
        # call over a setting the user never passed to the capability.
        aspect_ratio = self.aspect_ratio
        if aspect_ratio is None and self.dimensions is None and isinstance(self.native, ImageGenerationTool):
            aspect_ratio = self.native.aspect_ratio
        if aspect_ratio is not None:
            settings['aspect_ratio'] = aspect_ratio
        return Tool[Any](
            _DirectImageGenerationTool(
                generator=generator,
                settings=settings,
                action=self.action,
                image_model=self.image_model,
            ).__call__,
            name='generate_image',
            description='Generate an image based on the given prompt.',
        )

    def _native_unique_id(self) -> str:
        return ImageGenerationTool.kind

    def _resolved_native(self) -> ImageGenerationNativeTool[AgentDepsT]:
        """Get the ImageGenerationTool for the fallback, with capability-level overrides applied."""
        return self._resolve_native_with_overrides(ImageGenerationTool, self._image_gen_kwargs())

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        if self.fallback_subagent_model is None:
            return None
        from pydantic_ai.common_tools.image_generation import image_generation_tool

        return image_generation_tool(model=self.fallback_subagent_model, native_tool=self._resolved_native())

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        capability = self
        if isinstance(self.local, (ImageGenerator, ImageGenerationModel)):
            # The base only knows how to wrap a `Tool` or a toolset, so the generator is resolved
            # into one on a copy. Building it here rather than keeping it on the capability is what
            # keeps a replaced or merged instance from sending an earlier one's settings.
            capability = replace_no_init(self, local=self._direct_local_tool(self.local))
        toolset = super(ImageGeneration, capability).get_toolset()
        # A callable `native` is resolved per request by the framework, so whether it yields a tool
        # that supersedes the generator can't be known here without invoking it a second time.
        # A resolved `native` tool also means the base wrapped the local toolset for `unless_native`,
        # so the diagnostics join that prepare function instead of nesting a second wrapper.
        if (
            not isinstance(toolset, PreparedToolset)
            or not isinstance(self.native, ImageGenerationTool)
            or not self._has_direct_generator
        ):
            return toolset

        direct_only = self._direct_only_geometry()
        native_only = self._native_only_settings()
        if not direct_only and not native_only:
            return toolset

        add_unless_native = toolset.prepare_func

        def _warn_about_settings_the_chosen_path_drops(
            ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
        ) -> Awaitable[list[ToolDefinition]] | list[ToolDefinition]:
            # Read through `__dict__` because a run context rehydrated across a durable boundary
            # (`TemporalRunContext` inside an activity, where a `DynamicCapability` re-resolves this
            # toolset) deliberately doesn't carry the live model and raises on attribute access.
            # `ctx.model` is an `AbstractModel`, and only a regular `Model` carries the profile that
            # says whether the native tool supersedes the generator.
            model: AbstractModel | None = ctx.__dict__.get('model')
            if isinstance(model, Model):
                # Which side of the swap runs is the request's to know, and each side drops what
                # only the other can express. Both notices carry the same `stacklevel`: every frame
                # between here and user code is framework toolset plumbing of unbounded depth, so
                # this attributes to the immediate caller rather than misreporting an arbitrary
                # internal frame as the user's.
                native_supersedes = ImageGenerationTool in model.profile.get(
                    'supported_native_tools', SUPPORTED_NATIVE_TOOLS
                )
                if native_supersedes and direct_only:
                    warnings.warn(
                        f'The `ImageGeneration` native tool supersedes the direct generator on {model.model_name}, '
                        f'so direct-only setting(s) go unapplied: {", ".join(direct_only)}. '
                        'Pass `native=False` to guarantee them.',
                        UserWarning,
                        stacklevel=2,
                    )
                elif not native_supersedes and native_only:
                    warnings.warn(
                        _NATIVE_ONLY_SETTINGS_DROPPED.format(settings=', '.join(native_only)),
                        UserWarning,
                        stacklevel=2,
                    )
            return add_unless_native(ctx, tool_defs)

        return replace(toolset, prepare_func=_warn_about_settings_the_chosen_path_drops)
