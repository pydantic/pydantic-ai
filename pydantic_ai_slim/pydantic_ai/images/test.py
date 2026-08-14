import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai.messages import BinaryImage
from pydantic_ai.usage import RequestUsage

from .base import ImageGenerationInput, ImageGenerationModel
from .result import GeneratedImage, ImageGenerationResult
from .settings import ImageGenerationSettings

_TOKEN_SPLIT_RE = re.compile(r'[\s",.:]+')
_TINY_PNG = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
    b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01'
    b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0  # pragma: no cover
    return len(_TOKEN_SPLIT_RE.split(text.strip()))


@dataclass(init=False)
class TestImageGenerationModel(ImageGenerationModel):
    """A deterministic image generation model for testing.

    This model returns a single 1x1 PNG without making any API calls, and records the reference images
    and settings used in the last call via the `last_images` and `last_settings` attributes.

    Example:
    ```python
    from pydantic_ai import ImageGenerator
    from pydantic_ai.images import TestImageGenerationModel

    test_model = TestImageGenerationModel()
    generator = ImageGenerator('openai:gpt-image-2')


    async def main():
        with generator.override(model=test_model):
            await generator.generate('A test image', settings={'aspect_ratio': '16:9'})
            print(test_model.last_settings)
            #> {'aspect_ratio': '16:9'}
            print(test_model.last_images)
            #> []
    ```
    """

    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    _model_name: str
    """The model name to report in results."""

    _provider_name: str
    """The provider name to report in results."""

    last_images: list[ImageGenerationInput]
    """The reference images passed to the most recent generate call."""

    last_settings: ImageGenerationSettings | None = None
    """The settings used in the most recent generate call."""

    def __init__(
        self,
        model_name: str = 'test',
        *,
        provider_name: str = 'test',
        settings: ImageGenerationSettings | None = None,
    ):
        """Initialize the test image generation model.

        Args:
            model_name: The model name to report in results.
            provider_name: The provider name to report in results.
            settings: Optional default settings for the model.
        """
        self._model_name = model_name
        self._provider_name = provider_name
        self.last_images = []
        self.last_settings = None
        super().__init__(settings=settings)

    @property
    def model_name(self) -> str:
        """The image generation model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The image generation model provider."""
        return self._provider_name

    async def generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageGenerationInput] | None = None,
        settings: ImageGenerationSettings | None = None,
    ) -> ImageGenerationResult:
        prompt, images, settings = self.prepare_generate(prompt, images=images, settings=settings)
        self.last_images = images
        self.last_settings = settings

        return ImageGenerationResult(
            images=[
                GeneratedImage(
                    content=BinaryImage(data=_TINY_PNG, media_type='image/png'),
                    size='1x1',
                    output_format='png',
                )
            ],
            prompt=prompt,
            usage=RequestUsage(input_tokens=_estimate_tokens(prompt)),
            model_name=self.model_name,
            provider_name=self.system,
            settings=settings,
            provider_response_id=str(uuid.uuid4()),
        )
