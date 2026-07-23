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
    """A deterministic image generation model for testing."""

    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    _model_name: str
    _provider_name: str
    last_images: list[ImageGenerationInput]
    last_settings: ImageGenerationSettings | None = None

    def __init__(
        self,
        model_name: str = 'test',
        *,
        provider_name: str = 'test',
        settings: ImageGenerationSettings | None = None,
    ):
        self._model_name = model_name
        self._provider_name = provider_name
        self.last_images = []
        self.last_settings = None
        super().__init__(settings=settings)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
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
        image_count = settings.get('n') or 1
        output_format = settings.get('output_format') or 'png'
        media_type = f'image/{output_format}'

        return ImageGenerationResult(
            images=[
                GeneratedImage(
                    content=BinaryImage(data=_TINY_PNG, media_type=media_type),
                    size='1x1',
                    output_format=output_format,
                )
                for _ in range(image_count)
            ],
            prompt=prompt,
            usage=RequestUsage(input_tokens=_estimate_tokens(prompt)),
            model_name=self.model_name,
            provider_name=self.system,
            settings=settings,
            provider_response_id=str(uuid.uuid4()),
        )
