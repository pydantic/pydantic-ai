"""DashScope (Alibaba Cloud) vision + structured output example.

Demonstrates image understanding with a Qwen VL model via the recommended
`alibaba:` provider prefix, then structured output and copy generation with a
text model such as `qwen-plus`.

Qwen VL models reliably return plain text for image prompts. For validated
structured fields, run a second agent with `output_type` on a text model.

Run with:

    export DASHSCOPE_API_KEY='your-key'   # or ALIBABA_API_KEY
    uv run -m pydantic_ai_examples.dashscope_vision

For the China region endpoint, set `ALIBABA_BASE_URL` (read by this example only;
see docs/models/openai.md to configure AlibabaProvider explicitly in your own app):

    export ALIBABA_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
    uv run -m pydantic_ai_examples.dashscope_vision
"""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel, Field

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.alibaba import AlibabaProvider

# Public sample image (Pydantic logo) — same URL used in docs/input.md
SAMPLE_IMAGE_URL = 'https://iili.io/3Hs4FMg.png'
SAMPLE_IMAGE_MEDIA_TYPE = 'image/png'

DEFAULT_VISION_MODEL = 'alibaba:qwen-vl-plus'
DEFAULT_TEXT_MODEL = 'alibaba:qwen-plus'

VISION_PROMPT = 'Describe this image in a few sentences.'
CAPTION_PROMPT_PREFIX = 'Describe this image and suggest concise tags based on: '
COPY_PROMPT_PREFIX = 'Write one short social-media sentence based on this image description: '


class ImageCaption(BaseModel):
    description: str
    tags: list[str] = Field(min_length=1, max_length=5)


def _print_step(title: str, result) -> None:
    print(f'\n--- {title} ---')
    print(result.output)
    print(f'usage: {result.usage}')


def _agent(model: str, *, output_type: type[ImageCaption] | None = None) -> Agent:
    """Build an Agent, optionally with a custom DashScope base URL for this example."""
    prefix, _, name = model.partition(':')
    if prefix != 'alibaba':
        raise ValueError(f'Expected alibaba:<model>, got {model!r}')

    base_url = os.getenv('ALIBABA_BASE_URL')
    if base_url:
        openai_model = OpenAIChatModel(name, provider=AlibabaProvider(base_url=base_url))
        if output_type is None:
            return Agent(openai_model)
        return Agent(openai_model, output_type=output_type)

    if output_type is None:
        return Agent(model)
    return Agent(model, output_type=output_type)


def main() -> None:
    api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('ALIBABA_API_KEY')
    if not api_key:
        raise SystemExit('Set DASHSCOPE_API_KEY or ALIBABA_API_KEY to run this example.')

    vision_model = os.getenv('PYDANTIC_AI_VISION_MODEL', DEFAULT_VISION_MODEL)
    text_model = os.getenv('PYDANTIC_AI_TEXT_MODEL', DEFAULT_TEXT_MODEL)

    print(f'Vision model: {vision_model}')
    print(f'Text model:   {text_model}')
    if base := os.getenv('ALIBABA_BASE_URL'):
        print(f'Base URL:     {base}')

    image_bytes = httpx.get(SAMPLE_IMAGE_URL, follow_redirects=True).content

    vision_agent = _agent(vision_model)
    vision = vision_agent.run_sync(
        [
            VISION_PROMPT,
            BinaryContent(data=image_bytes, media_type=SAMPLE_IMAGE_MEDIA_TYPE),
        ]
    )
    _print_step('Vision description (plain text)', vision)

    caption_agent = _agent(text_model, output_type=ImageCaption)
    caption = caption_agent.run_sync(f'{CAPTION_PROMPT_PREFIX}{vision.output}')
    _print_step('Image caption (structured, text model)', caption)

    copy_agent = _agent(text_model)
    blurb = copy_agent.run_sync(f'{COPY_PROMPT_PREFIX}{caption.output.description}')
    _print_step('Generated copy (text model)', blurb)


if __name__ == '__main__':
    main()
