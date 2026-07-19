from __future__ import annotations

# ruff: noqa: E402
import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

_THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)

import logfire

from pydantic_ai import Agent, Embedder, UploadedFile
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.images import ImageGenerator
from pydantic_ai.images.google import GoogleImageGenerationModel, GoogleImageGenerationSettings
from pydantic_ai.images.openai import OpenAIImageGenerationSettings
from pydantic_ai.images.xai import XaiImageGenerationModel, XaiImageGenerationSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.xai import XaiProvider

ProviderName = Literal['openai', 'google', 'xai']
_PROVIDER_NAMES: tuple[ProviderName, ...] = ('openai', 'google', 'xai')
_GENERATION_PROMPT = 'A cat with a cowboy hat, dancing in Rome.'
_MUTATION_PROMPT = 'Replace the cat with a dog while preserving the cowboy hat, dancing pose, and Rome setting.'


def main() -> None:
    providers = _parse_providers()
    logfire.configure(send_to_logfire='if-token-present')

    asyncio.run(_run_instrumented_demo(providers))

    logfire.force_flush()


async def _run_instrumented_demo(providers: set[ProviderName]) -> None:
    with logfire.span('pydantic_ai_images_demo'):
        await run_demo(providers)


def _parse_providers(args: Sequence[str] | None = None) -> set[ProviderName]:
    parser = argparse.ArgumentParser(description='Run the image generation provider demo.')
    parser.add_argument(
        '-p',
        '--provider',
        action='append',
        choices=_PROVIDER_NAMES,
        dest='providers',
        help='Provider to run. May be repeated; defaults to all providers.',
    )
    parsed_args = parser.parse_args(args)
    providers = cast(list[ProviderName] | None, parsed_args.providers)
    return set(providers or _PROVIDER_NAMES)


async def run_demo(providers: set[ProviderName] | None = None) -> None:
    selected_providers = set(_PROVIDER_NAMES) if providers is None else providers
    if 'openai' in selected_providers:
        await _run_openai_demo()
    if 'google' in selected_providers:
        await _run_google_demo()
    if 'xai' in selected_providers:
        await _run_xai_demo()


async def _run_openai_demo() -> None:
    model = OpenAIEmbeddingModel('text-embedding-3-small')
    embedder = Embedder(model, instrument=True)

    result = await embedder.embed_query('Hello from instrumented embeddings')

    try:
        cost = result.cost().total_price
    except LookupError:
        cost = Decimal(0)

    print('embedding dimensions:', len(result.embeddings[0]))
    print('embedding usage:', result.usage)
    print('embedding cost:', cost)

    agent = Agent(
        'openai:gpt-5.4-nano',
        instructions='Reply in one short sentence.',
    )
    agent.instrument = True
    agent_result = await agent.run('Say hello from the agent instrumentation test.')
    print('agent output:', agent_result.output)
    print('agent usage:', agent_result.usage)

    image_settings = OpenAIImageGenerationSettings(openai_size='1024x1024', openai_quality='low', output_format='png')

    for image_model_name in ['gpt-image-1', 'gpt-image-1-mini', 'gpt-image-1.5']:
        image_generator = ImageGenerator(f'openai:{image_model_name}', instrument=True)
        image_result = await image_generator.generate(
            _GENERATION_PROMPT,
            settings=image_settings,
        )

        image_path = _THIS_DIR / f'_demo_output_{image_model_name}.png'
        with image_path.open('wb') as f:
            f.write(image_result.images[0].content.data)

        try:
            image_cost = image_result.cost().total_price
        except LookupError:
            image_cost = Decimal(0)

        print('image count:', len(image_result.images))
        print('image model:', image_model_name)
        print('image media type:', image_result.images[0].content.media_type)
        print('image usage:', image_result.usage)
        print('image cost:', image_cost)
        print('image saved to:', str(image_path))

        if image_model_name == 'gpt-image-1.5':
            edited_image_result = await image_generator.generate(
                _MUTATION_PROMPT,
                images=[image_result.images[0].content],
                settings=image_settings,
            )
            edited_image_path = _THIS_DIR / '_demo_output_gpt-image-1.5_edited.png'
            with edited_image_path.open('wb') as f:
                f.write(edited_image_result.images[0].content.data)

            try:
                edited_image_cost = edited_image_result.cost().total_price
            except LookupError:
                edited_image_cost = Decimal(0)

            print('edited image count:', len(edited_image_result.images))
            print('edited image media type:', edited_image_result.images[0].content.media_type)
            print('edited image usage:', edited_image_result.usage)
            print('edited image cost:', edited_image_cost)
            print('edited image saved to:', str(edited_image_path))


async def _run_google_demo() -> None:
    google_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not google_api_key:
        raise RuntimeError('Set `GOOGLE_API_KEY` to run the Google image generation demo')
    google_provider = GoogleProvider(api_key=google_api_key)
    google_image_model = GoogleImageGenerationModel('gemini-3.1-flash-lite-image', provider=google_provider)
    google_image_generator = ImageGenerator(google_image_model, instrument=True)
    google_image_result = await google_image_generator.generate(
        _GENERATION_PROMPT,
        settings=GoogleImageGenerationSettings(google_image_config={'aspect_ratio': '1:1'}),
    )
    google_image_path = _THIS_DIR / '_demo_output_gemini-3.1-flash-lite-image.png'
    with google_image_path.open('wb') as f:
        f.write(google_image_result.images[0].content.data)

    try:
        google_image_cost = google_image_result.cost().total_price
    except LookupError:
        google_image_cost = Decimal(0)

    print('google image count:', len(google_image_result.images))
    print('google image media type:', google_image_result.images[0].content.media_type)
    print('google image usage:', google_image_result.usage)
    print('google image cost:', google_image_cost)
    print('google image saved to:', str(google_image_path))

    uploaded_image = await google_provider.client.aio.files.upload(
        file=google_image_path,
        config={'mime_type': google_image_result.images[0].content.media_type},
    )
    if not uploaded_image.uri or not uploaded_image.name:
        raise RuntimeError('Google Files API upload did not return a file URI and name')

    try:
        google_edited_image_result = await google_image_generator.generate(
            _MUTATION_PROMPT,
            images=[
                UploadedFile(
                    file_id=uploaded_image.uri,
                    provider_name='google',
                    media_type=uploaded_image.mime_type or google_image_result.images[0].content.media_type,
                )
            ],
            settings=GoogleImageGenerationSettings(google_image_config={'aspect_ratio': '1:1'}),
        )
    finally:
        await google_provider.client.aio.files.delete(name=uploaded_image.name)
    google_edited_image_path = _THIS_DIR / '_demo_output_gemini-3.1-flash-lite-image_edited.png'
    with google_edited_image_path.open('wb') as f:
        f.write(google_edited_image_result.images[0].content.data)

    try:
        google_edited_image_cost = google_edited_image_result.cost().total_price
    except LookupError:
        google_edited_image_cost = Decimal(0)

    print('google edited image count:', len(google_edited_image_result.images))
    print('google edited image media type:', google_edited_image_result.images[0].content.media_type)
    print('google edited image usage:', google_edited_image_result.usage)
    print('google edited image cost:', google_edited_image_cost)
    print('google edited image saved to:', str(google_edited_image_path))


async def _run_xai_demo() -> None:
    xai_api_key = os.getenv('XAI_API_KEY')
    if not xai_api_key:
        raise RuntimeError('Set `XAI_API_KEY` to run the xAI image generation demo')

    xai_provider = XaiProvider(api_key=xai_api_key)
    xai_image_model = XaiImageGenerationModel('grok-imagine-image', provider=xai_provider)
    xai_image_generator = ImageGenerator(xai_image_model, instrument=True)
    xai_settings = XaiImageGenerationSettings(xai_aspect_ratio='1:1', xai_resolution='1k')
    xai_image_result = await xai_image_generator.generate(_GENERATION_PROMPT, settings=xai_settings)
    xai_image_path = _THIS_DIR / '_demo_output_grok-imagine-image.jpeg'
    with xai_image_path.open('wb') as f:
        f.write(xai_image_result.images[0].content.data)

    try:
        xai_image_cost = xai_image_result.cost().total_price
    except LookupError:
        xai_image_cost = Decimal(0)

    print('xai image count:', len(xai_image_result.images))
    print('xai image media type:', xai_image_result.images[0].content.media_type)
    print('xai image usage:', xai_image_result.usage)
    print('xai image cost:', xai_image_cost)
    print('xai provider details:', xai_image_result.provider_details)
    print('xai image saved to:', str(xai_image_path))

    uploaded_image = await xai_provider.client.files.upload(
        xai_image_result.images[0].content.data,
        filename=xai_image_path.name,
    )
    try:
        xai_edited_image_result = await xai_image_generator.generate(
            _MUTATION_PROMPT,
            images=[
                UploadedFile(
                    file_id=uploaded_image.id,
                    provider_name='xai',
                    media_type=xai_image_result.images[0].content.media_type,
                )
            ],
            settings=xai_settings,
        )
    finally:
        await xai_provider.client.files.delete(uploaded_image.id)

    xai_edited_image_path = _THIS_DIR / '_demo_output_grok-imagine-image_edited.jpeg'
    with xai_edited_image_path.open('wb') as f:
        f.write(xai_edited_image_result.images[0].content.data)

    try:
        xai_edited_image_cost = xai_edited_image_result.cost().total_price
    except LookupError:
        xai_edited_image_cost = Decimal(0)

    print('xai edited image count:', len(xai_edited_image_result.images))
    print('xai edited image media type:', xai_edited_image_result.images[0].content.media_type)
    print('xai edited image usage:', xai_edited_image_result.usage)
    print('xai edited image cost:', xai_edited_image_cost)
    print('xai edited provider details:', xai_edited_image_result.provider_details)
    print('xai edited image saved to:', str(xai_edited_image_path))


if __name__ == '__main__':
    main()
