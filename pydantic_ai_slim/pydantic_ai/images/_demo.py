from __future__ import annotations

# ruff: noqa: E402
import argparse
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
from pydantic_ai.providers.google import GoogleProvider

ProviderName = Literal['openai', 'google']
_PROVIDER_NAMES: tuple[ProviderName, ...] = ('openai', 'google')
_GENERATION_PROMPT = 'A cat with a cowboy hat, dancing in Rome.'
_MUTATION_PROMPT = 'Replace the cat with a dog while preserving the cowboy hat, dancing pose, and Rome setting.'


def main() -> None:
    providers = _parse_providers()
    logfire.configure(send_to_logfire='if-token-present')

    with logfire.span('pydantic_ai_images_demo'):
        run_demo(providers)

    logfire.force_flush()


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


def run_demo(providers: set[ProviderName] | None = None) -> None:
    selected_providers = set(_PROVIDER_NAMES) if providers is None else providers
    if 'openai' in selected_providers:
        _run_openai_demo()
    if 'google' in selected_providers:
        _run_google_demo()


def _run_openai_demo() -> None:
    model = OpenAIEmbeddingModel('text-embedding-3-small')
    embedder = Embedder(model, instrument=True)

    result = embedder.embed_query_sync('Hello from instrumented embeddings')

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
    agent_result = agent.run_sync('Say hello from the agent instrumentation test.')
    print('agent output:', agent_result.output)
    print('agent usage:', agent_result.usage)

    image_settings = OpenAIImageGenerationSettings(openai_size='1024x1024', openai_quality='low', output_format='png')

    for image_model_name in ['gpt-image-1', 'gpt-image-1-mini', 'gpt-image-1.5']:
        image_generator = ImageGenerator(f'openai:{image_model_name}', instrument=True)
        image_result = image_generator.generate_sync(
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
            edited_image_result = image_generator.generate_sync(
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


def _run_google_demo() -> None:
    google_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not google_api_key:
        raise RuntimeError('Set `GOOGLE_API_KEY` to run the Google image generation demo')
    google_provider = GoogleProvider(api_key=google_api_key)
    google_image_model = GoogleImageGenerationModel('gemini-3.1-flash-lite-image', provider=google_provider)
    google_image_generator = ImageGenerator(google_image_model, instrument=True)
    google_image_result = google_image_generator.generate_sync(
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

    uploaded_image = google_provider.client.files.upload(
        file=google_image_path,
        config={'mime_type': google_image_result.images[0].content.media_type},
    )
    if not uploaded_image.uri or not uploaded_image.name:
        raise RuntimeError('Google Files API upload did not return a file URI and name')

    try:
        google_edited_image_result = google_image_generator.generate_sync(
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
        google_provider.client.files.delete(name=uploaded_image.name)
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


if __name__ == '__main__':
    main()
