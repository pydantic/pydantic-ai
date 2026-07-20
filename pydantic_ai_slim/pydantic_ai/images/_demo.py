"""Manual end-to-end demo for direct image APIs and the `ImageGeneration` capability.

The three provider sections exercise `ImageGenerator` directly: generation, normalized results,
usage/instrumentation, and reference-image editing. The separate capability section compares the
subagent compatibility path with the direct image-model fallback while keeping the outer agent tool
contract identical.

Run every section with no flags, isolate direct providers with repeatable `--provider`, or run only
the bridge comparison with `--capability`.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

_THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)

import logfire

from pydantic_ai import Agent, BinaryImage, Embedder, ModelRequest, ToolReturnPart, UploadedFile
from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.images import ImageGenerationResult, ImageGenerator
from pydantic_ai.images.google import GoogleImageGenerationModel, GoogleImageGenerationSettings
from pydantic_ai.images.openai import OpenAIImageGenerationSettings
from pydantic_ai.images.xai import XaiImageGenerationModel, XaiImageGenerationSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.xai import XaiProvider

ProviderName = Literal['openai', 'google', 'xai']
_PROVIDER_NAMES: tuple[ProviderName, ...] = ('openai', 'google', 'xai')
_GENERATION_PROMPT = 'A cat with a cowboy hat, dancing in Rome.'
_MUTATION_PROMPT = 'Replace the cat with a dog while preserving the cowboy hat, dancing pose, and Rome setting.'


@dataclass(frozen=True)
class DemoSelection:
    """Sections selected from the command line."""

    providers: frozenset[ProviderName]
    capability: bool


def _save_first_image(result: ImageGenerationResult, filename: str) -> Path:
    """Persist the first normalized image and print only its local path."""
    path = _THIS_DIR / filename
    path.write_bytes(result.images[0].content.data)
    print('saved:', str(path))
    return path


def main() -> None:
    selection = _parse_selection()
    logfire.configure(send_to_logfire='if-token-present')

    asyncio.run(_run_instrumented_demo(selection))

    logfire.force_flush()


async def _run_instrumented_demo(selection: DemoSelection) -> None:
    with logfire.span('pydantic_ai_images_demo'):
        await run_demo(selection)


def _parse_selection(args: Sequence[str] | None = None) -> DemoSelection:
    parser = argparse.ArgumentParser(description='Run image provider and capability demos.')
    parser.add_argument(
        '-p',
        '--provider',
        action='append',
        choices=_PROVIDER_NAMES,
        dest='providers',
        help='Direct image provider to run. May be repeated.',
    )
    parser.add_argument(
        '--capability',
        action='store_true',
        help='Run the separate ImageGeneration capability comparison.',
    )
    parsed_args = parser.parse_args(args)
    providers = cast(list[ProviderName] | None, parsed_args.providers)
    capability = cast(bool, parsed_args.capability)

    # With no filters the demo remains comprehensive. Once a section is selected explicitly,
    # only that section runs, so `--provider openai` no longer also incurs the capability calls.
    if providers is None and not capability:
        return DemoSelection(providers=frozenset(_PROVIDER_NAMES), capability=True)
    return DemoSelection(providers=frozenset(providers or ()), capability=capability)


async def run_demo(selection: DemoSelection | None = None) -> None:
    selection = selection or DemoSelection(providers=frozenset(_PROVIDER_NAMES), capability=True)
    if 'openai' in selection.providers:
        await _run_openai_demo()
    if 'google' in selection.providers:
        await _run_google_demo()
    if 'xai' in selection.providers:
        await _run_xai_demo()
    if selection.capability:
        await _run_image_capability_demo()


async def _run_openai_demo() -> None:
    """Demonstrate OpenAI instrumentation, direct generation, and reference-image editing."""
    print('\n=== OpenAI direct image API ===')

    # These non-image requests deliberately create familiar embedding and agent spans next to the
    # image spans in Logfire. They make it easy to compare usage/cost presentation without mixing
    # either request into the ImageGenerator implementation being demonstrated below.
    model = OpenAIEmbeddingModel('text-embedding-3-small')
    embedder = Embedder(model, instrument=True)

    await embedder.embed_query('Hello from instrumented embeddings')

    agent = Agent(
        'openai:gpt-5.4-nano',
        instructions='Reply in one short sentence.',
    )
    agent.instrument = True
    await agent.run('Say hello from the agent instrumentation test.')

    image_settings = OpenAIImageGenerationSettings(openai_size='1024x1024', openai_quality='low', output_format='png')

    for image_model_name in ['gpt-image-1', 'gpt-image-1-mini', 'gpt-image-1.5']:
        image_generator = ImageGenerator(f'openai:{image_model_name}', instrument=True)
        image_result = await image_generator.generate(
            _GENERATION_PROMPT,
            settings=image_settings,
        )

        _save_first_image(image_result, f'_demo_output_{image_model_name}.png')

        if image_model_name == 'gpt-image-1.5':
            # The same direct API switches from generation to editing solely because `images` is
            # present. The mutation prompt should replace the cat while preserving the composition.
            edited_image_result = await image_generator.generate(
                _MUTATION_PROMPT,
                images=[image_result.images[0].content],
                settings=image_settings,
            )
            _save_first_image(
                edited_image_result,
                '_demo_output_gpt-image-1.5_edited.png',
            )


async def _run_google_demo() -> None:
    """Demonstrate Google generation and Files API editing through one provider client."""
    print('\n=== Google direct image API ===')
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
    google_image_path = _save_first_image(
        google_image_result,
        '_demo_output_gemini-3.1-flash-lite-image.png',
    )

    # Uploading the generated image demonstrates the common `UploadedFile` input without embedding
    # its bytes in the edit request. The `finally` block proves the provider file is lifecycle-owned
    # by the caller and is deleted even if generation fails.
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
    _save_first_image(
        google_edited_image_result,
        '_demo_output_gemini-3.1-flash-lite-image_edited.png',
    )


async def _run_xai_demo() -> None:
    """Demonstrate xAI generation and Files API editing."""
    print('\n=== xAI direct image API ===')
    xai_api_key = os.getenv('XAI_API_KEY')
    if not xai_api_key:
        raise RuntimeError('Set `XAI_API_KEY` to run the xAI image generation demo')

    xai_provider = XaiProvider(api_key=xai_api_key)
    xai_image_model = XaiImageGenerationModel('grok-imagine-image', provider=xai_provider)
    xai_image_generator = ImageGenerator(xai_image_model, instrument=True)
    xai_settings = XaiImageGenerationSettings(xai_aspect_ratio='1:1', xai_resolution='1k')
    xai_image_result = await xai_image_generator.generate(_GENERATION_PROMPT, settings=xai_settings)
    xai_image_path = _save_first_image(xai_image_result, '_demo_output_grok-imagine-image.jpeg')

    # xAI exposes a different Files API transport, but the public edit call receives the same
    # `UploadedFile` abstraction used by Google. Reusing the provider client keeps upload, edit, and
    # deletion in one async lifecycle.
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

    _save_first_image(
        xai_edited_image_result,
        '_demo_output_grok-imagine-image_edited.jpeg',
    )


async def _run_image_capability_demo() -> None:
    """Compare the compatibility fallback and the direct image-model fallback."""
    print('\n=== ImageGeneration capability bridge ===')

    # Both configurations expose the same `generate_image(prompt)` tool to the outer agent and use
    # the same normalized settings. The difference being demonstrated is what executes behind that
    # tool: `fallback_model` starts a conversational subagent, while `local` calls the dedicated
    # image API directly. Keeping the two calls side by side makes their Logfire traces and output
    # files directly comparable without coupling this comparison to the OpenAI provider demo.
    subagent_capability = ImageGeneration(
        native=False,
        fallback_model='openai-responses:gpt-5.4',
        output_format='png',
        quality='low',
        size='1024x1024',
    )
    await _run_capability_path(
        subagent_capability,
        label='compatibility subagent fallback',
        output_filename='_demo_output_capability_subagent.png',
    )

    direct_capability = ImageGeneration(
        native=False,
        local='openai:gpt-image-1.5',
        output_format='png',
        quality='low',
        size='1024x1024',
    )
    await _run_capability_path(
        direct_capability,
        label='direct image-model fallback',
        output_filename='_demo_output_capability_direct.png',
    )


async def _run_capability_path(capability: ImageGeneration[Any], *, label: str, output_filename: str) -> None:
    """Run one capability path and extract the image returned to the outer agent."""
    capability_agent = Agent(
        'openai:gpt-5.4-nano',
        capabilities=[capability],
        instructions='Call `generate_image` exactly once and pass the user prompt through unchanged.',
    )
    capability_agent.instrument = True
    capability_result = await capability_agent.run(_GENERATION_PROMPT)

    # The capability returns `BinaryImage` as a tool result. Reading it from message history proves
    # that both fallback implementations satisfy the same agent-facing contract, independently of
    # any final text the outer model chooses to produce after the tool call.
    capability_image = next(
        part.content
        for message in capability_result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and isinstance(part.content, BinaryImage)
    )
    capability_image_path = _THIS_DIR / output_filename
    capability_image_path.write_bytes(capability_image.data)

    print(f'{label} saved:', str(capability_image_path))


if __name__ == '__main__':
    main()
