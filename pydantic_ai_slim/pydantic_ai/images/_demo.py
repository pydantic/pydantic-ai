from __future__ import annotations

# ruff: noqa: E402
import sys
from decimal import Decimal
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)

import logfire

from pydantic_ai import Agent, Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.images import ImageGenerator
from pydantic_ai.images.openai import OpenAIImageGenerationSettings


def main() -> None:
    logfire.configure(send_to_logfire='if-token-present')

    with logfire.span('pydantic_ai_images_demo'):
        run_demo()

    logfire.force_flush()


def run_demo() -> None:
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
            'A cat with a cowboy hat, dancing in Rome.',
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
                'Replace the cat with a dog while preserving the cowboy hat, dancing pose, and Rome setting.',
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


if __name__ == '__main__':
    main()
