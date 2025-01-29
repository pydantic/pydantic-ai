"""This example shows how an agent can be used to analyze an image with Mistral's Pixtral model."""

import asyncio
import base64
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import UserPromptChunk
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.ollama import OllamaModel

model = MistralModel('pixtral-12b-2409')

pixtral_agent = Agent(
    model,
    result_type=str,
    system_prompt=(
        """
        Role: you are a helpful assistant that can help with image analysis

        Tasks:
            - Answer the question of the user about the image
            - Your answer will be given in Markdown format.
            - If there is a chart or graph in the image, extract the data and present it in a table.
        """
    ),
)

openai_agent = Agent(
    'openai:gpt-4o',
    result_type=str,
    system_prompt=(
        """
        Role: you are a helpful assistant that can help with image analysis

        Tasks:
            - Answer the question of the user about the image
            - Your answer will be given in Markdown format.
            - If there is a chart or graph in the image, extract the data and present it in a table.
        """
    ),
)

vision_model = OllamaModel(model_name='llama3.2-vision')

ollama_agent = Agent(
    vision_model,
    result_type=str,
    system_prompt=(
        """
        Role: you are a helpful assistant that can help with image analysis

        Tasks:
            - Answer the question of the user about the image
            - Your answer will be given in Markdown format.
            - If there is a chart or graph in the image, extract the data and present it in a table.
        """
    ),
)


# Create a function to encode the image to base64
def image_as_content(image_path: str) -> str | None:
    """Encode an image file to a base64 string.

    Args:
        image_path (str): Path to the image.

    Returns:
        str: Base64 encoded image content.
    """
    try:
        image_file = Path(image_path)

        if not image_file.exists():
            msg = f'The file {image_file} does not exist.'
            raise FileNotFoundError(msg)

        image_format = image_file.suffix.lstrip('.').lower()
        base64_image = base64.b64encode(image_file.read_bytes()).decode('utf-8')

    except Exception as e:
        print(f'Error: {e}')
        return None

    else:
        return f'data:image/{image_format};base64,{base64_image}'


async def main():
    # This is the user prompt with an image URL.
    user_prompt_1 = [
        UserPromptChunk(type='text', content='What is this image?'),
        UserPromptChunk(
            type='image_url',
            content='https://cdn.statcdn.com/Infographic/images/normal/30322.jpeg',
        ),
    ]

    # Running the pixtral agent with the user prompt 1.
    result_pixtral = await pixtral_agent.run(user_prompt_1)
    print(result_pixtral)

    # # Running the openai agent with the user prompt 1.
    result_openai = await openai_agent.run(user_prompt_1)
    print(result_openai)

    # This is the user prompt with an image encoded in base64.
    img_base64_content = image_as_content(
        'examples/pydantic_ai_examples/infography.jpeg'
    )

    if img_base64_content:
        # This is the user prompt that will be sent to the agent.
        user_prompt_2 = [
            UserPromptChunk(type='text', content='What is this image?'),
            UserPromptChunk(
                type='image_url',
                content=img_base64_content,
            ),
        ]

        # Running the pixtral agent with the user prompt 2.
        result_pixtral_2 = await pixtral_agent.run(user_prompt_2)
        print(result_pixtral_2)

        # # Running the openai agent with the user prompt 2.
        result_openai_2 = await openai_agent.run(user_prompt_2)
        print(result_openai_2)

        # # Running the ollama agent with the user prompt 2.
        result_ollama_2 = await ollama_agent.run(user_prompt_2)
        print(result_ollama_2)


if __name__ == '__main__':
    asyncio.run(main())
