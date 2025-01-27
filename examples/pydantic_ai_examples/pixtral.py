"""This example shows how an agent can be used to analyze an image with Mistral's Pixtral model."""

import asyncio
import base64

from pydantic_ai import Agent
from pydantic_ai.messages import UserPromptChunk
from pydantic_ai.models.mistral import MistralModel

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


# Create a function to encode the image to base64
def encode_image(image_path: str) -> str | None:
    """Encode the image to base64."""
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f'Error: The file {image_path} was not found.')
        return None
    except Exception as e:  # Added general exception handling
        print(f'Error: {e}')
        return None


async def main():
    # This is the user prompt with an image URL.
    user_prompt_1 = [
        UserPromptChunk(type='text', content='What is the image?'),
        UserPromptChunk(
            type='image_url',
            content='https://cdn.statcdn.com/Infographic/images/normal/30322.jpeg',
        ),
    ]

    result = await pixtral_agent.run(user_prompt_1)
    print(result)

    # This is the user prompt with an image encoded in base64.
    base64_image = encode_image('examples/pydantic_ai_examples/infography.jpeg')

    # This is the user prompt that will be sent to the agent.
    user_prompt_2 = [
        UserPromptChunk(type='text', content='What is the image?'),
        UserPromptChunk(
            type='image_url', content=f'data:image/jpeg;base64,{base64_image}'
        ),
    ]
    result_2 = await pixtral_agent.run(user_prompt_2)
    print(result_2)


if __name__ == '__main__':
    asyncio.run(main())
