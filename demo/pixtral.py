from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.messages import UserPromptChunk

model = MistralModel('pixtral-12b-2409')

pixtral_agent = Agent(
    model,
    result_type=str,
    system_prompt=(
        "Role: you are a helpful assistant that can help with image analysis."
    ),
)

user_prompt = [
    UserPromptChunk(type="text", content="What is the image?"),
    UserPromptChunk(type="image_url", content="https://cdn.statcdn.com/Infographic/images/normal/30322.jpeg")
]

if __name__ == '__main__':
    result = pixtral_agent.run_sync(user_prompt)
    print(result)
