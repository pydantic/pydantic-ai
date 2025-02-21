from pathlib import Path

from rich.pretty import pprint

from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent, ImageUrl

image_url = 'https://goo.gle/instrument-img'

agent = Agent(model='google-gla:gemini-2.0-flash-exp')

output = agent.run_sync(
    [
        "What's in the image?",
        ImageUrl(url=image_url),
    ]
)
pprint(output)
