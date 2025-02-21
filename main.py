import requests
from rich.pretty import pprint

from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

image_path = 'https://goo.gle/instrument-img'
image = requests.get(image_path)

agent = Agent(model='google-gla:gemini-2.0-flash-exp')

# data = Path('docs/img/logfire-with-httpx.png').read_bytes()
# data2 = Path('docs/img/tree.png').read_bytes()

output = agent.run_sync(["What's in the image?", BinaryContent(data=image.content, media_type='image/jpeg')])
pprint(output)
