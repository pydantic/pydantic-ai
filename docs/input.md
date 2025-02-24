# Image & Audio Input

Nowadays, some of the LLMs are also able to understand audio and image content.

## Image Input

!!! info
    Some models don't support image input. Please check the model's documentation to see if it supports image input.

You can use [`ImageUrl`][pydantic_ai.ImageUrl] in case you have the URL in hands:

```py
from pydantic_ai import Agent, ImageUrl

image_url = ImageUrl(url='https://iili.io/3Hs4FMg.png')

agent = Agent(model='openai:gpt-4o')
result = agent.run_sync(['What company is this logo from?', image_url])
print(result.data)
# > This logo is from Pydantic, a Python library used for data validation and settings management using Python type annotations.
```

You can also use the `BinaryContent`, if you have it locally:

```py
import httpx

from pydantic_ai import Agent, BinaryContent

image_response = httpx.get('https://iili.io/3Hs4FMg.png')  # pydantic logo

agent = Agent(model='openai:gpt-4o')
result = agent.run_sync([
    'What company is this logo from?',
    BinaryContent(data=image_response.content, media_type='image/png')  # (1)!
])
print(result.data)
# > The logo is for Pydantic, a popular data validation and settings management library for Python.
```

## Audio Input

!!! info
    Some models don't support audio input. Please check the model's documentation to see if it supports audio input.


You can either use [`AudioUrl`][pydantic_ai.AudioUrl] or [`BinaryContent`][pydantic_ai.BinaryContent] to provide the audio input.

The use is analogous to the above.
