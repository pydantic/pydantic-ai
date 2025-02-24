import httpx

from pydantic_ai import Agent, BinaryContent

response = httpx.get('https://iili.io/3Hs4FMg.png')
image_bytes = response.content

image_content = BinaryContent(data=image_bytes, media_type='image/png')

agent = Agent(model='openai:gpt-4o')
result = agent.run_sync(['What company is this logo from?', image_content])
print(result.data)
# > The logo is for Pydantic, a popular data validation and settings management library for Python.
