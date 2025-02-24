import httpx

from pydantic_ai import Agent, BinaryContent

image_response = httpx.get('https://iili.io/3Hs4FMg.png')

image_content = BinaryContent(data=image_response.content, media_type='image/png')

agent = Agent(model='openai:gpt-4o')
result = agent.run_sync(['What company is this logo from?', image_content])
print(result.data)
# > The logo is for Pydantic, a popular data validation and settings management library for Python.
