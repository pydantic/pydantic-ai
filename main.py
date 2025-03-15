import base64
from io import BytesIO

from PIL import Image

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModelSettings

agent = Agent(
    model='google-gla:gemini-2.0-flash-exp',
    model_settings=GeminiModelSettings(response_modalities=('Image', 'Text')),
)
result = agent.run_sync(
    'Hi, can you create a 3d rendered image of a handsome potato? Animate it like the if it was from the 50s.'
)
base64_data = result.data.data
decoded_data = base64.b64decode(base64_data)
image = Image.open(BytesIO(decoded_data))
image.show()
