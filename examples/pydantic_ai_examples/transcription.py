"""Example of a transcription of an audio file.

Run with:

    uv run -m pydantic_ai_examples.transcription
"""

import asyncio

from pydantic_ai.models.openai import OpenAIModel

# Initialize the OpenAI model with the desired model name
openai_model = OpenAIModel(model_name='whisper-1')


# Asynchronous function to get the transcription of the audio file
async def get_transcription():
    # Open the audio file in binary mode
    with open('/path/to/audio.mp3', 'rb') as audio_file:
        # Call the transcription method of the model
        transcription_text = await openai_model.transcription(audio_file)

    return transcription_text


if __name__ == '__main__':
    # Run the asynchronous function
    asyncio.run(get_transcription())
