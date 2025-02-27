# Example of Audio Transcription with OpenAIModel

This example shows how to use the `OpenAIModel` class from the `pydantic_ai.models.openai` module to transcribe an audio file using the OpenAI API.

## Example Code

```python
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


# Run the asynchronous function
asyncio.run(get_transcription())
```

## An example of a response could be
```
Lorem ipsum dolor sit amet consectetur adipiscing elit congue, ridiculous dapibus porttitor suspendisse torquent netus elementum ornare, eros quisque lacinia taciti imperdiet consequat magna. Lectus ad vel mollis pretium eget commodo donec velit, elementum phasellus libero himenaeos dictumst nascetur nibh penatibus egestas, proin various condimentum curabitur venenatis ultrices hendrerit. Iaculis etiam fringilla viverra porttitor cras tortor molestie dictum, tristique lacus tincidunt torquent ultrices natoque ad lectus nisl, facilisis phasellus quam nibh consequat quisque pulvinar.```
