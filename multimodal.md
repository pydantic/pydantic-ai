# Multimodal Support

## Kind

### Image

- **Claude** supports only base64 encoded images.
  - https://docs.anthropic.com/en/docs/build-with-claude/vision#example-multiple-images
- **Groq** supports both url and base64 encoded images.
  - https://console.groq.com/docs/vision
- **Mistral** supports both url and base64 encoded images.
  - https://docs.mistral.ai/capabilities/vision/
- **OpenAI** support both url and base64 encoded images.
  - https://platform.openai.com/docs/guides/vision

### Audio

- **OpenAI** supports base64 encoded audio.
  - https://platform.openai.com/docs/guides/audio?example=audio-in

### Video

- **VertexAI** supports urls.
  - https://docs.anthropic.com/en/docs/build-with-gemini/video-support

### Documents

#### PDF

- **Claude** supports only base64 encoded PDFs.
  - https://docs.anthropic.com/en/docs/build-with-claude/pdf-support

```python
@dataclass
class DocumentPart:  # more pdf

    data: str
    """The base64 encoded data of the document part."""

    media_type: Literal['application/pdf']
```
