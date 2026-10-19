from typing_extensions import TypedDict


class TranscriptionSettings(TypedDict, total=False):
    """Common settings for configuring transcription models.

    These settings apply across transcription model providers. Provider-specific settings classes (e.g.
    [`OpenAITranscriptionSettings`][pydantic_ai.transcription.openai.OpenAITranscriptionSettings])
    extend this with additional provider-prefixed options.
    """

    language: str
    """The language spoken in the audio, as an [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) code like `'en'` or `'de'`.

    Transcription models detect the language when it isn't given, but can guess wrong on short or
    accented clips; supplying it improves accuracy and latency.
    """

    temperature: float
    """The sampling temperature to use for the transcription.

    Lower values make the output more focused and deterministic.
    """

    extra_headers: dict[str, str]
    """Extra headers to send to the model.

    This follows the existing `ModelSettings` and `EmbeddingSettings` escape-hatch pattern.
    Prefer provider-prefixed typed settings when a setting is part of the supported public API.
    """

    extra_body: object
    """Extra body to send to the model.

    This follows the existing `ModelSettings` and `EmbeddingSettings` escape-hatch pattern.
    Prefer provider-prefixed typed settings when a setting is part of the supported public API.
    """


def merge_transcription_settings(
    base: TranscriptionSettings | None, overrides: TranscriptionSettings | None
) -> TranscriptionSettings | None:
    """Merge two sets of transcription settings, with overrides taking precedence.

    Args:
        base: Base settings (typically from the transcriber or model).
        overrides: Settings that should override the base (typically per-call settings).

    Returns:
        Merged settings, or `None` if both inputs are `None`.
    """
    # A shallow merge: an override replaces a whole value, `extra_headers` and `extra_body` included,
    # rather than being merged key by key. Matches `merge_model_settings`.
    if base and overrides:
        return base | overrides
    else:
        return base or overrides
