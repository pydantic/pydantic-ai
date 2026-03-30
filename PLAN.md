# Design: First-class support for non-LLM model types (Image, Audio, etc.)

## Problem

The `ImageGeneration` capability's `fallback_model` currently requires a **conversational LLM** that happens to support image generation via the `ImageGenerationTool` builtin (e.g. `gpt-4o`, `gpt-5`). It does **not** work with dedicated image generation APIs like `gpt-image-1` because:

1. The subagent creates `Agent(model, output_type=BinaryImage)`, which requires the model profile to have `supports_image_output=True`
2. `gpt-image-1` is not recognized as supporting image output in `openai_model_profile()` — it matches none of the patterns (`gpt-5`, `o3`, `4.1`, `4o`)
3. Even if the profile was fixed, dedicated image APIs may not support the full conversational Agent loop (system prompts, tool calls, etc.)

This is confusing: users naturally expect `ImageGeneration(fallback_model='openai-responses:gpt-image-1')` to work.

## Current state

- `Model` is the only model base class, designed for conversational LLMs
- `EmbeddingModel` was added as a separate base class for embedding APIs (not subclassing `Model`)
- `KnownModelName` covers only LLM models; `KnownEmbeddingModelName` is separate
- Image generation only works as a builtin tool on LLMs, not as a standalone API

## Proposed direction

### Short-term (this PR scope)

- Document that `fallback_model` requires a conversational LLM with image generation support
- List correct model examples (`gpt-4o`, `gpt-5`, Google image models)
- Remove misleading `gpt-image-1` references

### Medium-term

- Add `gpt-image-1` to `openai_model_profile()` with `supports_image_output=True` (if it works via Responses API)
- If `gpt-image-1` doesn't support the full Agent loop, the subagent approach won't work — need a simpler path

### Long-term: `ImageModel` / `ImageGenerator` base class

Similar to how `EmbeddingModel` is a separate model type for embeddings, introduce:

```python
class ImageModel:
    """Base class for image generation APIs."""
    async def generate(self, prompt: str, **kwargs) -> BinaryImage: ...
```

With provider implementations:
```python
class OpenAIImageModel(ImageModel): ...  # wraps gpt-image-1 / DALL-E
class GoogleImageModel(ImageModel): ...  # wraps Imagen
```

And a `KnownImageModelName` type:
```python
KnownImageModelName = Literal[
    'openai-image:gpt-image-1',
    'google-image:imagen-3.0',
    ...
]
```

Then `fallback_model` could accept `Model | ImageModel | str`:
- If it's a `Model` (LLM): use current subagent approach
- If it's an `ImageModel`: call `model.generate(prompt)` directly (no Agent loop needed)
- If it's a string: infer from `KnownModelName` or `KnownImageModelName`

This mirrors the `EmbeddingModel` pattern and could extend to audio, video, etc.

### Even longer-term: unified model registry

```python
# Single infer function that routes based on model name prefix
model = infer_any_model('openai-image:gpt-image-1')  # -> OpenAIImageModel
model = infer_any_model('openai:gpt-4o')              # -> OpenAIChatModel
model = infer_any_model('openai-embed:text-embedding-3-small')  # -> OpenAIEmbeddingModel
```

Or a mapping-based approach:
```python
MODEL_TYPE_REGISTRY = {
    'openai': OpenAIChatModel,
    'openai-responses': OpenAIResponsesModel,
    'openai-image': OpenAIImageModel,
    'openai-embed': OpenAIEmbeddingModel,
    ...
}
```

## Questions to resolve

1. Should `ImageModel` be a new base class (like `EmbeddingModel`) or an adapter around existing `Model`?
2. Should `fallback_model` accept both `Model` and `ImageModel`, or should there be separate parameters?
3. How should the `ImageGenerationTool` config (quality, size, etc.) map to `ImageModel.generate()` kwargs?
4. Should this extend to other modalities (audio generation, video generation)?
5. How does this interact with the capability system — should `ImageGeneration` auto-detect the model type?
