# Plan: Unify Bedrock ARN / Inference Profile Support

## Context

**Issues**: #4694 (meta), #4001 (count_tokens), #4653 (embeddings)
**Supersedes PRs**: #4614, #4654 (will be closed)

When users use AWS Bedrock application inference profiles (ARNs), two things break:
1. `BedrockConverseModel.count_tokens()` fails — the `count_tokens` API doesn't accept ARNs
2. `BedrockEmbeddingModel` can't determine the handler (Titan/Cohere/Nova) from an ARN

Both existing PRs solve their respective problem in isolation. Douwe wants a unified solution: a `bedrock_inference_profile` field on settings that acts as the `modelId` override for API calls, so `model_name` can stay as the base model name for everything else (profile detection, count_tokens, handler selection).

**Key constraint**: The existing documented `profile=...` approach (pass ARN as model_name + explicit profile) must continue to work — this is purely additive.

## Approach

Add `bedrock_inference_profile` to both `BedrockModelSettings` and `BedrockEmbeddingSettings`. When set, it's used as `modelId` in API calls; the base `model_name` handles everything else.

## Changes

### 1. `BedrockModelSettings` — add field
**File**: `pydantic_ai_slim/pydantic_ai/models/bedrock.py` (after line 331)

Add `bedrock_inference_profile: str` to the TypedDict with a docstring explaining it overrides `modelId` in converse/converse_stream calls.

### 2. `BedrockConverseModel._messages_create` — use the field
**File**: `pydantic_ai_slim/pydantic_ai/models/bedrock.py` (line 586)

Change:
```python
'modelId': self.model_name,
```
to:
```python
'modelId': settings.get('bedrock_inference_profile') or self.model_name,
```

The `settings` variable is already available (line 581). `count_tokens` at line 432 already uses `remove_bedrock_geo_prefix(self.model_name)` — no change needed there.

### 3. `BedrockEmbeddingSettings` — add field
**File**: `pydantic_ai_slim/pydantic_ai/embeddings/bedrock.py` (after line 191, in the settings TypedDict)

Add `bedrock_inference_profile: str` with the same semantics.

### 4. `BedrockEmbeddingModel._invoke_model` — use the field
**File**: `pydantic_ai_slim/pydantic_ai/embeddings/bedrock.py` (line 631-659)

Add a `settings: BedrockEmbeddingSettings` parameter to `_invoke_model`. Use it to resolve modelId:
```python
model_id = settings.get('bedrock_inference_profile') or self._model_name
```

Update callers:
- `_embed_batch` (line 583): pass `settings` to `_invoke_model`
- `_embed_concurrent.embed_single` (line 611): pass `settings` to `_invoke_model`

Both already have `settings: BedrockEmbeddingSettings` in scope.

### 5. Update docs — recommend new approach
**File**: `docs/models/bedrock.md` (lines 287-315)

Restructure the "Using AWS Application Inference Profiles" section:
- Show the **new recommended approach first**: base model name + `bedrock_inference_profile` in settings
- Highlight that this keeps `count_tokens` working
- Then mention the existing `profile=...` approach as an alternative (still works, but count_tokens won't)

**File**: `docs/embeddings.md`

Add equivalent section for embeddings showing `bedrock_inference_profile` usage.

### 6. Tests — VCR cassettes with real inference profiles

**Steps 1-3 are done already (code + docs + stub tests). This step replaces the stub tests with VCR tests.**

#### 6a. Create AWS inference profiles

Using the `default` profile from `~/.aws/credentials` in `us-east-1`:

```bash
# Converse model: Nova Micro
aws bedrock create-inference-profile \
  --inference-profile-name pydantic-ai-test-nova-micro \
  --model-source '{"copyFrom": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0"}' \
  --region us-east-1

# Embedding model: Titan Embed Text v2
aws bedrock create-inference-profile \
  --inference-profile-name pydantic-ai-test-titan-embed \
  --model-source '{"copyFrom": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"}' \
  --region us-east-1
```

Save the returned ARNs for use in tests.

#### 6b. Replace stub tests with VCR tests

**File**: `tests/models/test_bedrock.py`

Remove `_CapturingBedrockClient`, `_CapturingBedrockProvider`, and the 3 stub tests. Replace with VCR tests:

- `test_bedrock_inference_profile_converse`: uses `bedrock_inference_profile` in settings with the Nova Micro inference profile ARN → asserts response has content + model_name is the base model
- `test_bedrock_inference_profile_count_tokens`: same settings → calls `count_tokens` with the base model name → asserts token count returned (proves count_tokens works when model_name is the base model even with inference profile set)

**File**: `tests/test_embeddings.py`

Remove the 2 mock-based inference profile tests. Replace with VCR test:

- `test_inference_profile_embed`: uses `bedrock_inference_profile` in settings with the Titan Embed inference profile ARN → asserts embeddings returned + model_name is the base model

#### 6c. Record cassettes

```bash
# Set AWS creds
export AWS_PROFILE=default
export AWS_REGION=us-east-1

# Record converse tests
uv run pytest tests/models/test_bedrock.py::test_bedrock_inference_profile_converse tests/models/test_bedrock.py::test_bedrock_inference_profile_count_tokens --record-mode=new_episodes -x

# Record embedding test
uv run pytest tests/test_embeddings.py::TestBedrock::test_inference_profile_embed --record-mode=new_episodes -x
```

#### 6d. Sanitize cassettes

After recording, review cassettes for any account IDs or sensitive data. The custom serializer already filters auth headers, but the inference profile ARN in the URI will contain the account ID — this is fine (it's in the cassette URL, same as existing tests).

## What we DON'T do

- No ARN validation / error on `__init__` — Douwe explicitly said the old `profile=...` way should still work, so we can't reject ARNs as model_name
- No `is_bedrock_arn()` helper — not needed without validation
- No changes to `BedrockModelProfile` or `BedrockProvider.model_profile()` — the inference profile is runtime config (settings), not a capability flag (profile)
- No changes to `count_tokens` — it already uses `remove_bedrock_geo_prefix(self.model_name)` which works correctly when model_name is the base model

## Verification

1. `make format && make lint` — passes (**done**)
2. `make typecheck` — passes (**done**)
3. VCR tests with real inference profiles verify end-to-end behavior
4. Existing VCR tests still pass (no regression) (**done**)
5. Doc example tests pass (**done**)

## Key files

- `pydantic_ai_slim/pydantic_ai/models/bedrock.py` — BedrockModelSettings + _messages_create
- `pydantic_ai_slim/pydantic_ai/embeddings/bedrock.py` — BedrockEmbeddingSettings + _invoke_model
- `docs/models/bedrock.md` — inference profile documentation
- `docs/embeddings.md` — embedding inference profile docs
- `tests/models/test_bedrock.py` — converse model tests
- `tests/test_embeddings.py` — embedding model tests
