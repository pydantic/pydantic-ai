# profiles/ Guidelines

Profiles describe model-family capability facts and schema/request quirks.

- Put intrinsic model-family facts here: structured-output support/defaults, native tool support, thinking support, JSON schema transformation, return-schema support, prompted-output templates, and model-family quirks.
- Do not put provider client/auth behavior here; that belongs in providers or model adapters.
- Prefer explicit capability fields over scattered `isinstance` or provider-name checks.
- Put a fact about one provider's API in that provider's `Provider.model_profile()`, never in a family profile function that other providers inherit. Precedent: `providers/anthropic.py::AnthropicProvider.model_profile` sets `supports_inline_system_prompts`, because the Bedrock Converse API and OpenAI-compatible gateways share `anthropic_model_profile()`.
- A fact that depends on the SDK client class belongs in the model's `_instance_profile_defaults()`, or in its `profile` property when the client can't send the feature at all, not here: `Provider.model_profile()` sees only the model name. See "Gate a capability on a profile flag" in [Provider Adapters](../../../agent_docs/provider-adapters.md).
- Express a difference between model families on one API as a `Literal`-valued field, such as `bedrock_thinking_variant`, so the adapter never checks a model-name prefix.
- Express the media a tool result accepts as the MIME types it accepts, as `google_supported_mime_types_in_tool_returns` does, so an adapter can decide per file.
- When a feature only applies to some models, model the support fact clearly and fail or degrade in the layer that owns the user-facing behavior.
- Keep profile merging and user overrides in mind; avoid assuming a complete concrete profile object when sparse profile data is allowed.
