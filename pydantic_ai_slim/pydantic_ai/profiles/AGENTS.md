# profiles/ Guidelines

Profiles describe model-family capability facts and schema/request quirks.

- Put intrinsic model-family facts here: structured-output support/defaults, native tool support, thinking support, JSON schema transformation, return-schema support, prompted-output templates, and model-family quirks.
- Do not put provider client/auth behavior here; that belongs in providers or model adapters.
- Prefer explicit capability fields over scattered `isinstance` or provider-name checks.
- When a feature only applies to some models, model the support fact clearly and fail or degrade in the layer that owns the user-facing behavior.
- Keep profile merging and user overrides in mind; avoid assuming a complete concrete profile object when sparse profile data is allowed.
