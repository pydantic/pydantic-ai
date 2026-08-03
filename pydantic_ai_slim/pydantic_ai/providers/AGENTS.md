# providers/ Guidelines

Providers own API clients, authentication, base URLs, HTTP lifecycle, and provider-level model/profile inference.

- Put provider API behavior in providers or concrete model adapters, not graph/tool/output code.
- Use provider/model profiles as the source of truth for capability facts that vary by provider or model family.
- Keep provider-specific settings typed and provider-prefixed so users can discover which API owns each field.
- Verify provider facts against upstream docs or SDK types before adding compatibility branches.
- Preserve normalized core behavior while retaining provider-specific metadata in structured metadata/provider-details fields.
