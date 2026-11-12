## This check

Review the UI adapter boundary in both directions, and CLAI 2's local web UI as its own boundary.

### UI adapters: every trust or disclosure decision is a named flag with a secure default

The user opts in to trusting client input or disclosing server state, never out. Precedents: `manage_system_prompt='server'`, `allowed_file_url_schemes={'http', 'https'}`, the reset of client `force_download='allow-local'`, `allow_uploaded_files` off by default, and `instructions` never sent to the client.

**Outbound, server to client.** Flag a field newly sent through `dump_messages`, a stream chunk, `UIMessage.metadata` or an AG-UI event that is not behind an opt-in flag. Sensitive fields: `provider_url`, `provider_name`, `provider_details`, `provider_response_id`, `instructions`, `run_id`, `conversation_id`, MCP server identifiers, and any identifier derived from a URL. Raw exception text, file paths, internal URLs, model configuration and usage detail are also sensitive. `timestamp` is safe.

**Inbound, client to server.** Flag a field newly read from client-submitted history without validation. Known-dangerous fields: `provider_response_id`, `instructions`, `force_download='allow-local'`, a forged `ToolCallPart` or `BuiltinToolCallPart` at the history tail, non-`http(s)` `FileUrl` schemes, stale reasoning signatures, and `run_id` or `conversation_id`.

**Client payload cost.** Client-submitted messages, parts and AG-UI frontend tool definitions are untrusted input. Flag work that grows super-linearly with them, including tool schemas that reach `_json_schema.py` reference inlining or the decision models.

**The chokepoint rule.** Inbound sanitization belongs in `UIAdapter.sanitize_messages`, not in one adapter. Flag validation added to only one adapter or outside `sanitize_messages`: the other adapter stays exposed.

### CLAI 2 local web UI

The per-launch token grants the local user control of that launch. Do not require UI-adapter flags or `sanitize_messages` here. Check that untrusted Markdown, model output, tool output and plugin content are sanitized before rendering, and that external links use safe opener and referrer behavior. Check that the token never reaches another origin, a log or another session. Require exact Host and token checks, and exact Origin and token checks for WebSockets. Check that session watch, input, queued events and reconnect state stay scoped to their session.

### Severity in this check

A field that newly crosses the adapter boundary in either direction without a secure-default flag is `high`. It is `critical` when it is exploitable in the default configuration.
