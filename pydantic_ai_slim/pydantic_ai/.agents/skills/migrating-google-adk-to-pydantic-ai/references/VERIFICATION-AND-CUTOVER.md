# Verification and Cutover

Verify through the application's supported caller boundary. Mocked models are useful for deterministic control-flow checks; they do not prove provider, storage, streaming, or deployment parity.

## Contract ledger

Keep one row per observed contract:

| Contract | Source evidence | Target check | Status | Residual risk |
|---|---|---|---|---|
| Public input/output/error | Existing test or captured fixture | Same boundary assertion | preserved / changed / unverified | What remains |

Use these statuses precisely:

- `preserved`: an executable check compares the relevant source and target behavior;
- `changed`: the user accepted a stated difference and impact;
- `not applicable`: the active slice does not use the feature;
- `unverified`: evidence is missing, so the contract is unfinished and cutover cannot proceed;
- `blocked`: a requested contract cannot be completed without a decision or external dependency.

## Proportionate checks

For every migration, check:

- public sync/async entrypoint and parameter names actually used;
- final output or structured validation behavior;
- tool names, schemas, errors, and externally visible side effects;
- dependency resolution and imports from a clean environment;
- no `google-adk` import remains inside the migrated slice; imports elsewhere are expected until cutover.

Add these checks only when the source path uses them:

| Feature | Required observation |
|---|---|
| Multi-turn session | Persist, reload, continue, and isolate users/sessions |
| Scoped state | Session/user/app/temp lifetimes and concurrent writes |
| Memory | Ingest, retrieve, tenant scope, update/delete policy |
| Artifacts | Versioning, latest lookup, scope, size/errors, cleanup |
| Graph/dynamic workflow | Typed hand-off, routes, joins, loops, ordering, partial failure |
| Parallel work | Isolation, deterministic merge, cancellation, write conflicts |
| Resume/durability | Kill at each checkpoint, restart, replay, idempotent effects |
| Approval/input/auth | Correlation, approve/deny, timeout, duplicate response, identity |
| Callbacks/plugins | Registration/capability order, short-circuit, mutation, errors, cleanup |
| Streaming/events | Serialized event sequence, finalization, cancellation, state commit timing |
| MCP | Server lifecycle, filtering/naming, auth, errors, reconnect |
| Skills/scripts | Discovery, instruction/resource loading, execution permissions |
| Code execution | Filesystem/network containment, secrets, timeout, persistence |
| Realtime/A2A | Protocol-level fixture or live end-to-end check |

## Cutover

1. Resolve the project from a clean environment and probe imports for the selected Pydantic AI and provider packages plus the separate `pydantic-ai-harness` distribution and its selected extras when used.
2. Run source characterization tests and target parity tests. Record whether each is offline, recorded-provider, real-service, or operational evidence.
3. Exercise restart/replay before traffic moves when the source promised resumability. A normal second run is not a restart test.
4. Keep compatibility adapters until all callers consume the new message/event/output shapes.
5. Do not move traffic or remove ADK while any contract is `unverified` or `blocked`.
6. Remove ADK services and deployment configuration only after no retained endpoint, evaluation, session migration, or rollback path needs them.
7. Report accepted `changed` contracts and their impact. Do not call a migration complete based only on passing test counts.
