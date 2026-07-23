# Capability Authoring

Harness capabilities should be small, composable batteries built on Pydantic AI
primitives.

## Choose The Abstraction

- Use `AbstractCapability` when the feature contributes instructions, model
  settings, toolsets, native tools, or lifecycle hooks.
- Use a `WrapperToolset` when the feature changes how an existing toolset is
  presented or called.
- Use a leaf `AbstractToolset` when the feature owns a new collection of tools.
- Use hooks when behavior belongs at a specific point in the agent lifecycle.
- Use capability ordering only when composition semantics require it. Keep the
  reason visible in the code or docstring.

If the feature changes provider wire behavior, normalized message structure,
tool execution semantics, output selection, or durable execution primitives, it
probably belongs in Pydantic AI core first.

## Public Shape

Each capability package should normally have:

- `__init__.py` with public exports
- `_capability.py` for the public capability class
- `_toolset.py` only if the capability needs toolset behavior
- `README.md` with focused usage docs (serves GitHub and PyPI)
- a unified-docs page at `docs/<capability>.md` (the `docs/` folder is flat --
  no `capabilities/` or `experimental/` subdirectories). It mirrors the README
  for the docs site, drops badges, links other harness pages with relative `.md`
  links and Pydantic AI docs with root-relative `/ai/...` links, links its
  source module, and -- where the capability exposes a public class -- may end
  with a `::: pydantic_ai_harness.<Class>` autodoc block. The README and this
  page are kept in sync (see `review-checklist.md` "Docs").
- mirrored tests under `tests/<capability>/`

The root `pydantic_ai_harness/__init__.py` should re-export stable public
capabilities. Keep implementation helpers private unless users need them.

### Capability Submodules And Exports

The `experimental` tier is retired. ACP is the sole remaining experimental
capability (`pydantic_ai_harness/experimental/acp/`); do not add new capabilities
there.

New capabilities land as a top-level submodule `pydantic_ai_harness/<name>/`.
They are not re-exported from the root `pydantic_ai_harness/__init__.py`: each
capability keeps its own optional dependencies, so importing the root package
must not pull in a capability's extras. Users import a capability from its
submodule (`from pydantic_ai_harness.<name> import ...`).

Naming: the module name is the capability name, one module per capability or
strategy. Prefer a longer descriptive name over a terse one (e.g.
`overflowing_tool_output`, not `overflow`). A known term is fine as-is (e.g.
`compaction`). If you are unsure what to name a capability, ask the user (via the
ask-user tool) rather than guessing -- a name is a public commitment once shipped.

When a capability's module path changes, keep the old path working as a
`DeprecationWarning` shim so existing imports do not break.

Top-level re-exports in `pydantic_ai_harness/__init__.py` (`CodeMode`,
`FileSystem`, `Shell`, `ManagedPrompt`) are the exception, not the rule. Once an
export has shipped in a published release it is a backward-compatibility
commitment: do not move, rename, or break it. Do not add new top-level
re-exports.

APIs are subject to change between releases; breaking changes ship deprecation
warnings where practical.

## API Design

- Prefer a small dataclass capability with typed fields.
- Name fields by the user concept, not the implementation mechanism.
- Accept the most generic useful input types.
- Avoid `Any` in new public signatures.
- Avoid casts. Fix the type shape instead.
- Keep defaults conservative and easy to explain.
- Do not add package dependencies without a clear issue and package-manager
  command.
- New remote-execution capabilities cap tool output with
  `max_output_bytes` / `max_output_lines` (the `modal_sandbox` names), not a new
  spelling. The released `max_output_chars` (shell) and `max_read_lines`
  (filesystem) predate this convention and stay for compatibility.
- Line offsets in model-facing file tools are 1-indexed, matching `grep -n`,
  editors, and stack traces (`modal_sandbox` is the reference; `filesystem` is
  0-based pending migration).

## Composition Checks

Before treating a capability as done, check how it composes with:

- other capabilities in the same `Agent(..., capabilities=[...])`
- toolsets and wrapper toolsets
- `ToolSearch`
- deferred tools and approval flows
- provider-native versus local fallback tools
- streaming/event behavior when the capability emits or wraps events
- durable execution when the capability affects tool calls, context,
  serialization, retries, or lifecycle ordering

`CodeMode` is a useful reference for wrapper-toolset composition, tool
selection, `ToolSearch` interaction, public docs, and test depth.

## CI And Dependency Footprint

Most capabilities add a package extra and a test module, which is cheap. Some
pull heavier machinery: a Docker image, an external service that needs a secret
or auth token, a large system binary (a cloud CLI), or live network calls. That
machinery makes CI slower and more failure-prone, and every unrelated PR would
otherwise pay for it on the critical path.

When a capability needs machinery of that weight:

- Keep its runtime dependency behind the capability's own extra, so importing the
  root package never pulls it in (see "Capability Submodules And Exports").
- Scope its expensive CI job to the capability. A `changes` job
  (`dorny/paths-filter`) reports whether the PR touched the capability's paths,
  and the heavy job runs only when it did. Run it unconditionally on `push`/tag
  so releases still exercise the live path:
  `if: always() && (github.event_name != 'pull_request' || needs.changes.outputs.<name> == 'true')`.
- Keep the aggregate `check` job green when the heavy job is skipped but red when
  it runs and fails. `re-actors/alls-green` with
  `allowed-skips: changes, <heavy-job>` does both: a skip does not block, a real
  failure still votes.
- Grant the `changes` job `pull-requests: read`. Without a checkout, paths-filter
  lists PR files through the GitHub API, which needs that scope. A public repo
  allows it under `contents: read`, but it fails on a private repo or under
  tightened default token scopes.
- Scope any secret to the step that needs it (not the job `env`) and bind the job
  to a CI `environment` that holds the secret, so checkout and setup steps never
  see it.

The `localstack` capability's `localstack-integration` job is the reference for
this shape. Whether the heavy job blocks merges (listed in `check`'s `needs`) or
only signals is the capability owner's call; state which in the PR.

## External-Service Assumptions And Refresh

A capability that wraps an external service, image, or CLI (`localstack`,
`modal_sandbox`, `exa`, `macroscope`) depends on facts that live outside this
repo and change on the vendor's schedule: auth requirements, version-gated
behavior, default ports, endpoints, wire formats. When one of those shifts the
capability can break in a way local tests miss (the emulator or API is mocked or
absent). Record the load-bearing assumptions so a future agent can refresh them
deliberately instead of rediscovering them from a failure.

For each such capability, keep a short block -- a module docstring or a comment
near the constants it pins -- that lists each external assumption with the date
it was last verified and a link to the authoritative source, and says how to
re-check it. Before changing auth, version, or protocol handling, re-verify the
relevant assumptions against those sources first. When you confirm one still
holds, bump its date; when it changed, update the code and the date together.

`localstack` is the worked example (verified 2026-07):

- The default `localstack/localstack` image requires `LOCALSTACK_AUTH_TOKEN` to
  start since LocalStack 2026.03.0; a pre-2026.03.0 tag (for example `4.x`) runs
  tokenless. Source: <https://docs.localstack.cloud/aws/getting-started/auth-token/>.
- Edge port `4566`; health at `/_localstack/health`; `localhost.localstack.cloud`
  resolves to `127.0.0.1` (needed for S3 subdomain-style addressing). Source:
  <https://docs.localstack.cloud/aws/capabilities/config/configuration/>.
- The AWS CLI accepts any unambiguous prefix of a global option (`--endpoint`
  for `--endpoint-url`) and the last value wins, so the command guard rejects
  prefixes of forbidden globals, not just exact names. Re-verify against the
  installed `aws` CLI if that guard changes.

## Docs

Each user-facing capability needs docs close to the code. Explain:

- what problem it solves
- minimal usage
- key options
- how it composes with relevant Pydantic AI features
- important safety or execution constraints

Keep examples runnable with the declared extras.
