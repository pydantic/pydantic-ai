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

New capabilities land as a top-level submodule `pydantic_ai_harness/<name>/`
AND are re-exported **lazily** from the root `pydantic_ai_harness/__init__.py` —
top-level importability is the norm (ruled 8/13): every public capability class
(and public constants like `DEFAULT_ALLOWED_COMMANDS`) is added to `__all__`,
to the `TYPE_CHECKING` import block, and to the `__getattr__` dispatch. The lazy
`__getattr__` pattern is what makes this safe with optional dependencies:
importing the root package pulls in nothing, and accessing an extra-gated name
without its extra installed raises the standard install-hint `ImportError`.
`tests/test_placeholder.py::test_all_exports_are_importable` enforces the
contract — a new-capability PR that skips the root export fails it. The sole
exception is `experimental` (ACP), which stays submodule-only. Docs and examples
prefer the top-level import (`from pydantic_ai_harness import <Name>`).

### Naming Capabilities

Class names follow one rule with two branches:

1. **Noun, when the capability names a thing**: something the model can use --
   a tool, faculty, or integration (`Shell`, `Memory`, `PydanticAIDocs`); a
   subsystem with its own stores, types, or tools (`CodeMode`,
   `StepPersistence`, `RepoContext`, `ToolOutputLimits`); or a named strategy
   (`SlidingWindowCompaction`, `TieredCompaction`). Use the noun the feature's
   documentation would use.
2. **Imperative verb phrase, when the capability acts on the run and one verb
   phrase states its entire contract** (`ClearToolResults`,
   `DeduplicateFileReads`, `WarnNearLimits`, `WarnOnCacheBusts`). The name
   should read as a plain answer to "what will this do to my agent?". Do not
   wrap the action in a nominalization (`ToolResultEviction`): the noun form
   adds formality, not information. If the verb phrase would be incomplete or
   generic, the capability is a feature area -- use branch 1.

Additional constraints:

- The first word of a verb-phrase name must read unambiguously as a verb.
  `AuthorCapabilities` fails this ("author" reads as a noun first); pick a
  different verb or use a noun name.
- No abbreviations: `PydanticAIDocs`, not `PyaiDocs`.
- Name the behavior, not the problem it solves: `ToolOutputLimits`, not
  `OverflowingToolOutput`.
- If the natural noun already names a public type (usually the callable the
  capability wraps, like core's `HistoryProcessor`), the capability takes the
  verb phrase and the noun stays reserved for the type.
- Third-party integrations carry the vendor name (`ExaSearch`, `ModalSandbox`,
  `LocalStack`). Pydantic-family products (Logfire) do not need it.
- Ties go to the incumbent: many capabilities read acceptably in both forms.
  If an existing shipped name is not misleading, keep it.

Module naming: a single-capability module is the snake_case of its class
(`tool_output_limits`); a family of related capabilities gets a domain module
(`compaction`, `guardrails`); a vendor integration gets a vendor module (`exa`,
`logfire`). One module per capability, family, or vendor. Prefer a longer
descriptive name over a terse one (e.g. `tool_output_limits`, not `overflow`).
If you are unsure what to name a capability, ask the user (via the ask-user
tool) rather than guessing -- a name is a public commitment once shipped.

When a class or module is renamed, keep the old name working for at least one
release: a renamed module keeps a shim package at its old path, and a renamed
class keeps a module-level `__getattr__` alias, both emitting
`HarnessDeprecationWarning` via the helpers in `pydantic_ai_harness/_warn.py`.

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
- New remote-execution capabilities cap tool output with
  `max_output_bytes` / `max_output_lines` (the `modal_sandbox` names), not a new
  spelling. The released `max_output_chars` (shell) and `max_read_lines`
  (filesystem) predate this convention and stay for compatibility.
- Line offsets in model-facing file tools are 1-indexed, matching `grep -n`,
  editors, and stack traces (`modal_sandbox` is the reference; `filesystem` is
  0-based pending migration).

### Internal Agents Carry The Capability's Name

A capability that constructs its own `Agent` (a summarizer, an advisor, a
reminder generator) passes `name=` explicitly, set to the capability's
snake_case name: `Agent(model, name='summarizing_compaction', ...)`. That name
is the `agent_name` on the run span, which is what Logfire groups runs and
costs by.

Without it, core infers the name from the caller's frame locals. Inside a
capability method that lands on `agent`, `self`, or whatever a durability
wrapper happened to bind, so the run is unattributable in tracing. Matching
the capability name lets a reader trace spend straight back to the
capability without knowing its internals.

Test it through `instrument_all_agents` + `agent_run_names(capfire)` from
`tests/conftest.py`; per-agent `instrument=` on the outer agent does not reach
the inner one.

### Policy Lives In The Pluggable Component

When a capability takes a dependency behind a `Protocol` -- `PlanStore`,
`MemoryStore`, a client, a sandbox -- retry, fallback and degradation policy
belongs to the implementation of that protocol, not to fields on the capability.

The capability's job is to state what it does when the dependency fails, and to
keep the protocol small enough to wrap. A caller who wants
degraded-but-alive behavior writes a wrapping implementation; that is what the
protocol is for. The alternative is a capability that grows one `on_x_error=`
field per dependency and ends up owning behavior it cannot test end to end.

Worked example: `planning`'s tail reminder reads its `PlanStore` on every model
request, so a store that raises fails the run. `Planning` has no error-handling
knob; the README documents the behavior and sketches a wrapping store instead.

This is the same boundary as `core-boundary.md`, one level down: there, harness
does not reimplement core semantics; here, a capability does not reimplement its
dependency's operational policy.

## Composition Checks

Before treating a capability as done, check how it composes with:

- other capabilities in the same `Agent(..., capabilities=[...])`
- toolsets and wrapper toolsets
- `ToolSearch`
- deferred tools and approval flows
- provider-native versus local fallback tools
- streaming/event behavior when the capability emits or wraps events
- A stateful capability, or one that overrides `for_run`, must exercise its
  public `Agent` path with every supported durability capability. Verify that
  durable wrappers resolve the run-local toolset and state. If state cannot
  survive activity, process, or replay boundaries, fail before the first tool
  call and document the incompatibility.

`CodeMode` is a useful reference for wrapper-toolset composition, tool
selection, `ToolSearch` interaction, public docs, and test depth.

### Deciding What Two Of It Mean

Every capability answers "what if an agent has two of me?", and the answer is
the default `id` written in the class body -- there is no separate policy to
declare. Pydantic AI reads it: a class-declared `id` means two are one
configuration stated twice, and it merges them field by field; no default `id`
means two are two, kept apart under derived ids. `combine` only needs
overriding when the field-by-field merge cannot express the answer, such as a
budget that should take the *smaller* of two values.

Pick by asking what the capability's toolset registers:

- **Fixed tool names** (`read_file`, `run_command`, `write_memory`) --
  declare a default `id`, *unless* the capability's configuration is an access
  boundary. Two of them could never coexist anyway; they collided on the tool
  name. Declaring the id turns that late, confusing toolset error into a merge,
  which is what lets two packaged harnesses that each carry the capability
  compose.
- **Fixed tool names, but the configuration is an access boundary**
  (`FileSystem` allow/deny patterns, `Shell` command lists) -- the default merge
  is the wrong answer, because unioning two allow-lists *widens* what the agent
  can reach, and neither author asked for the other's paths. Either leave the
  capability anonymous, so two of it keep colliding on the tool name and the
  user has to choose, or declare a default `id` **and** write a `combine` that
  narrows rather than unions. Do not take the default and hope.
- **Names derived from a resource** -- derive the `id` from the same thing, the
  way `StackOne` uses `stackone-{account_id}` and Pydantic AI's `MCP` uses its
  URL. Two resources stay two capabilities; two of one resource collide, which
  is the right answer for a mistake.
- **Neither: hooks, guards, observers** -- no default `id`. Several per agent
  is the normal shape, and `CombinedCapability` chains them.

### An `id` Is Not A Namespace

A derived `id` keeps two capabilities apart. It does not keep their *tools*
apart, and the two are easy to conflate.

`StackOne(account_id='a')` and `StackOne(account_id='b')` get the ids
`stackone-a` and `stackone-b`, so both survive to the run -- but the tool names
come from StackOne's server, and two accounts on the same provider list the same
ones. The run then fails on the tool name, not the id:

```
UserError: StackOneToolset 'stackone-b' defines a tool whose name conflicts with
existing tool from StackOneToolset: 'bamboohr_list_employees'
```

The `id` is the capability's identity: it names the durable operation, keys the
instructions, and is what `load_capability` asks for. Namespacing tools is a
separate job, and Pydantic AI has a capability for it:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools

from pydantic_ai_harness import StackOne

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[
        PrefixTools(StackOne(account_id='hr-account'), prefix='hr'),
        PrefixTools(StackOne(account_id='crm-account'), prefix='crm'),
    ],
)
```

The model then sees `hr_bamboohr_list_employees` and
`crm_bamboohr_list_employees`, and each routes to its own account.

Record the choice in `tests/test_capability_combine.py`;
`test_every_capability_declares_a_combine_policy` fails until you do. Write the
reason from what the capability actually does, not from what would be
convenient: a reason like "one per rooted directory" is wrong if the toolset's
fixed tool names make two unreachable.

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
  compares the event's base and head commits against the capability's paths,
  and the heavy job runs only when a PR or branch push touched them. Run the
  heavy job unconditionally on tags so releases still exercise the live path:
  `if: always() && (github.ref_type == 'tag' || needs.changes.outputs.<name> == 'true')`.
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
