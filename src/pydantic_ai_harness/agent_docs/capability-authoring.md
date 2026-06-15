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
- `README.md` with focused usage docs
- mirrored tests under `tests/<capability>/`

The root `pydantic_ai_harness/__init__.py` should re-export stable public
capabilities. Keep implementation helpers private unless users need them.

### Experimental Vs Released Exports

New capabilities land under `pydantic_ai_harness/experimental/`. Each
experimental package calls `warn_experimental('<name>')` in its `__init__.py` so
importing it emits `HarnessExperimentalWarning`. Anything under `experimental`
may change or be removed in any release, without a deprecation period.

Promote a capability to a top-level package and a top-level re-export in
`pydantic_ai_harness/__init__.py` only when its API is stable.

Top-level exports in `pydantic_ai_harness/__init__.py` are the intended public
surface. Once an export has shipped in a published release it is a
backward-compatibility commitment: do not move, rename, or break it. (`CodeMode`
is a shipped, released example.) Do not relocate an already-top-level capability
into `experimental`.

## API Design

- Prefer a small dataclass capability with typed fields.
- Name fields by the user concept, not the implementation mechanism.
- Accept the most generic useful input types.
- Avoid `Any` in new public signatures.
- Avoid casts. Fix the type shape instead.
- Keep defaults conservative and easy to explain.
- Do not add package dependencies without a clear issue and package-manager
  command.

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

## Docs

Each user-facing capability needs docs close to the code. Explain:

- what problem it solves
- minimal usage
- key options
- how it composes with relevant Pydantic AI features
- important safety or execution constraints

Keep examples runnable with the declared extras.
