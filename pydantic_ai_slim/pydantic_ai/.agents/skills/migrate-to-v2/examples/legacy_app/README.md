# legacy_app

A multi-file v1.x application exercising the highest-value deprecations a real
migration encounters together — used by `scripts/validate.py` to assert the
union of warnings hits every important codemod, and used by graders to feed a
fresh migration attempt to a downstream agent.

Modules:

- `agents.py` — one `Agent(...)` with `instrument=`, `history_processors=`,
  `prepare_tools=`, `event_stream_handler=`, `tool_retries=`, `output_retries=`,
  `mcp_servers=` all set. This is the merged-`capabilities=[...]` migration
  test plus the `retries={...}` dict test in one place.
- `messages_legacy.py` — `vendor_details` / `vendor_id` / `call_id` field
  reads, plus the `Usage` class.
- `evals_setup.py` — `Dataset(...)` without `name=`.
- `main.py` — entry point; `run()` triggers everything.

The v2 golden is in `../modern_app/` with the same module layout.
