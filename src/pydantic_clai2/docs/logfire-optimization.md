# Logfire agent optimization in CLAI

`/logfire optimize` is a small, advisory counterpart to Logfire's **Propose
improvement** flow. It lists agents, previews recent evidence, then reviews
that evidence with the current session model. It never changes code or writes
proposals to Logfire.

## Platform investigation

These paths refer to the Logfire platform source inspected for this change.
They are implementation references, not a stable public API contract.

- `src/walkthroughs/product-surfaces.toml`, surface
  `ai-engineering.agent-optimization`, points to `agents/run-baseline` and
  `scheduled-optimization/basic`.
- `src/walkthroughs/scheduled-optimization/README.md` describes the manual
  Optimize tab flow and recurring proposals, evidence thresholds, and optional
  Slack/webhook notifications. Applying an agent proposal only marks it applied;
  the user edits the agent's prompt in code.
- `src/packages/logfire-services/logfire_services/optimizer/agent_system_prompt.py`
  resolves agent runs and the current recorded system instructions. `shared.py`
  gathers bounded trace evidence, renders transcripts, prioritizes failures,
  and supports fetching more detail for selected evidence. `agent.py` records
  issues, success patterns, proposed solutions, and recommendations for problems
  a prompt cannot fix. Its prompt guidance favors small, evidence-backed edits
  that preserve successful behavior. It can return no change.
- `src/services/logfire-backend/logfire_backend/routes/ui_api/projects/agents.py`
  implements `GET /ui-api/organizations/{organization}/projects/{project}/agents/optimizer/evidence-preview/`
  and the WebSocket at
  `/ui-api/organizations/{organization}/projects/{project}/agents/optimizer/proposal/stream-ws/`.
  Preview accepts `agent_name`, `max_candidate_spans` (default 500),
  `lookback_minutes` (default 10080), and deployment environments. The WebSocket
  accepts an agent optimization request and returns progress/result/error frames.
  These use UI project-user authorization and AI feature checks, not the SDK
  device-flow credentials saved by CLAI. The WebSocket carries UI authorization
  in its subprotocol; it is not an HTTP proposal API.
- Frontend evidence-preview calls are in
  `src/services/logfire-frontend/src/packages/api/__generated__/agents/agents.ts`.
  Scheduled proposal history, apply, and dismiss routes are in
  `routes/ui_api/projects/optimization.py`.

No public optimizer/proposal endpoint was found that accepts the observability
sign-in credentials. CLAI deliberately avoids private UI authentication and
copying the platform's optimizer, scheduling, or persistence machinery.

## Public API path

Observability setup uses its existing device sign-in user token to create a
write token and additionally calls:

```text
POST /v1/organizations/{organization}/projects/{project}/read-tokens
Authorization: <device-flow user token>
{"description": "CLAI /logfire optimize"}
```

The route is in `routes/v1/sdk.py`; its `project:read_token` authorization also
supports API keys/OAuth through `require_sdk_or_api_project_auth` in
`routes/v1/auth.py`. The response returns the read token once. CLAI saves it
in `/keys` and discards the user token as before. Failure to create a read token
does not break tracing. Older project setups must run setup again.

Evidence uses `GET /v1/query` (`routes/v1/query.py`) with the saved project read
token in `Authorization`, `Accept: application/json`, and `sql`,
`min_timestamp`, `limit`, and `json_rows=true`. The response includes `rows`
and an `X-Logfire-Context` project label. A write token alone cannot query.

The query recognizes Pydantic AI `invoke_agent NAME` and `agent run` spans,
using `gen_ai.agent.name` or `agent_name`. It reads
`gen_ai.system_instructions`, `pydantic_ai.all_messages`, `final_result`, and
run status/error fields from `records`. It picks up to twelve runs in seven
days, failures first, and uses the newest recorded prompt in that sample.
Transcripts are bounded and thinking is omitted. All evidence stays in memory.

## Deliberate limits

This is a sampled review, not an evaluation or a guarantee of improvement.
It does not fetch child spans, group conversations, detect every failure signal,
or compare against a previous proposal. It currently supports named Pydantic AI
agents without spaces in their names. With different prompt versions in the
sample, findings may refer to older behavior. No recorded system prompt means
recommendations only, never an invented replacement.

`propose` sends recorded prompt/message content to the current session's model
and incurs its usual cost; credentials are never sent to the model. `preview`
only queries Logfire and shows an outcome summary. Keep sensitive content out
of instrumentation as needed. Server-side query permissions and limits apply.
A revoked read token is refused with instructions to run setup again.

The gate binds read access to the write-token **key name** saved by setup, as
observability's account tag does. Manually replacing the secret inside that key
is outside this binding: run setup again when switching projects. Disabling
observability removes the command. Resetting its project row clears the gate;
keys are left for the user to delete, and remote tokens can be revoked in Logfire.

A future public proposal API accepting project/user credentials would let CLAI
replace the local review while retaining the command and sign-in gate.
