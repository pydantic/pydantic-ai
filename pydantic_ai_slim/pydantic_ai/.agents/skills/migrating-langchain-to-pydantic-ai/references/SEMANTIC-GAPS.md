# Semantic Gaps and Parity Spikes

Use this reference before choosing target primitives. A migration is safe only when each source guarantee is preserved, intentionally changed, or assigned to an explicit external owner.

## Contents

- [Build a semantic-gap register](#build-a-semantic-gap-register)
- [Prompts and message history](#prompts-and-message-history)
- [Tools, context, and schemas](#tools-context-and-schemas)
- [Structured output and run termination](#structured-output-and-run-termination)
- [State, checkpoints, and approval](#state-checkpoints-and-approval)
- [Middleware, retries, and limits](#middleware-retries-and-limits)
- [Streaming and event contracts](#streaming-and-event-contracts)
- [Concurrency and cancellation](#concurrency-and-cancellation)
- [Spike protocol](#spike-protocol)
- [Primary references](#primary-references)

## Build a semantic-gap register

Write this before target code:

| Source guarantee | Observed target behavior | Pydantic workaround | Outcome | Required probe | Residual / owner |
|---|---|---|---|---|---|
| checkpoint resumes after approval | deferred results start another agent run | deferred requests plus an authenticated durable pending-action record | `integration-required` | crash before/after result persistence | selected workflow store / application workflow |

Record two separate classifications. First, the behavior decision:

- `equivalent`: the tested versions preserve the same contract.
- `adapted`: an application adapter preserves the contract.
- `intentional-change`: product owners accept a documented change.
- `external-owner`: infrastructure outside the agent preserves it.
- `unproved`: block cutover or constrain the migrated slice.

Second, the workaround outcome: `validated-native`, `validated-adapter`, `integration-required`, or `blocker`. A source behavior can be non-equivalent while its workaround is validated; conversely, a promising native API remains unproved until its parity probe passes.

Never use `native` to mean “the names look similar.” Do not leave `redesign` as the proposed solution: follow the affected row into [Validated Workaround Recipes](WORKAROUND-RECIPES.md), implement the smallest viable construction, and classify it as `validated-native`, `validated-adapter`, `integration-required`, or `blocker`. Record installed versions and link each claim to a test, trace, source line, or official documentation.

## Prompts and message history

### Current instructions versus historical prompts

Current LangChain `create_agent` keeps its request system message outside graph state and prepends it at the model boundary; dynamic prompt middleware replaces that request field rather than checkpointing a `SystemMessage`. Custom or legacy graphs may instead store explicit system messages, so inspect the source and capture the actual model request. Pydantic AI distinguishes `instructions` from `system_prompt`:

- Current Pydantic AI `instructions` are sent for the current agent. Instructions stored on historical `ModelRequest` objects are not model-visible instructions on the next run.
- Dynamic `@agent.instructions` functions are reevaluated for every run, including a run with `message_history`.
- Pydantic AI `system_prompt` is the historical-prompt mechanism. With non-empty history, a new system prompt is not generated unless the chosen reinjection behavior does so.

Do not mechanically replace every LangChain prompt with `instructions` or use Pydantic AI `system_prompt` merely because the source used the phrase “system prompt.” Choose from observed model-boundary behavior:

| Required behavior | Pydantic AI choice |
|---|---|
| only the current agent's policy applies | `instructions` |
| historical prompts from earlier runs/agents remain authoritative | `system_prompt` plus tested history round-trip |
| UI/database history may omit the system prompt | sanitize history, then consider `ReinjectSystemPrompt` |
| per-request locale, tenant policy, or feature state | dynamic `instructions` from typed dependencies |

Probe the actual model boundary with `FunctionModel` and `AgentInfo.instructions`; inspecting serialized history alone is misleading because old request objects can still contain their original `instructions` field.

Multiple dynamic prompts also compose differently. LangChain `dynamic_prompt` middleware replaces the request's system message, and wrap middleware nests, so a later replacement can overwrite an earlier one. Pydantic AI composes registered instructions and orders stable static instructions before dynamic instructions. If the source has more than one prompt-mutating middleware, capture the exact final prompt and evaluation count; do not convert each entry to `@agent.instructions` and assume the concatenation is equivalent. When replacement is intentional, ordered `Hooks.on.before_model_request` processors can replace the latest request instructions; scrub those request-only instructions before persisting history if the source does not checkpoint them. Spike two dependency values and a continuation before adopting this adapter.

### Message conversion is a protocol migration

LangChain messages and Pydantic AI messages differ in tool-call IDs, tool-return pairing, reasoning/provider metadata, multimodal parts, timestamps, and serialization. Do not cast or rename them in place.

Define and test a converter only when history must be retained. Include:

- normal user/assistant turns;
- multiple parallel tool calls and out-of-order returns;
- interrupted or partially answered tool calls;
- provider reasoning/signature metadata that must round-trip;
- malformed or untrusted browser history;
- branch/fork conversation identity.

Pydantic AI can repair model history before a request, including dangling tool calls and orphaned tool results. LangGraph's `add_messages` instead merges messages by ID. Feed malformed and partially completed histories through both implementations and compare the actual fake-model input, not only stored JSON.

Prefer an application-owned conversation DTO at public boundaries. Keep provider/framework messages inside adapters.

## Tools, context, and schemas

`ToolRuntime.context` and `RunContext.deps` both inject runtime values, but parity depends on what the model can choose. Keep authenticated identity, credentials, clients, idempotency services, and policy evaluators out of the tool schema.

For every tool, diff the model-visible definitions from both implementations:

- name and description;
- parameter names, required fields, defaults, enums, nullability, and additional properties;
- injected parameters excluded from JSON schema;
- return content visible to the model versus application metadata;
- validation error text and whether it is retried;
- timeout, cancellation, idempotency, and side effects.

Pydantic AI's LangChain wrappers are transitional: LangChain retains argument validation and framework dependencies. A wrapped tool is not proof of native parity.

Audit source-only control flags before wrapping. For example, LangChain `return_direct=True` can end its agent immediately after the tool, while `tool_from_langchain` delegates the call without preserving that stop policy. If the model selects the call as a terminal action, make the callable a named output function with `ToolOutput`; a parity spike must prove the terminal value, one execution, and model-call count. This changes the callable from an ordinary optional tool into an output choice, so retain multiple output choices when the source has other terminal routes.

Sync Pydantic AI tools may run in worker threads while async tools run on the event loop. Test context propagation, cancellation, thread safety, and resource lifetimes when source tools relied on thread-local callbacks or event-loop affinity.

## Structured output and run termination

The same schema type does not imply the same wire strategy:

- LangChain may automatically choose `ProviderStrategy` when the selected model supports provider-native structured output, otherwise `ToolStrategy`.
- Pydantic AI uses tool output for a plain structured `output_type` by default. Select `NativeOutput`, `ToolOutput`, `PromptedOutput`, or `TextOutput` deliberately when wire behavior matters.
- LangChain returns structured output in agent state (commonly `structured_response`); Pydantic AI returns `result.output`. Preserve the application's public result DTO instead of exposing either shape.
- Pydantic AI unions that include `str` permit plain text to end the run. Exclude it when structured output is mandatory.

Characterize invalid schema retries, multiple output candidates, provider fallback, and a response that emits terminal output alongside function tools.

### `run()` and `run_stream()` are not interchangeable

Pydantic AI `run()` completes the agent graph. `run_stream()` commits the first matching streamed output immediately. When a response co-emits structured output and a function tool, this can change the final result even if the agent uses the same `end_strategy`.

A verified probe should make the co-emitted function tool raise `ModelRetry`:

- under `run()` with the default graceful strategy, the first output is rejected and the corrected later output wins;
- under `run_stream()`, the first matching output is already committed, so retry-after-tool-failure cannot replace it.

Use `run(event_stream_handler=...)`, `run_stream_events()`, or `iter()` when the application needs complete agent execution plus events. Test the chosen method; do not infer parity from `run()` tests.

## State, checkpoints, and approval

Split source state into four contracts:

1. run dependencies: typed services and immutable request context;
2. conversation: model messages;
3. workflow state: counters, plans, joins, pending actions, and domain progress;
4. long-term memory: cross-thread facts and retrieval indexes.

A LangGraph checkpointer stores graph state at super-step boundaries and may retain task-level pending writes, next nodes, checkpoint ancestry, namespaces, and thread identity. It supports behaviors such as fault recovery, history, replay, and fork. Pydantic AI `message_history` supplies model context; it does not provide those workflow guarantees.

### Interrupt and approval semantics differ

LangGraph resume restarts the interrupted node from its beginning. Code before `interrupt()` runs again; multiple interrupts are matched by position. Keep pre-interrupt effects idempotent or move them after the interrupt/separate node.

Pydantic AI deferred approval ends or pauses at a tool-call boundary. An external resume supplies `DeferredToolResults` keyed by tool-call ID together with the original history. Pydantic AI does not automatically persist the history, dependencies, authenticated principal, or application resume token. Approval is not an authorization boundary: never trust a browser/client-provided `DeferredToolResults` merely because its ID is well formed. Authenticate the reviewer and correlate the result server-side to an issued pending call, authorized principal/tenant, tool name, and approved original arguments or an explicitly authorized override.

Map these details explicitly:

| LangChain/LangGraph behavior | Migration decision |
|---|---|
| ordered decisions for an interrupt batch | persist and map each decision to a stable Pydantic tool-call ID |
| approve, edit, reject, or respond | map approve/override/deny; design a result path for human `respond` behavior |
| node restarts on resume | decide which pre-interrupt hooks/effects must rerun or must not rerun |
| checkpointer owns thread resume | choose application/durable-workflow storage and atomic correlation |
| replay may re-execute later nodes | retain idempotency keys and reconcile unknown-after-crash outcomes |

Prove denial executes the protected tool zero times. Prove approval executes it exactly once with the original authenticated dependencies, correlation, and either the original arguments or a separately authorized override. Add crash injection before persistence, after persistence, before the side effect, and after an unknown side-effect outcome.
Also forge an approval for another thread/tenant and an unknown or already-consumed tool-call ID; the server-side boundary must reject both before the agent or tool executes.

## Middleware, retries, and limits

LangChain middleware order is behavioral: before hooks run in list order, after hooks in reverse, and wrappers nest with the first middleware outermost. Middleware can update graph state, jump, short-circuit, or transform model/tool calls. A flat list of Pydantic AI hooks does not prove the same nesting.

Trace exact order on success, validation failure, tool failure, interrupt, resume, cancellation, and terminal output. Decide whether behavior belongs in a hook, capability, toolset wrapper, model wrapper, output validator, tool body, or application boundary.

Keep retry domains separate:

| Source policy | Why it is not 1:1 | Likely owner |
|---|---|---|
| LangGraph node `RetryPolicy` | reruns a node after selected exceptions/backoff | workflow or service retry |
| LangChain tool retry middleware | may turn tool errors into model-visible feedback | tool wrapper or `ModelRetry` only for model-correctable failures |
| provider/transport retry | repeats network requests without asking the model to change | model/provider transport |
| Pydantic AI output/tool retry | per-domain validation/model-correction budget | agent/tool/output configuration |

`ModelRetry` tells the model to try differently; do not use it as a generic transient-network retry. A repeated tool invocation may duplicate a side effect.

In particular, LangChain `ToolRetryMiddleware` can retry the same tool handler locally with the same request before returning to the model. Pydantic AI `@agent.tool(retries=N)` limits validation/`ModelRetry` feedback cycles that involve another model request. A direct translation can change model-call count, arguments, latency, and cost. Use a `Hooks.on.tool_execute` wrapper or service-client retry when the source guarantee is same-handler local retry; reserve `ModelRetry` for model-correctable failures. Spike a fail-once tool and count both model and tool calls.

Likewise, LangGraph `recursion_limit`, LangChain model-call middleware limits, Pydantic AI `request_limit`, and `tool_calls_limit` count different units and may fail differently. LangChain limit middleware can track a persisted thread total and may end with a synthetic assistant message; Pydantic AI run limits raise `UsageLimitExceeded` and reset unless the caller supplies accumulated usage. One verified Pydantic AI behavior is batch-atomic enforcement: if a parallel tool-call batch would exceed `tool_calls_limit`, none of that batch executes. Test boundary values, repeat across a persisted thread, and report the failure shape plus the unit each limit protects.

## Streaming and event contracts

LangGraph stream modes can expose full state, state updates, model messages/tokens, custom events, checkpoints, tasks, and debug data. Pydantic AI exposes model-part, tool, deferred, node, and final-result events through different APIs. Neither event grammar is a public-contract replacement for the other.

Define a versioned application event envelope with correlation, sequence, event kind, payload, and terminal/error semantics. Then adapt both implementations to it during shadowing. In Pydantic AI event streams, emit the public terminal event from `AgentRunResultEvent`, which represents completed execution, rather than treating an earlier output-selection event as completion.

Probe:

- ordering of token, tool-start, tool-result, approval, state-update, and final events;
- whether final means model output chosen or all side effects completed;
- cancellation and cleanup when the consumer disconnects;
- reconnect/resume cursor behavior and duplicate delivery;
- bounded buffering/backpressure for slow consumers;
- durable execution limitations, since some workflow integrations cannot stream in real time.

## Concurrency and cancellation

LangGraph parallel nodes execute within super-steps and merge through channel reducers. `Send` creates dynamic branches. Pydantic AI function tools are concurrent by default when the model emits multiple calls; `sequential=True` makes a tool a barrier, and a run-level execution mode can serialize tools.

LangGraph rejects conflicting same-step writes when a channel has no suitable reducer; annotated reducers define combination. Pydantic Graph parallel tasks can share mutable state, so a mechanical state-port can turn a deterministic reducer or conflict into a race and lost update. Prefer branch-local results and an explicit typed join/reducer; do not use shared dependencies as a mutable state channel. Carry a source index when visible order matters, sort at the join, and use `ReducerContext.cancel_sibling_tasks()` when the reducer's contract ends remaining branch work.

Do not replace graph fan-out with default tool concurrency. Preserve:

- maximum concurrency and rate limits;
- deterministic result order versus completion order;
- reducer/merge rules and duplicate keys;
- sibling cancellation when one branch fails;
- partial-result policy and retry scope;
- event order separately from execution order;
- side-effect isolation and compensation.

Spike a slow-first/fast-second pair. Prove overlap when allowed, barrier order when required, failure cancellation, and externally visible event ordering.

## Spike protocol

Use `TestModel` for schema/registration checks and `FunctionModel` for exact trajectories. Disable live model requests. Keep each spike small enough to explain one semantic claim.

1. Pin and print LangChain, LangGraph, and Pydantic AI versions.
2. Build the smallest source reproduction and run it before translating.
3. Capture model-visible instructions, tool definitions, messages, events, state/checkpoints, calls, and side effects.
4. Build the smallest target reproduction using the proposed primitive.
5. Feed both the same fixture and compare an application-owned trace.
6. Add failure injection and boundary values, not only the success path.
7. Promote the probe to a parity test or record why it is version-specific and temporary.

At minimum, spike these when present:

- two dynamic-instruction dependency values and a history continuation;
- invalid and multiple structured outputs;
- two source dynamic-prompt middleware entries versus composed Pydantic AI instructions;
- output co-emitted with a successful and retrying tool under the chosen stream API;
- a fail-once tool under LangChain retry middleware versus Pydantic AI retry feedback;
- oversized parallel tool-call batch;
- concurrent tools plus a sequential barrier;
- interrupt/deny/approve with crash and replay injection;
- same-step graph writes with and without a reducer;
- malformed history containing dangling calls, orphan results, and duplicate message IDs;
- a transitional `return_direct` tool and model-call count;
- per-run and persisted-thread limit exhaustion plus failure shape;
- middleware order across failure and resume;
- fan-out failure/cancellation and reducer order;
- event sequence plus consumer cancellation.

## Primary references

- LangChain: [middleware execution order](https://docs.langchain.com/oss/python/langchain/middleware/custom#execution-order), [structured output](https://docs.langchain.com/oss/python/langchain/structured-output), and [human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- LangGraph: [checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers), [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts), [fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance), and [streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- Pydantic AI: [instructions](https://pydantic.dev/docs/ai/core-concepts/agent/#instructions), [message history](https://pydantic.dev/docs/ai/core-concepts/message-history/), [structured output](https://pydantic.dev/docs/ai/output/), [parallel tools](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#parallel-tool-calls--concurrency), [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/), and [streaming](https://pydantic.dev/docs/ai/core-concepts/agent/#streaming)
