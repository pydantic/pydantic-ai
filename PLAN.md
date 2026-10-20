# Recover CI Review work through retained evidence

## Proposal and scope

Retain completed investigation, diagnose delegate failures, and recover missing work through a corrected operation, a useful saved conversation, or a fresh narrow assignment. Incomplete coverage is the last resort when recovery cannot complete. A successful exit is not evidence of a complete review.

This PR changes only `PLAN.md`. Implement only after agreement on this design and after public-engine migration [#10347](https://github.com/pydantic/pydantic-ai/pull/10347) merges. As checked on 2026-10-19, that PR remains open at [`3f4e3e76`](https://github.com/pydantic/pydantic-ai/tree/3f4e3e76a8a2df28247552f7d3e7725b8a142097). Reconcile its merged source before implementation.

Keep existing job, request, time, account and concurrency limits, models, token settings and provider retry policy. Add no investigation cutoff, replacement timer or retry allowance. Preserve existing admission/reserves: outstanding child-request reservations, the parent's final request tenth, and child timeout bounded by remaining run time minus the five-minute parent reserve. Preserve the prompt's final-20-percent guidance. These are source behavior, not new enforcement phases.

Two explicit policy changes are proposed: replace blanket redispatch prohibition with cause-specific recovery; replace wait-for-all with consumption of settled evidence and concurrent verification preparation. Neither permits a broad duplicate review or bypasses admission.

## Evidence and implementation boundary

Public incidents [#10259](https://github.com/pydantic/pydantic-ai/issues/10259), [#10309](https://github.com/pydantic/pydantic-ai/issues/10309) and [#10570](https://github.com/pydantic/pydantic-ai/issues/10570#issuecomment-6228556788) document run deadlines. Their [first](https://github.com/pydantic/pydantic-ai/actions/runs/38601315565) and [second](https://github.com/pydantic/pydantic-ai/actions/runs/38661230896) runs motivate separating delegate interruption from parent work after delegation. They predate [#10501](https://github.com/pydantic/pydantic-ai/pull/10501), merged as [`d7d15bf9`](https://github.com/pydantic/pydantic-ai/commit/d7d15bf9cc0c4a70e756fdf1929edd8c52e2a637). They establish neither failure of its prompt reserve nor the prevalence of repeated investigation.

References use migration head `3f4e3e76` for repository integration and current main [`5c8a6c12`](https://github.com/pydantic/pydantic-ai/tree/5c8a6c12cbdd9ea709d635b0d6e7a15cbb8d0ae7) for reusable harness mechanisms.

| Existing mechanism | Reuse and boundary |
|---|---|
| [Context preparation][context] and [review prompt][prompt] | Reuse annotated diffs, balanced disjoint baseline assignments, prior threads, reviewed/unreviewed lists and parent verification rules. Assignment proves no inspection. |
| [`cli.py :: task`, transport, compaction and output validation][cli] | Custom read-only `Task` returns output or error text; streamed tool results are truncated. Compaction already trims/summarizes history. Preserve reservations and usage debit in `finally`, including cancellation. |
| [`DelegationTasks` / `DelegationReports`][tasks] | Reuse stable identity, generation/run identity, captured interrupted messages, atomic history save, steering, explicit resume, untrusted reports and cancellation/draining. Custom `Task` does not currently use this opt-in owner. |
| [`StepPersistence`][steps] and [`inspect_recovery`][recovery] | Reuse snapshots, completed-call events and unresolved tool effects. Failed effects can have happened; persistence grants no replay permission. Memory capture cannot survive process loss. |
| [Typed delegation events][events] | Correlate identity, outcomes and usage. Text can be truncated; unmanaged refusal can emit no start event and propagating failure no end event. Missing events prove no completion. |
| [`WorkflowPolicy`, `build_agent`][policy] and [engine import][engine] | Coordinate around repository `Task`/policy composition. Keep public engine/CLAI canonical session and usage reporting. Migration retains `Coder(sub_agents=False)` and custom `Task`. |
| [Safe outputs][workflow] and [compiled host application][host] | Preserve read-only permissions, detection, host-owned writes, markers, head binding and deduplication. Nonempty output does not prove final review publication. |

Register a narrow managed-task adapter only in CI Review's repository agent composition. Keep shared `cli.task` unchanged for other workflows. Retain its runner/accounting and attach existing capture/persistence capabilities. Keep review semantics repository-local. Wholesale `SubAgents` adoption would change accounting: managed live shared ceilings differ from `Task` reservations/debit. Prove adapter compatibility first. A missing public primitive requires a separate API decision; add no generic scheduler or public checkpoint/review API.

## Failure matrix

**S** means a source path/guard exists, not measured frequency. **I** means public incident evidence. **H** means a review-specific hypothesis requiring reproduction and deployment evidence. Error text alone does not select recovery.

| Failure | Evidence | Diagnosis and recovery |
|---|---|---|
| Transient provider/transport failure | **S:** SDK retries and retryable-429 transport [CLI][cli] | Let existing retries act first. After exhaustion, distinguish provider timeout from child/run deadline. Retry a known idempotent operation only when dependency health and existing admission permit; no stacked blind loop. |
| Account quota/auth/shared outage | **S:** non-retryable account/quota codes and health classification [CLI][cli]; **H:** cause of a particular failure | Inspect structured cause and existing provider health. A fresh subagent on the same broken dependency cannot recover. Preserve cooldown/gates, credentials, models and limits; retained facts may still support local work. |
| Deterministic tool/config/permission error | **S:** MCP error handling [CLI][cli]; migration [path correction](https://github.com/pydantic/pydantic-ai/pull/10347) | Correct proven bad arguments/paths, then repeat only the safe failed operation. Do not restart arbitrary bugs, permission denials or missing configuration as transient. Surface uncorrectable causes. |
| Context loss/repeated unproductive investigation | **S:** compaction [CLI][cli]; **H:** resulting repetition/coverage loss | Compare retained evidence and unanswered questions. Steer/resume useful history; replace a contaminated/stalled conversation with accepted source evidence and residual work. |
| Isolated crash/child timeout | **S:** `task` error/timeout [CLI][cli], [managed lifecycle][tasks]; **I:** historical deadlines | Inspect cause, completed steps and pending effects. Correct a recoverable cause first. Resume useful history or replace missing work after settling/draining the prior child. Unknown crashes do not authorize replay. |
| Local child budget exhausted | **S:** child allowance [CLI][cli], [local/shared distinction][toolset] | Keep spent usage charged. Residual work uses only currently admissible parent headroom; local exhaustion does not imply shared exhaustion. No reset allowance. |
| Shared requests/time/account exhausted | **S:** [admission][cli], [run deadline][policy], host limits | Replacement cannot restore the exhausted resource. Prepare an honest terminal result from retained verified evidence if still possible. Existing run deadline remains terminal. |
| Refusal before start | **S:** [admission][cli], [event contracts][events] | Record refusal separately; claim no inspection. Reconsider only when existing reservations settle and admission genuinely changes. User/external refusal grants no bypass. |
| Malformed/incomplete successful output | **S:** string output and list validation [CLI][cli]/[prompt][prompt]; **H:** particular malformed report | Preserve valid evidence, reject invalid claims, request report repair from useful history or recover actual coverage gaps. Exit success proves neither coverage nor findings. |
| Verification failure | **S:** parent checks [prompt][prompt]; **H:** disputed candidate | Discard disproved claims; target reproducible unresolved questions without repeating baseline review. Failed reproduction cannot promote a finding. |
| External/user cancellation | **S:** [cancellation/draining][tasks], [deadline][policy], [host guard][host] | Propagate cancellation; do not resume/replace automatically. Distinguish child timeout. Saving is best effort; hard kill can prevent cleanup, artifacts and output. |
| Failed/ambiguous publication | **S:** [detection/host stages][host] | Distinguish prepared, queued, detected and applied output. Reconcile GitHub target/head/marker before repeating a possibly applied write. Never rerun investigation to repair publication. |

## Evidence and recovery contract

Keep run-scoped semantic records alongside managed tasks; reference existing histories/artifacts. A CI-Review-only progress tool submits inspection claims/candidates to the owner; step persistence supplies operation receipts. This tool grants no repository-write or publication access. The adapter records typed runtime causes before rendering parent-visible output. Preserve attempt history when the managed task's generation changes. Record:

- Repository, reviewed head, source/artifact version/hash, workflow run/attempt, parent, assignment, purpose/question and owner.
- Task/conversation/run/generation identity, predecessor attempt and recovery reason; refusal even before a task exists.
- Completed evidence with source locations and successful read/tool receipts; pending files/questions, candidates and uncertainty.
- Runtime failure cause/scope, unresolved effects, existing budget/admission state and publication stage.
- Separate untrusted claims, parent-accepted source facts and parent-verified findings. Only the parent promotes findings.

Validate head, assignment and attempt before accepting updates. Persist validated updates before acknowledgment. Serialize conflicting updates; late/duplicate delivery cannot overwrite accepted state. Missing/corrupt records leave coverage unknown. Retain range/truncation information. Baseline inspection needs the required diff/full source and the assigned reviewer's explicit completion claim; reads, summaries and assignment membership alone are insufficient.

Three distinct recovery operations:

1. **Retry an operation:** diagnose and correct/repeat an idempotent read/tool. Reconcile partial effects first, including publication. Exceptions do not prove nothing happened.
2. **Steer/resume:** continue useful history with accepted evidence and a concrete missing question. Steering reaches the next request; settled work needs explicit resume. Preserve original parent/shared accounting and current admission. Never use `allow_resume` to undo a user stop.
3. **Replace:** settle/drain the old child and preserve evidence. Give a fresh conversation accepted source facts, pending questions, relevant failed attempts and verification purpose. Do not reissue the broad review or treat prior conclusions as independent proof.

Progress means completed new coverage, a resolved concrete question or a reproducible verified finding. Activity, elapsed time and exit status are not progress. An unchanged dependency or repeated no-progress result requires diagnosis and a useful changed action. Recover while a justified action is feasible under existing admission; otherwise record why recovery cannot complete. Add no numeric retry/token/time caps.

## Parallel investigation and parent verification

Keep complete baseline coverage: every non-generated changed source file has exactly one baseline owner. Preserve generated-file treatment. Track concern assignments separately; overlapping checks name distinct questions and outputs. Use existing slots, without increasing concurrency or turning every file into an all-angles matrix.

For a streaming change, one owner can inspect changed implementation, another wire-format compatibility, while the parent prepares cancellation/reproduction checks. Share head-bound source artifacts and prior-thread facts; preserve independent judgments. A shared conclusion cannot satisfy both checks. If no slot is available, queue the question under existing admission.

For parallel CI Review assignments, launch managed background work and return task receipts instead of holding the parent tool batch open. Attach `DelegationReports` to deliver settled reports into the active parent. Keep the task owner inside the workspace/tool resource lifetime; drain all owned work before releasing resources. Existing slot/request reservations remain held until the child settles.

Consume settled reports while unrelated work continues. Validate coverage/evidence and prepare targeted verification immediately. For each surviving finding, the parent still inspects surrounding source, verifies trigger/reproduction and severity, checks false-positive rules/prior threads, and validates the commentable location. Reuse exact source artifacts/receipts instead of repeating all searches and baseline investigation. Before final output, require all assigned work/checks to settle or record recovery infeasibility, drain remaining tasks and mark gaps incomplete. The root needs this explicit join: managed reports alone do not join root children. Unresolved independent checks stay pending even when another reviewer agrees.

| Before | Proposed behavior |
|---|---|
| Delegate inspects `a.py`, then times out investigating `b.py`; only an error returns. | Retain `a.py` evidence/candidate status. Diagnose, then resume useful history or assign only unresolved `b.py` work under existing admission. |
| Delegate rereads files after compaction. | Restore accepted source references and unanswered question; steer or replace when history no longer helps. No new completed evidence means no progress. |
| Successful report omits `c.py`; parent waits for every other result. | Preserve valid results, recover `c.py`, and prepare verification of settled candidates concurrently. Approval still requires complete coverage/resolved required checks. |
| Account rejects requests, or GitHub may have accepted a review. | Diagnose shared unavailability or reconcile publication. Another broad delegate solves neither. |

## Tradeoffs and implementation sequence

| Choice | Benefit and cost |
|---|---|
| Managed-task adapter plus repository-local evidence | Reuses lifetime/history without another scheduler; must prove tool, accounting, cancellation and reporting compatibility. |
| Raw history alone | Useful supporting evidence, but reconstructing semantic coverage/trust from prose wastes work. Add only missing review records. |
| Resume versus replacement | Resume preserves context; replacement removes a bad trajectory but rebuilds context. Choose from saved progress and residual questions. |
| Shared source facts, independent concerns | Avoids repeated acquisition while protecting judgment; needs explicit purpose/ownership. |
| Existing reserves and cause-specific recovery | Improves useful work without changing limits. Cannot guarantee completion/publication before hard termination. |

After migration merge and design agreement:

1. Reconcile main, migration and published runtime. Reproduce source-observed failures and relevant hypotheses at the repository boundary. Prove the managed-task adapter before adding machinery.
2. Add evidence/attempt records using managed identity, capture, step persistence and effect inspection. Validate trust, interrupted updates and duplicate delivery.
3. Register the adapter only for CI Review; retain shared `cli.task` elsewhere. Add cause-specific repair/resume/replacement, background receipts, parent reports and the final join. Preserve reservations, spent usage, deadline, provider gates and resource ownership.
4. Update seed/managed review prompts together: replace blanket refusal and wait-for-all; retain baseline coverage, reserves and rigorous verification. Test concern ownership/concurrent preparation.
5. Validate final review against findings/coverage through existing safe outputs. Inline-only is insufficient. Preserve bot-author restrictions, 30-inline ceiling, markers, head binding and redundant-review/noop policy. Add no GitHub writer.
6. Run targeted [runner][tests], [headless][headless], delegation/output tests and workflow guard; recompile affected workflows only. Verify eligible deployed execution through host publication before comparing outcomes.

## Deterministic acceptance cases

Use in-process scripted models/tools, injected time, controlled events, temporary stores and fake host transport. Extend existing tests; avoid real waits and live providers. These are future implementation criteria, not implemented behavior.

| Case | Required result |
|---|---|
| Interrupted after completed inspection; crash before first request | Acknowledged evidence survives; no invented coverage. Diagnose before resume/replacement. |
| Retryable transport versus quota/auth/outage | Existing SDK/transport policy; no nested blind retry or fresh child on unchanged broken dependency. |
| Bad path versus denial/bug; unknown effects | Correct proven safe call only; deterministic uncorrectable cause stays explicit. Reconcile before repeating. |
| Useful history versus lost/repetitive context | Steer/resume useful history; replace contaminated history with residual work. Preserve completed evidence, no broad replay. |
| Local limit versus shared exhaustion; concurrent children | Preserve spent usage/reservations; recovery passes current admission. No reset or excess over original budgets. |
| Pre-start refusal; reservation later released | No false start/coverage; reconsider only after admission changes and missing work is feasible. |
| Malformed/omitted/stale/cross-assignment claims | Reject invalid claims, retain valid evidence, recover gaps; successful exit cannot yield approval. |
| Concurrent/late updates, corrupt store, head change | Isolate attempts, freeze publication input; stale updates cannot overwrite accepted facts. Unknown coverage cannot approve. |
| Baseline plus independent concern; early settled report | Receipts release the parent batch; reports reach the active parent. Prepare verification before unrelated work settles without higher concurrency; final output joins required work. |
| Disproved candidate or unfinished verification | Discard false positive; keep unresolved proof/checks pending and recover when feasible. No unverified findings. |
| Repeated activity without evidence progress | Diagnose/change next action; never reward elapsed time or blindly repeat. |
| Recovery infeasible, with/without verified HIGH/CRITICAL | Name all unreviewed files/required checks. Verified HIGH/CRITICAL requires `REQUEST_CHANGES`; otherwise `COMMENT`, never approval, subject to bot-author restrictions. |
| Inline-only sink, detection failure, ambiguous host write | Require final review or policy-valid noop; distinguish preparation/publication. Reconcile marker/head before retry; no investigation replay. |
| External cancellation/hard kill; unrelated callers | Propagate cancellation, no automatic recovery/publication guarantee. Ordinary `Agent`, other workflows and public APIs retain behavior. |

## Validation limits and agreement gate

Bounded telemetry study did not establish a comparable current-source post-#10501 CI Review execution through publication. Historical overlapping delegates and parent work after delegation do not measure duplicated coverage, its causes or concurrent-verification benefit. No speed, token, quota or quality improvement is established.

[Provider incident #10583](https://github.com/pydantic/pydantic-ai/issues/10583) closed on 2026-10-19; closure is not successful execution. Provider-gated/skipped reviews prove no execution. The smallest deployed check must join current-source eligibility, root/delegate execution and head-bound host publication. Establish normal/error/cancellation/fallback telemetry coverage; repair demonstrated gaps before comparing outcomes. Keep private measurements/payloads private. Distinguish observed tokens/API-equivalent cost from shared quota and actual charges.

Seek agreement on this contract and adapter before runtime work. Admission/reserve policy stays unchanged. If compatibility needs a policy/public API change, return with the concrete conflict. Hard cancellation, unavailable resources/provider and failed host writes can still prevent publishing an honest incomplete review.

[context]: https://github.com/pydantic/pydantic-ai/blob/3f4e3e76a8a2df28247552f7d3e7725b8a142097/scripts/gather-pydantic-ai-review-context.sh
[prompt]: https://github.com/pydantic/pydantic-ai/blob/3f4e3e76a8a2df28247552f7d3e7725b8a142097/.github/workflows/shared/prompts/pydantic-ai-pr-review.md
[cli]: https://github.com/pydantic/pydantic-ai/blob/3f4e3e76a8a2df28247552f7d3e7725b8a142097/.github/scripts/pydantic_ai_gh_aw_shim/cli.py
[tasks]: https://github.com/pydantic/pydantic-ai/blob/5c8a6c12cbdd9ea709d635b0d6e7a15cbb8d0ae7/src/pydantic_ai_harness/pydantic_ai_harness/subagents/_tasks.py
[toolset]: https://github.com/pydantic/pydantic-ai/blob/5c8a6c12cbdd9ea709d635b0d6e7a15cbb8d0ae7/src/pydantic_ai_harness/pydantic_ai_harness/subagents/_toolset.py
[events]: https://github.com/pydantic/pydantic-ai/blob/5c8a6c12cbdd9ea709d635b0d6e7a15cbb8d0ae7/src/pydantic_ai_harness/pydantic_ai_harness/subagents/_events.py
[steps]: https://github.com/pydantic/pydantic-ai/blob/5c8a6c12cbdd9ea709d635b0d6e7a15cbb8d0ae7/src/pydantic_ai_harness/pydantic_ai_harness/step_persistence/_capability.py
[recovery]: https://github.com/pydantic/pydantic-ai/blob/5c8a6c12cbdd9ea709d635b0d6e7a15cbb8d0ae7/src/pydantic_ai_harness/pydantic_ai_harness/step_persistence/recovery.py
[policy]: https://github.com/pydantic/pydantic-ai/blob/3f4e3e76a8a2df28247552f7d3e7725b8a142097/.github/scripts/pydantic_ai_gh_aw_shim/agent.py
[engine]: https://github.com/pydantic/pydantic-ai/blob/3f4e3e76a8a2df28247552f7d3e7725b8a142097/.github/workflows/shared/engine-zai.md
[workflow]: https://github.com/pydantic/pydantic-ai/blob/3f4e3e76a8a2df28247552f7d3e7725b8a142097/.github/workflows/pydantic-ai-pr-review.md
[host]: https://github.com/pydantic/pydantic-ai/blob/3f4e3e76a8a2df28247552f7d3e7725b8a142097/.github/workflows/pydantic-ai-pr-review.lock.yml
[tests]: https://github.com/pydantic/pydantic-ai/blob/3f4e3e76a8a2df28247552f7d3e7725b8a142097/.github/scripts/test_pydantic_ai_runner.py
[headless]: https://github.com/pydantic/pydantic-ai/blob/3f4e3e76a8a2df28247552f7d3e7725b8a142097/.github/scripts/test_workflow_agent.py
