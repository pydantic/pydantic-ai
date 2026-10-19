# Finish CI Review within its existing budget

## Proposal

Retain useful delegate evidence as work progresses, and give repository CI Review an enforced finalization phase. A delegate timeout should leave the parent with candidate findings and an honest coverage record. Stop broad investigation early enough to verify consequential candidates and prepare a review within the existing run budget.

This is a design proposal. This PR changes only this file. Implementation follows human agreement on the decisions below; the plan can then be removed.

## Evidence and existing behavior

The public incident runs show two separate failure paths:

- [#10259](https://github.com/pydantic/pydantic-ai/issues/10259): [run 38601315565](https://github.com/pydantic/pydantic-ai/actions/runs/38601315565) logged three delegate timeouts and returned only timeout errors to the parent, which later reached its own 28-minute cutoff.
- [#10309](https://github.com/pydantic/pydantic-ai/issues/10309): [run 38661230896](https://github.com/pydantic/pydantic-ai/actions/runs/38661230896) logged all three delegate returns before the parent still reached its 28-minute cutoff. Retaining delegate results alone does not solve parent completion.
- [#10570](https://github.com/pydantic/pydantic-ai/issues/10570#issuecomment-6228556788) records another typed run deadline after overlapping delegate assignments. These incidents predate the merged safeguards; they demonstrate failure modes, not a measured regression after those safeguards.

[#10501](https://github.com/pydantic/pydantic-ai/pull/10501), merged as [`d7d15bf9`](https://github.com/pydantic/pydantic-ai/commit/d7d15bf9cc0c4a70e756fdf1929edd8c52e2a637), already introduced disjoint balanced assignments, explicit reviewed/unreviewed lists, wrapped provider-timeout classification, and a prompt-level final-20-percent reserve. Preserve those changes.

At source revision [`62b7bf0a`](https://github.com/pydantic/pydantic-ai/tree/62b7bf0aeac9a03066263469e650f40957d84cae):

- [`scripts/gather-pydantic-ai-review-context.sh`](https://github.com/pydantic/pydantic-ai/blob/62b7bf0aeac9a03066263469e650f40957d84cae/scripts/gather-pydantic-ai-review-context.sh) prepares assignments and annotated diffs. Assignment is not evidence of inspection.
- [`cli.py :: task()`](https://github.com/pydantic/pydantic-ai/blob/62b7bf0aeac9a03066263469e650f40957d84cae/.github/scripts/pydantic_ai_gh_aw_shim/cli.py) runs a read-only delegate under a timeout, returning final output or an error string. It does not return a structured partial review on failure. Its stream handler emits tool activity but truncates live tool-result text; logs are not a reliable candidate store.
- In the same file, `_run_timeout_secs()` derives the effective agent budget from the job timeout minus teardown headroom. `_run_with_timeout()` cancels at that deadline and emits a typed error. `task()` enforces an existing five-minute parent reserve; `time_budget_notice()` advises the parent to finish. Neither stops parent investigation at the prompt's 20% boundary.
- `require_safe_output()` retries an empty sink. Outside Attention, an inline comment alone can satisfy its non-emptiness check; this does not prove a final review exists.
- The [review prompt](https://github.com/pydantic/pydantic-ai/blob/62b7bf0aeac9a03066263469e650f40957d84cae/.github/workflows/shared/prompts/pydantic-ai-pr-review.md) requires parent verification and forbids restarting broad coverage after delegate failure. The [workflow](https://github.com/pydantic/pydantic-ai/blob/62b7bf0aeac9a03066263469e650f40957d84cae/.github/workflows/pydantic-ai-pr-review.md) gives the agent read-only GitHub permissions. The [compiled host job](https://github.com/pydantic/pydantic-ai/blob/62b7bf0aeac9a03066263469e650f40957d84cae/.github/workflows/pydantic-ai-pr-review.lock.yml) applies safe outputs after detection and does not run on workflow cancellation.

## Established invariants

- Verified HIGH or CRITICAL findings require `REQUEST_CHANGES`, even with incomplete coverage. Without such a finding, incomplete coverage requires `COMMENT`, never approval. Complete coverage follows the existing severity mapping.
- Delegate content remains untrusted data. Retention, valid syntax and completed reads do not verify a finding. The parent checks the trigger, surrounding code, false-positive rules, prior threads and commentable location.
- Name every unreviewed changed file in the review body. Omitted or invalid coverage is incomplete. Preserve existing generated-file treatment.
- Preserve bot-author verdict restrictions, the 30-inline-comment ceiling, head binding, review markers and safe-output permission/detection boundaries. Preserve the redundant-review/noop path.
- Keep existing run/request budgets, model, token settings, provider limits and account concurrency. Do not replay the whole review after a timeout or ambiguous publication failure.
- Scope enforcement to repository CI Review. Ordinary `Agent` runs, other workflows and public library APIs retain their behavior.

## Reuse before adding machinery

| Existing mechanism | Contribution | Boundary |
|---|---|---|
| Prepared assignments, diffs and review instructions | Assignment membership and verification inputs | Do not create another partition or copy severity policy |
| `capture_run_messages`, tool events and read-only toolsets | Completed-call evidence and interrupted history | An attempted read or narrative does not prove inspection |
| [`DelegationTasks` and `DelegationReports`](https://github.com/pydantic/pydantic-ai/blob/62b7bf0aeac9a03066263469e650f40957d84cae/src/pydantic_ai_harness/pydantic_ai_harness/subagents/README.md#managed-delegation-sessions) | Stable identity, cancellation/draining, atomic history persistence and explicitly untrusted reports | These are opt-in managed `SubAgents` features; custom `Task` does not currently use that owner |
| Budget derivation and request admission | One effective deadline and shared accounting | Finalization cannot reset either budget |
| Safe-output tools, detection, host application and markers | Validated transport, GitHub writes and deduplication | Prepared output is not proof of publication |
| Migration's repository `WorkflowPolicy` | Application-specific lifecycle and terminal health metadata | Reuse this integration point; do not fork the public engine |

Managed delegation already [captures interrupted messages and saves task state](https://github.com/pydantic/pydantic-ai/blob/62b7bf0aeac9a03066263469e650f40957d84cae/src/pydantic_ai_harness/pydantic_ai_harness/subagents/_tasks.py). Evaluate a narrow adapter before adding another child-lifetime store. Raw history still needs CI Review interpretation. Legacy [`TodoWrite`](https://github.com/pydantic/pydantic-ai/blob/62b7bf0aeac9a03066263469e650f40957d84cae/.github/scripts/pydantic_ai_gh_aw_shim/todo_write.py) retains no state across calls and is not an evidence store.

This favors deterministic workflow checks over a longer deadline prompt. Matt Pocock and Robert C. Martin discuss tool constraints and the limits of prompt steering in [their agent-practice discussion](https://www.youtube.com/watch?v=zcLPGC-tvgk). That is design guidance; repository source and incident evidence establish this problem.

## Recommended design

### Retain progress outside the final delegate response

Keep run-scoped records owned by the CI Review supervisor. Bind records to repository, PR head, run/attempt, assignment and delegate invocation. A narrow typed tool lets delegates report progress without repository writes, permission to mark findings verified, or publication access. Commit each validated update to supervisor-owned state before acknowledging it. Preserve completed updates across delegate cancellation, with atomic local snapshots where the existing task owner supports them. Do not reuse evidence across runs or resume automatically.

Progress contains:

- Candidate identity, changed-file location, concrete suspected trigger, severity claim and evidence references or remaining uncertainty.
- Separate completed/pending coverage claims.
- Runtime-owned invocation status: running, complete, timed out, refused, failed or cancelled.

Validate assignment membership, path containment, disjoint coverage, identity and update order at ingestion. Reject malformed, stale and cross-assignment data. Bound record sizes as data validation without adding model token caps. Serialize concurrent updates per invocation and freeze a consistent snapshot at finalization.

Report a candidate before attempting its full proof. A timeout can then return its error/status plus the last accepted progress to the parent. Retained history can support a targeted check when no candidate was reported, but partial reasoning never becomes a finding automatically. A thought never emitted cannot be recovered.

Count a file as reviewed only with both runtime evidence that its diff and full source were successfully read and a delegate claim that inspection finished, for the same head and assignment. Reads are necessary, not sufficient. Failed, truncated or partial-range reads require evidence that the missing content was read. Missing updates remain unreviewed; do not infer coverage from time spent or successful delegate exit.

Parent verification creates a separate accepted-finding record with evidence and disposition. Only the parent can promote a candidate. Late delegate updates cannot overwrite accepted findings or frozen publication state. A lost or unavailable store degrades coverage and is reported; it cannot produce approval.

### Enforce finalization within the existing deadline

Use the existing monotonic deadline and effective agent budget. Recommended investigation cutoff: the earlier of the existing delegate-reserve boundary and the start of the final 20% of that budget. Preserve teardown headroom outside the agent budget. The internal verification/publication split remains a design choice; neither adds time.

The supervisor owns a one-way transition:

1. **Investigate:** admit the existing disjoint delegates and normal parent inspection. Check admission at execution as well as advertisement, including queued calls.
2. **Stop investigation:** close admission, cancel and drain owned investigation/delegate tasks, and retain completed updates. Include in-flight model, shell and compaction calls; their ordinary timeouts must not consume the reserve. Keep publisher resources alive. Reject updates racing after snapshot freeze. Cleanup shares the absolute deadline. If cleanup cannot finish, report failure rather than start conflicting work or claim completion.
3. **Verify and prepare:** provide retained candidates, accepted findings and missing coverage to the parent. Admit only targeted verification of recorded candidates, with an internal cutoff that leaves output time. No replacement delegates or broad second review. Stop unfinished verification at that cutoff; leave candidates unverified and mark affected coverage incomplete.
4. **Publish:** disable investigation tools and assemble the review from accepted findings and coverage through existing safe outputs. If requests are exhausted or the provider is unavailable, prepare an honest incomplete `COMMENT` without another model call, or `REQUEST_CHANGES` for already verified blocking findings. Include verified finding text in the review body if inline preparation cannot finish. Never invent proof for an unfinished candidate.

The timer interrupts in-flight investigation instead of waiting for another model turn. Keep retained state outside the cancelled task. A model-assisted finalizer uses only remaining shared requests and time, without a fresh allowance. Early normal completion enters the same final output validation.

For CI Review, require a final review or policy-valid redundant noop; an inline-only sink is incomplete. Reject approval with unknown coverage and reject a non-blocking verdict with accepted blocking findings, subject to existing bot-author restrictions. Proposed refinement: compare coverage as well as verdict before selecting a redundant noop, so a prior review cannot hide newly incomplete coverage. Do not change Attention's completion predicate or other workflows' sink behavior.

### Keep publication failures explicit

Distinguish a prepared review, accepted safe output, host-applied output and publication failure. Enqueue comments and the single review through existing safe-output tools, preserving caps, target and marker. Bind output to the reviewed head and attempt. Only host application and the resulting GitHub review establish publication.

If preparation, detection or GitHub application fails, retain the prepared result and a compact diagnostic through existing artifacts/summary reporting where available. Surface the failed stage through terminal classification and recovery handling. Do not mark completion or recovery. For an ambiguous write response, reconcile the existing marker and target before retrying. Add no independent GitHub writer and never blindly resubmit a possibly applied review.

A graceful application deadline offers an opportunity to prepare an honest result. Hard Actions cancellation, process termination, resource loss or a failed GitHub write can prevent publication. Cleanup, snapshots and artifact upload are best effort under an external kill. Automatic replay/resume is outside this proposal.

## Compatibility with the public-engine migration

As checked on 2026-10-19, [#10347](https://github.com/pydantic/pydantic-ai/pull/10347) is **open**, at [`3f4e3e76`](https://github.com/pydantic/pydantic-ai/tree/3f4e3e76a8a2df28247552f7d3e7725b8a142097). Its [`agent.py :: WorkflowPolicy`](https://github.com/pydantic/pydantic-ai/blob/3f4e3e76a8a2df28247552f7d3e7725b8a142097/.github/scripts/pydantic_ai_gh_aw_shim/agent.py) wraps CLAI's run, enforces the existing deadline and emits health metadata. `build_agent()` retains the custom read-only `Task`; switching engines does not itself add partial review retention or enforced finalization. Its [tests](https://github.com/pydantic/pydantic-ai/blob/3f4e3e76a8a2df28247552f7d3e7725b8a142097/.github/scripts/test_workflow_agent.py) exercise the headless entry point and tool surface.

Prefer implementing after #10347, at that repository boundary, after reconciling its current head. If implementation must precede migration, agree a single repository-local policy owner and migration adapter first, then run the same acceptance cases through both entry paths. Do not copy policy into the public engine or invent a second result/health protocol. Preserve canonical session/usage reporting, task-key fallback, stale-PR noop preflight and host output ownership.

There is no demonstrated need for a new public abstraction. If adapting `DelegationTasks` needs a reusable library change, demonstrate the missing behavior and seek a separate explicit API decision before widening scope.

## Alternatives and human decisions

| Option | Benefit | Tradeoff |
|---|---|---|
| Prompts only | Small change | Cannot interrupt in-flight work or enforce publication; #10501 already reserves prompt time |
| Recover raw history only | Reuses capture/persistence | Reconstructing ambiguous partial text consumes finalization time; useful supporting evidence |
| Typed progress plus supervisor phases | Explicit trust boundary and deterministic fallback | Adds workflow state and cancellation/race tests; recommended |
| Adopt managed delegation wholesale | Reuses identity, persistence and cancellation | Changes `Task` integration/accounting; prove a narrow adapter first |
| Public review/checkpoint API | Could serve other callers | No demonstrated missing primitive or second caller; separate API decision required |

Before implementation, decide:

- Whether to accept typed progress and parent acceptance records, including the stricter coverage evidence rule.
- Whether managed-delegation adaptation or a small custom-`Task` record best preserves request admission, cancellation and read-only behavior.
- The enforced cutoff and verification/publication split, including deterministic output without a final model call and coverage-sensitive noop selection.
- Sequencing with #10347 and the precise safe-output preparation integration, retaining the host writer and wire formats.

Honest verified findings and coverage within existing limits are invariants. The proposed storage, tools and phase boundaries are design choices, not established framework contracts.

## Implementation sequence after agreement

1. Recheck main and #10347. Add deterministic failing cases for interruption and parent cutoff at the chosen repository entry point. Prove the managed-delegation reuse path before adding a store.
2. Add candidate/coverage retention and read-evidence validation. Preserve successful `Task` results; add bounded partial data only for CI Review. Cover concurrent and interrupted updates.
3. Add enforced phases with an injected clock and controllable in-flight operations. Preserve shared time/request accounting and timeout classification.
4. Add CI Review final-output validation and deterministic fallback through existing safe-output preparation. Test host failures and ambiguous outcomes without network writes. Update the seed prompt and managed-prompt deployment together to match enforcement.
5. Run targeted workflow/headless/delegate/output tests and the workflow guard. Recompile affected workflows only if integration requires it. Verify ordinary workflow and Agent compatibility.
6. After deployment, observe eligible agent execution and successful host publication bound to one target/run/attempt. Compare coverage, verified findings, tokens and elapsed time only on comparable executed samples. Keep telemetry details private. Incident closure needs separate recovery proof or explicit verified task retirement; this plan closes none.

## Minimal deterministic acceptance matrix

Use in-process scripted models/tools, an injected monotonic clock, temporary state and fake safe-output/host transport. Avoid real deadline waits, interpreter spawns and live providers. Extend existing [shim tests](https://github.com/pydantic/pydantic-ai/blob/62b7bf0aeac9a03066263469e650f40957d84cae/.github/scripts/test_pydantic_ai_runner.py) and migration headless tests rather than create another harness.

| Case | Required result |
|---|---|
| Candidate reported, delegate interrupted during next operation | Completed update survives; invocation is interrupted; unverified claim never becomes a published fact |
| One file inspected, another only assigned/read partially | Only fully read and reported-complete files count; remaining paths are unreviewed |
| Malformed/cross-assignment/stale updates and concurrent delegates | Reject invalid data, isolate records, preserve valid updates; no coverage inflation |
| Delegate returns incomplete lists, refuses or fails before reporting | Missing coverage remains unreviewed; no replacement fan-out |
| Cutoff with model, tool, delegate or compaction in flight | Cancel/drain owned investigation within original deadline; no broad work or late mutation |
| Parent receives all delegates, then attempts more broad work | Execution gate rejects post-cutoff work; finalization still runs |
| Verification reaches its cutoff | Candidate stays unverified; coverage gap recorded; output reserve remains |
| Partial coverage with/without accepted HIGH/CRITICAL finding | `REQUEST_CHANGES` with verified blocker, otherwise `COMMENT`; never approval |
| Complete coverage, bot author, redundant prior review | Existing severity/bot restrictions and coverage-sensitive noop policy preserved |
| Requests exhausted/provider unavailable during finalization | No new allowance/model; deterministic honest review from retained state |
| Wrapped provider transport timeout versus supervisor deadline | Preserve provider-timeout classification; only the run deadline is a run-time-limit failure, and external cancellation remains cancellation |
| Inline-only sink, valid final review, failed/ambiguous host write | Inline-only incomplete; verdict checked against state; failures remain failures, reconcile before retry |
| Hard cancellation and unrelated callers | Cancellation propagates without publication guarantee; ordinary Agent/other workflows unchanged |

This design-only PR needs source/prose review and normal text checks. The matrix is acceptance work for implementation, not a claim these behaviors exist today.

## Validation limits

A bounded Logfire baseline check did not establish a comparable post-#10501 agent-execution sample. No speed, usage or quality improvement is established. Successful pre-activation or provider-gated runs do not prove review completion. [Provider incident #10583](https://github.com/pydantic/pydantic-ai/issues/10583) remains open as checked on 2026-10-19; this can gate deployed acceptance but does not prove a deadline defect or recovery.

The smallest live integration check is one eligible execution that reaches the affected review phases and successfully applies a head-bound review. Deterministic interruption tests establish the failure-path behavior; comparable normal and interrupted execution samples are needed before claiming operational improvement. If an eligible execution lacks root/delegate/detection telemetry, repair that coverage before drawing conclusions. Hard cancellation and failed GitHub writes remain outside any publication guarantee.
