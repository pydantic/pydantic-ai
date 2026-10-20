# CI Review: resume timed-out delegates and enforce the final reserve

This PR changes only `PLAN.md`. It proposes three changes to the CI Review runner shim and prompt, and asks for agreement before implementation.

## Evidence

Two CI Review runs on current source (`5c8a6c12`) have executed the agent since #10501 merged. In both, at least one delegate hit the 15-minute `Task` timeout.

| Run | PR | What happened |
|---|---|---|
| [39358200683](https://github.com/pydantic/pydantic-ai/actions/runs/39358200683) | #10444, 26 files | 2 of 3 delegates timed out. Both had read every assigned diff; one had already concluded it had no findings. The parent spent the next 11 minutes re-reviewing their files. The time-budget notice was in each of its last 8 requests; 7 of them made more investigation calls. The run hit its 28-minute limit while the parent was still thinking toward a submission. **Nothing was posted.** |
| [39360027162](https://github.com/pydantic/pydantic-ai/actions/runs/39360027162) | #10557, 28 files | 1 of 3 delegates timed out 52 s into writing its final "no findings" report. The parent re-reviewed its 11 files in 6.4 minutes, reading full source for only 2 of the 9 modified ones. It then posted an [APPROVE](https://github.com/pydantic/pydantic-ai/pull/10557#pullrequestreview-5580367529) saying "all 28 changed files reviewed". |

What the telemetry shows:

- **Timeouts come from output volume.** Each delegate generates 20–30k output tokens, mostly thinking, at about 45 tokens/s, with about 8 s before the first chunk of each request. Request count and repeated reads are not the bottleneck. One delegate lost about 2.5 minutes to two slow first chunks.
- **A timeout discards finished work.** `cli.py :: task` wraps `sub.run()` in `asyncio.wait_for`. On timeout it keeps the usage but drops the messages, and returns only `error: sub-agent timed out after 15min`.
- **The prompt-only reserve does not hold.** The parent receives `TIME_BUDGET_NOTICE` and keeps investigating.

The three earlier incidents (#10259, #10309, #10570) were also time-limit hits. All three predate #10501.

## Changes

### 1. Resume a timed-out delegate from its saved history

When the timeout fires, `task` keeps the sub-agent's message history up to the last completed response, dropping a partial response and any unanswered tool calls. It then makes one more request on that history with no tools: "write your report now from what you have; list the files you did not review". The parent receives that report, marked as produced after a timeout.

- The resume runs only when existing admission allows it: time left after the parent's reserve, and request headroom from `_subagent_request_limit`. Otherwise `task` returns the timeout error, as it does today.
- Without tools the resume is exactly one request. In both runs that fits the time remaining: about 6 minutes were admissible in 39358200683, and the delegate in 39360027162 was already partway through its report.
- The prompt changes "Do not dispatch it again" to say that a resumed report counts like any other delegate report. The parent verifies its findings and treats files the report lists as unreviewed as unreviewed.

### 2. Enforce the parent's time reserve

When the run's remaining time falls inside `RUN_TIME_RESERVE_SECS`, the condition that already triggers `time_budget_notice`, the parent's next request offers only the safe-output tools. The request in flight is not cancelled. The notice stays, so the model knows why its tools are gone.

The reserve and its length are unchanged; only compliance becomes deterministic. In 39358200683 this would have left the parent four minutes to post an incomplete review instead of nothing.

### 3. Hold files the parent takes over to the delegate bar

The prompt already requires a delegate to read a file's diff and full source before claiming it reviewed. Files the parent reviews itself after a delegate failure meet the same bar. New files count after their diff alone, because the annotated diff contains the whole file. Anything else is listed as unreviewed, so the review cannot approve.

This is a prompt rule. If a deployed run shows the parent ignoring it, the follow-up is an approval check in the shim based on the `Read` calls it already executes.

## Not in this plan

These are deliberate scope decisions:

- **Background delegation and verifying findings while other delegates run.** The most it could save is the time the parent sat waiting: 0.5 and 5.5 minutes in these runs.
- **A run-scoped evidence store, failure-cause matrix, and resume-versus-replace selection.** Resume covers both observed timeouts. Replacement and the other failure classes have no post-#10501 occurrence yet.
- **Limits, models, concurrency and retry policy.** These stay as they are, including the 15-minute delegate timeout and the request-limit tenth. The request limit was not hit in either run.
- **Telemetry gaps.** Cut-off requests record no tokens, and model HTTP calls emit no spans, so retries are invisible. A separate PR fixes these before the comparison below.

## Implementation

The changes are in `.github/scripts/pydantic_ai_gh_aw_shim/cli.py` (`task` and the parent agent's tool selection) and `.github/workflows/shared/prompts/pydantic-ai-pr-review.md` (Step 3 and coverage rules). #10347 keeps the custom `Task` tool and `time_budget_notice`; whichever lands second rebases onto the other.

Tests go in `.github/scripts/test_pydantic_ai_runner.py`, in-process, with scripted models and injected time:

- A delegate that times out after completed reads yields a report built from its history, with its unreviewed files listed.
- A timeout with no admissible time or request headroom returns the error unchanged and makes no resume request.
- A partial final response and unanswered tool calls are trimmed before the resume request.
- Inside the reserve, the parent's request offers only safe-output tools. Before the reserve, the full toolset is offered.
- Usage from the resume request is charged to the parent's run.

## Measuring the result

After deployment, compare matched CI Review runs on large PRs (three delegates) against the two runs above:

- the share of runs that post a review
- delegate timeouts that end in a usable resumed report
- parent minutes spent after delegation
- input and output tokens per run

This needs the telemetry fixes first, and enough runs to compare like with like.
