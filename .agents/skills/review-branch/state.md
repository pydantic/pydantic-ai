# review-state.json

`/review-branch` keeps one state file per worktree at `.agents/skills/review-branch/review-state.json`.
Tools outside the suite may read it; never duplicate its state elsewhere.

```json
{
  "version": 1,
  "branch": "<branch-name>",
  "base_ref": "<remote>/<ref>",
  "merge_base": "<sha>",
  "head_sha": "<reviewed commit sha>",
  "diff_hash": "sha256:<hex>",
  "last_run_at": "<ISO-8601>",
  "state_comment_id": <github-issue-comment-id or null>,
  "current_wave": 1,
  "waves": [
    {
      "wave_number": 1,
      "name": "code-correctness",
      "reviewers": {
        "<reviewer-id>": {
          "status": "pending|blocked|running|complete|skipped|failed",
          "blocking": <int>,
          "warnings": <int>,
          "required": <int>,
          "report_path": "<path>",
          "reason": "<for skipped/blocked/failed only>"
        }
      },
      "adjudication": {
        "status": "pending|complete|skipped|failed",
        "report_path": "tmp/wave<N>/claim-check.md",
        "confirmed": <int>, "rescoped": <int>, "refuted": <int>, "unsettled": <int>
      },
      "blocking_total": <int>,
      "required_total": <int>,
      "gating_blocking_total": <int>,
      "gate_status": "open|closed"
    },
    { "wave_number": 2, "name": "test-cassettes-docs", "...": "same shape" },
    { "wave_number": 3, "name": "cross-cutting", "...": "same shape" }
  ]
}
```

- Reviewer ids and waves: the reviewer table in `SKILL.md`. `review-claim-check` is recorded
  under the wave's `adjudication`, never among its reviewers.
- `blocking`, `warnings` and `required` are post-adjudication counts. `required` counts the
  findings whose `Verdict:` is `do`.
- `gating_blocking_total` exists on wave 1 only: BLOCKING across the three gating lanes. Wave 1's
  `gate_status` keys on it. `blocking_total` counts every wave-1 lane and still holds PASS.
- `diff_hash` is the SHA-256 of `tmp/code.diff`. Resume only when `merge_base`, `head_sha` and
  `diff_hash` all match the prepared snapshot.
- `state_comment_id` is the beacon's issue-comment id (`beacon.md`). Keep it across resets.
