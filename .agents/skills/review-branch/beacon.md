# Review-state comment (beacon)

`/review-branch` Step 8 mirrors the run into one PR comment, edited in place on every run. The
comment tells a maintainer in the GitHub UI whether and how far the branch was reviewed
locally, what the suite found, and what it already fixed. It mirrors state; it is not a triage
surface. The fix flow stays the inline review-comment process.

## Content

1. The attribution line your instructions require on a GitHub comment, if any, then the hidden
   marker `<!-- pyai-review-state -->`.
2. The per-wave table. Its count columns are always the **current open** counts: a resolved
   required item is `0`, never `0 (2 resolved)`. Resolved work lives in the verdict line and
   the applied bucket.
3. The verdict line. Append ` · <N> applied` whenever at least one finding was fixed.
4. The `Reviewed at` line.
5. One collapsed `<details>` bucket per non-empty category, in this order: blocking, required,
   applied, informational.

### Verdict transitions

When an edit changes the verdict from the previous edit, state the change and its cause on the
verdict line, never a bare new snapshot: `✅ PASS — was 🔧 3 required; all 3 resolved in <sha>`.
Say that the suite resolved findings it flagged itself. A PASS reached by clearing flagged `do`
items must not read like a PASS that was clean from the start.

When wave 3 closes clean, the verdict reads `✅ all 3 waves passed, 0 blocking`. When a push
changes `diff_hash`, the next run rewrites the same comment to show the re-review in progress.

### Buckets

Assign each finding once: BLOCKING severity → blocking; WARNING with `Verdict: do` → required;
`outcome: fixed` → applied; `Verdict: defer` or `skip` → informational. Blocking and required
buckets sort by actionability, so a required fix cannot hide among heads-ups. The applied bucket
sorts by outcome.

Open every bucket body with the italic disclaimer
`_Automated review — flagged by the review suite, not yet confirmed by a maintainer._` These are
often contributor PRs. Reference code as `file` · `symbol`, never by line number.

| Bucket | `<summary>` | Each finding carries |
|---|---|---|
| Blocking | `🛑 <N> blocking` | title, reference, background (what the code does), the problem (what breaks), why it blocks (impact, who hits it) |
| Required | `🔧 <N> required (do)` | title, reference, the rule or the consequence if shipped, the fix. Frame the bucket as a checklist, not as options. |
| Applied | `✅ <N> applied since the review` | title, reference, one line: what the review flagged and the fix that landed |
| Informational | `⚠️ <N> informational (defer/skip)` | title, reference, one line of context; a `defer` names its tracking artifact, a `skip` names why it is net-neutral or a scope boundary |

- **Applied** holds only findings this suite raised and then fixed within the review cycle,
  never unrelated human edits. A finding's `outcome` becomes `fixed` when the orchestrator
  addresses a blocking or `do` item before the beacon edit, or when a re-run after a push
  confirms it resolved. When the applied items are the ones that held the verdict below PASS,
  use the summary `✅ <N> required action(s) resolved` and open the body with
  `_the <N> required actions the prior round flagged, now resolved in <sha>._`
- A `defer` with no tracking artifact is a review bug. List it under required as `do`.
- Omit a bucket whose count is 0. The exception: keep the applied bucket whenever at least one
  finding was fixed, even on a clean verdict. A review that found three things and fixed them
  must not render like a review that found nothing. A clean run with no fixes shows only the
  table and the verdict.

## Template

```markdown
🔍 Local review suite (automated, pre-merge)
| Wave | Status | Blocking | Required (do) | Warnings (defer/skip) |
|---|---|---|---|---|
| 1 · code-correctness | <✅ closed / 🛑 open / ⏳ running / — not yet> | <N> | <N> | <N> |
| 2 · tests/cassettes/docs | <✅ closed / — not yet> | <N> | <N> | <N> |
| 3 · cross-cutting/siblings | <✅ closed / — not yet> | <N> | <N> | <N> |

**Verdict:** <✅ all 3 waves passed, 0 blocking, 0 required / ⏳ wave <K> in progress / 🛑 blocked — <N> blocking in wave <K> / 🔧 <N> required action(s) to address before PASS>< · <N> applied>
Reviewed at <ISO-8601> · head `<review_head>` · diff `<short-hash>`

<details>
<summary>🛑 <N> blocking</summary>

_Automated review — flagged by the review suite, not yet confirmed by a maintainer._

**1. <title>** · `<file>` · `<symbol>`
<background> <the problem> <why it blocks>
</details>
```

The other buckets follow the same `<details>` shape with the fields from the bucket table. The
script below prepends the marker, and the attribution line when `ATTRIBUTION` holds one.

## Post or edit

Find the comment through the marker, keep its id in `review-state.json` → `state_comment_id`,
and PATCH it on every run. PR comments are issue comments, so the `issues/.../comments` REST
endpoints work even under a GraphQL rate limit. Write the composed body, from the `🔍` line
down, to `$BODY_FILE` first.

```bash
MARKER='<!-- pyai-review-state -->'
BODY=$(printf '%s\n%s\n' "$MARKER" "$(cat "$BODY_FILE")")
# ATTRIBUTION: the attribution line your instructions require on a GitHub comment, or unset.
if [ -n "${ATTRIBUTION:-}" ]; then
  BODY=$(printf '%s\n%s\n' "$ATTRIBUTION" "$BODY")
fi
STATE=.agents/skills/review-branch/review-state.json
CID=$(jq -r '.state_comment_id // empty' "$STATE" 2>/dev/null)
if [ -z "$CID" ]; then
  # No stored id (e.g. a fresh worktree): reuse a surviving marked comment.
  CID=$(gh api repos/pydantic/pydantic-ai/issues/"$PR_NUMBER"/comments --paginate \
        --jq "[.[] | select(.body | contains(\"$MARKER\"))] | last | .id" 2>/dev/null)
fi
if [ -n "$CID" ] && [ "$CID" != "null" ]; then
  gh api -X PATCH repos/pydantic/pydantic-ai/issues/comments/"$CID" -f body="$BODY" >/dev/null
else
  CID=$(gh api repos/pydantic/pydantic-ai/issues/"$PR_NUMBER"/comments -f body="$BODY" --jq .id)
fi
tmp=$(mktemp); jq --argjson id "$CID" '.state_comment_id = $id' "$STATE" > "$tmp" && mv "$tmp" "$STATE"
```
