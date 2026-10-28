#!/usr/bin/env bash
# Shared PR Bots / PR Gates calculation. Read the files API JSON on stdin.
set -euo pipefail

FILES=$(cat)
: "${1:?base SHA required}"
GENERATED=$(jq -j '.[] | .filename, "\u0000"' <<<"$FILES" |
  git check-attr --source="$1" -z --stdin linguist-generated |
  jq -Rs 'split("\u0000") | [range(0; length; 3) as $i |
    select(.[$i + 2] == "true" or .[$i + 2] == "set") | .[$i]]')
FILES=$(jq --argjson generated "$GENERATED" 'map(select(.filename as $path | $generated | index($path) | not))' <<<"$FILES")

CODE=$(jq '[.[] | select(
  (.filename | (startswith("tests/") or startswith("docs/") or endswith(".md")) | not) and
  (.filename != "uv.lock") and (.filename | contains("/cassettes/") | not)
) | .additions + .deletions] | add // 0' <<<"$FILES")
DOCS=$(jq '[.[] | select(
  (.filename | (startswith("docs/") or endswith(".md"))) and
  (.filename != "uv.lock") and (.filename | contains("/cassettes/") | not)
) | .additions + .deletions] | add // 0' <<<"$FILES")
TESTS=$(jq '[.[] | select(
  (.filename | startswith("tests/")) and (.filename | endswith(".md") | not) and
  (.filename | contains("/cassettes/") | not)
) | .additions + .deletions] | add // 0' <<<"$FILES")
SCORE=$((CODE + DOCS / 2 + TESTS / 2))
if [ "$SCORE" -le 100 ]; then
  SIZE='size: S'
elif [ "$SCORE" -le 500 ]; then
  SIZE='size: M'
elif [ "$SCORE" -le 1500 ]; then
  SIZE='size: L'
else
  SIZE='size: XL'
fi
jq -n --arg size "$SIZE" --argjson score "$SCORE" --argjson code "$CODE" --argjson docs "$DOCS" --argjson tests "$TESTS" \
  '{size: $size, score: $score, code: $code, docs: $docs, tests: $tests}'
