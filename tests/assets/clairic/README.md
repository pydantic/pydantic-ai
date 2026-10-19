# Clairic transport fixtures

These JSON results came from authenticated Cloudflare MCP executions on 2026-10-19.
They are fixtures, not HTTP recordings or VCR cassettes. HTTP headers and public traces
were unavailable. MockTransport tests use these results to verify adapter behavior.

- `scalar.json`: native scalar/list values and uncertainty. The text list includes an
  incorrect extra name; the adapter must preserve its value and confidence.
- `prediction.json`: numeric/date estimates, forecast and 90-minute duration. Each
  requested quantile array contains 99 values. Point values differ from medians.
- `image.json`: image-only input using `tests/assets/kiwi.jpg`; includes image token usage.
