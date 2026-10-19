# Clairic transport fixtures

The JSON result fixtures came from authenticated Cloudflare MCP executions on 2026-10-19.
They are not HTTP recordings or VCR cassettes. The `test_live_*` tests in
`tests/models/test_cloudflare_clairic.py` record the public `Agent.run` path against real
Cloudflare HTTP requests and responses. Their cassettes are in
`tests/models/cassettes/test_cloudflare_clairic/`.

- `scalar.json`: native scalar/list values and uncertainty. The text list includes an
  incorrect extra name; the adapter must preserve its value and confidence.
- `prediction.json`: numeric/date estimates, forecast and 90-minute duration. Each
  requested quantile array contains 99 values. Point values differ from medians.
- `image.json`: image-only input using `tests/assets/kiwi.jpg`; includes image token usage.
