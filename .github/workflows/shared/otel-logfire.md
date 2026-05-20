---
# Logfire OTLP observability shared import
# Exports gh-aw distributed traces (agent GenAI spans, setup/conclusion spans)
# to Pydantic Logfire via OTLP/HTTP.
#
# gh-aw POSTs OTLP/HTTP JSON to {endpoint}/v1/traces, so the endpoint below is
# the bare Logfire ingest base URL (no /v1/traces path). It MUST be a
# compile-time literal: gh-aw adds the parsed static hostname to the AWF
# network firewall allowlist at compile time, so it cannot be a ${{ vars.* }}
# expression. This is the standard Pydantic Logfire ingest URL; if you use a
# non-standard/self-hosted Logfire (i.e. a custom LOGFIRE_URL), update the
# literal below to match and recompile.
#
# Required secret:
#   LOGFIRE_WRITE_TOKEN — a Logfire project *write* token (distinct from the
#   LOGFIRE_READ_EXTERNAL_VARIABLES read key used for dynamic prompts). Used as
#   the Authorization header value for OTLP ingest.
#
# Usage:
#   imports:
#     - shared/otel-logfire.md
observability:
  otlp:
    endpoint: https://logfire-api.pydantic.dev
    headers:
      Authorization: ${{ secrets.LOGFIRE_WRITE_TOKEN }}
    if-missing: warn
---
