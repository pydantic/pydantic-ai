---
# Logfire OTLP observability shared import
# Exports gh-aw distributed traces (agent GenAI spans, setup/conclusion spans)
# to Pydantic Logfire via OTLP/HTTP.
#
# gh-aw POSTs OTLP/HTTP JSON to {endpoint}/v1/traces, so this endpoint must be
# the bare Logfire ingest base URL (no /v1/traces path).
#
# Use vars.LOGFIRE_URL to avoid hardcoding endpoints in each workflow. Keep
# network allowlists in sync with the possible hosts for this variable.
#
# Required secret:
#   LOGFIRE_TOKEN — a Logfire project write token. Used as the Authorization
#   header value for OTLP ingest and passed directly to the agent container so
#   the Logfire Python SDK can also use it natively.
#
# Usage:
#   imports:
#     - shared/otel-logfire.md
observability:
  otlp:
    endpoint: ${{ vars.LOGFIRE_URL || 'https://logfire-api.pydantic.dev' }}
    headers:
      Authorization: ${{ secrets.LOGFIRE_TOKEN }}
    if-missing: warn
---
