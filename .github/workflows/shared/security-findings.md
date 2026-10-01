---
# Security problems in released code go to the private triage Slack channel, never to
# a public issue or comment. The markdown below is appended to the agent's prompt.
safe-outputs:
  jobs:
    report-security-finding:
      description: "Privately report a security problem in released code to the maintainers"
      runs-on: ubuntu-slim
      max: 1
      permissions: {}
      inputs:
        summary:
          description: "One line: affected component and kind of problem"
          required: true
          type: string
        details:
          description: "Affected code (file:line), how it is triggered, impact, and the reproduction you ran"
          required: true
          type: string
      steps:
        - env:
            WEBHOOK: ${{ secrets.PYDANTIC_AI_TRIAGE_SLACK_WEBHOOK_URL }}
            RUN_URL: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
          run: |
            jq -c --arg run "$RUN_URL" '.items[] | select(.type == "report_security_finding")
              | {text: ":lock: *Security finding:* \(.summary)\n\n\(.details[:3500])\n\n<\($run)|run>"}' \
              "$GH_AW_AGENT_OUTPUT" | curl --fail-with-body -sS -H 'Content-Type: application/json' --data @- "$WEBHOOK"
---

## Security problems in released code are never public

Issues, PR comments, reviews and `noop` messages from this workflow are all public. If you
find a security problem (injection, SSRF, path traversal, auth or approval bypass, a secret
or cross-user data leak, unsafe deserialization, sandbox escape) in code that has already
shipped, rather than in changes made by the pull request under review:

1. Call `report_security_finding`. It reaches the maintainers privately.
2. Do not mention it in any issue, comment or review.
3. End with `mcp__safeoutputs__noop` and the message `One finding was reported privately.`

A problem introduced by the pull request under review is normal review feedback.
