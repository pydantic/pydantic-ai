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
          run: |
            jq -c '.items[] | select(.type == "report_security_finding") | {text: "\(.summary)\n\n\(.details)"}' \
              "$GH_AW_AGENT_OUTPUT" | curl --fail -sS -H 'Content-Type: application/json' --data @- "$WEBHOOK"
---

## Security problems in released code are never public

Issues, PR comments, reviews and `noop` messages from this workflow are all public. If you
find a security vulnerability in code that has already shipped (not in changes made by the
pull request under review), report it only with `report_security_finding`, mention it
nowhere else, and end with `mcp__safeoutputs__noop` and the message
`One finding was reported privately.`
