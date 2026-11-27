"""Files whose changes need a maintainer's `compliance:approved` on the pull request's current head.

`tests/models/test_claude_agent_compliance.py` pins what keeps `ClaudeAgentModel` within Anthropic's terms for
using a Claude subscription, and an agent that "fixes" a failing test there could put users' accounts at risk.
`.github/workflows/AGENTS.md` ("PR metadata gates enforce the merge bar") describes `Gate: compliance lock`, which
`review_gates.py` publishes. This module is locked too, so taking a file off the list needs the same approval.
"""

LOCKED_PATHS = frozenset(
    {
        'tests/models/test_claude_agent_compliance.py',
        # The stand-in Claude Code the tests read the options it's started with from.
        'tests/models/claude_code_fake.py',
        '.github/scripts/compliance_lock.py',
    }
)
APPROVAL_LABEL = 'compliance:approved'
