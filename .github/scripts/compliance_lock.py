"""Files whose changes need a maintainer's `compliance:approved` on the pull request's current head.

`tests/models/test_claude_agent_compliance.py` pins what keeps `ClaudeAgentModel` within Anthropic's terms for
using a Claude subscription: the user's own unmodified Claude Code, credentials never read or forwarded, and the
options it runs with. An agent that "fixes" a failing test there could put users' accounts at risk, so
`review_gates.py` fails `Gate: compliance lock` on a PR that changes one of these files until a maintainer
approves it. This module is locked too, so taking a file off the list needs the same approval.
"""

LOCKED_PATHS = frozenset({'tests/models/test_claude_agent_compliance.py', '.github/scripts/compliance_lock.py'})
APPROVAL_LABEL = 'compliance:approved'
