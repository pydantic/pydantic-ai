## Backwards compatibility in UI adapters (specially AG-UI)

Since [3971](https://github.com/pydantic/pydantic-ai/pull/3971#discussion_r3011028336) we decided to introduce the policy of sticking to the lower (existing) version requirement. In short, this means:
- version requirement bumps are disallowed
- new functionality should be gated behind version checks (including imports)
- older versions don't error out when they encounter new functionality, but instead skip it

The inbound half of that last rule lives in `ag_ui/_forward_compat.py`: AG-UI's `Message` and `InputContent` are discriminated unions, so a `role` or `type` added after the installed version is a hard validation failure for the whole request unless it's skipped first. Skipping is scoped to exactly that — anything else malformed must still fail.

Tests for version-gated behavior belong behind `requires_ag_ui('<version>')` in `tests/test_ag_ui.py`, never behind the module-level `imports_successful()` gate: a name that only exists above the floor in that import block skips the entire module on CI's `test-lowest-versions` job, which is the only job that exercises the floor these gates exist for.
