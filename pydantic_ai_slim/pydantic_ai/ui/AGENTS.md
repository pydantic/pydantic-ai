## Backwards compatibility in UI adapters (specially AG-UI)

Since [3971](https://github.com/pydantic/pydantic-ai/pull/3971#discussion_r3011028336) we decided to introduce the policy of sticking to the lower (existing) version requirement. In short, this means:
- version requirement bumps are disallowed
- new functionality should be gated behind version checks (including imports)
- older versions don't error out when they encounter new functionality, but instead skip it
