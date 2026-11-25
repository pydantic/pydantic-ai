"""A `BrowserUse` whose browser agent takes its model steps without a browser, for the capability contracts.

`tests/harness/capability_cases.py` builds it for the `BrowserUse` case's calls. It lives here, beside the tests that
need `browser-use`, because only the `harness browser-use` CI job installs it and type-checks this directory.
"""

from __future__ import annotations

from dataclasses import dataclass

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import UserMessage

from pydantic_ai_harness.browser_use import BrowserUse


class _Finished:
    """The history of a task the agent finished in every step."""

    def final_result(self) -> str:
        return 'The Pro plan costs $20 a month.'

    def errors(self) -> list[str | None]:
        return [None, None]  # pragma: no cover

    def is_successful(self) -> bool:
        return True

    @property
    def structured_output(self) -> None:
        return None  # pragma: no cover


@dataclass
class _TwoSteps:
    """Stands in for a browser-use agent: one model call per step on the task's chat model, two steps."""

    llm: BaseChatModel

    async def run(self, max_steps: int = 500) -> _Finished:
        for _ in range(2):
            await self.llm.ainvoke([UserMessage(content='The pricing page.')])
        return _Finished()


def offline_browser_use() -> BrowserUse[None]:
    """`BrowserUse` with its defaults, so the task's chat model is the run's own, and a stand-in browser agent.

    The session it builds is never started, so no browser launches.
    """
    return BrowserUse(browser_agent=lambda request: _TwoSteps(request.llm))
