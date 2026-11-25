"""CDP Browser capability -- a throwaway Chrome the model drives with raw Chrome DevTools Protocol commands.

The dated external-service assumptions this package relies on (launch, auto-attach, request interception,
teardown) are recorded in the `_client` module docstring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec import BaseDurabilityCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness._browser_egress import EgressPolicy
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness.cdp_browser._toolset import DEFAULT_ALLOWED_CDP_METHODS, CDPBrowserToolset

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent


@dataclass
class CDPBrowser(AbstractCapability[AgentDepsT]):
    """Lets the agent drive a Chrome tab by sending raw Chrome DevTools Protocol commands.

    Protocol reference: <https://chromedevtools.github.io/devtools-protocol/>.

    Adds one tool, `browser_cdp`. Each run launches its own Chrome with a throwaway profile and closes it when the
    run ends. Raw CDP is a low-level API, so models do best with it when they write code that calls the tool in
    loops: combine it with `CodeMode`.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import CDPBrowser, CodeMode

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[CDPBrowser(), CodeMode()])
    ```

    Requires Chrome or Chromium, and the `cdp-browser` optional extra:

    ```bash
    pip install 'pydantic-ai-harness[cdp-browser]'
    ```

    Only the commands in `allowed_methods` can be sent. Every request any frame or worker makes is paused until the
    egress policy allows it: by default every public host is reachable and private, loopback, link-local and other
    reserved addresses are refused. With that block on, `http` and `https` connections go through a local proxy that
    resolves each name and connects only to addresses that pass the block, so DNS rebinding cannot reach a private
    address.

    WebSocket connections go through a local proxy that applies the whole policy.

    Durable execution (e.g. `TemporalDurability`) is not supported: a live browser cannot survive activity replay
    or worker restart, so combining both on one agent raises `UserError` at agent construction.
    """

    _: KW_ONLY

    headless: bool = True
    """Run Chrome without a window."""

    chrome_path: str | None = None
    """The Chrome or Chromium binary. Defaults to `CHROME_PATH`, then the usual install locations."""

    chrome_sandbox: bool = True
    """Run Chrome with its renderer sandbox, as `PlaywrightBrowser.chromium_sandbox` does.

    On by default: this capability opens pages nobody vetted, and the sandbox is what keeps a renderer that a crafted
    page compromises from reaching the host. Set `False` where the sandbox cannot start, such as a container without
    the kernel privileges it needs, or a Linux host that restricts unprivileged user namespaces for a Chromium build
    with no AppArmor profile of its own. A renderer exploit then runs with the agent's own access.
    """

    allowed_methods: Sequence[str] = DEFAULT_ALLOWED_CDP_METHODS
    """The CDP commands the model may send. Defaults to `DEFAULT_ALLOWED_CDP_METHODS`."""

    allowed_domains: list[str] | None = None
    """Egress allowlist. `None` (default) allows every public host.

    Each entry matches its exact host and any subdomain. It bounds top-level navigation and the requests a page's
    scripts use to move data (`fetch`, XHR, EventSource, `sendBeacon`); passive subresources and sub-frame documents
    are left alone. Pass an `EgressPolicy` as `policy` for anything this does not express.
    """

    block_private_addresses: bool = True
    """Refuse addresses that are not globally routable, for every request of every frame and worker.

    Covers the cloud metadata endpoint (`169.254.169.254`), loopback (`127.0.0.1`, `::1`, `localhost`) and the RFC 1918
    ranges, whatever `allowed_domains` says. A hostname is resolved first, so a name pointing at one of those
    addresses is refused too, as is one whose lookup does not answer. Set `False` when the agent should reach a local
    app or an internal dashboard.
    """

    policy: EgressPolicy | None = None
    """Full egress policy, the same `EgressPolicy` `PlaywrightBrowser` takes, for rules the shorthands cannot express.

    Mutually exclusive with `allowed_domains` and `block_private_addresses`. Absent from `from_spec`, since a
    policy subclass can carry arbitrary code.
    """

    id: str | None = 'cdp_browser'
    """Stable capability ID, so its instructions are addressable and `defer_loading=True` needs none.

    An agent has one `browser_cdp` tool, so two `CDPBrowser` sharing this `id` are one browser stated twice when they
    agree, and an error when they differ, since a merge could widen the commands or the egress one of them was given.
    To keep two, give each its own `id` and wrap them in `PrefixTools`.
    """

    _policy: EgressPolicy = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.policy is not None and (self.allowed_domains is not None or not self.block_private_addresses):
            raise UserError(
                'Pass either policy or the allowed_domains/block_private_addresses shorthands, not both: '
                'set those fields on the policy instead, so one object decides.'
            )
        # A tuple, so the same commands given as a list and as a tuple compare equal in `combine`.
        self.allowed_methods = tuple(self.allowed_methods)
        self._policy = self.policy or EgressPolicy(
            allowed_domains=self.allowed_domains, block_private_addresses=self.block_private_addresses
        )

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one browser stated twice; two that disagree raise rather than merge."""
        return one_per_id(capabilities)

    def for_agent(self, agent: AbstractAgent[AgentDepsT, object]) -> AbstractCapability[AgentDepsT]:
        """Refuse to bind to a durable-execution agent.

        Durable execution replays tool calls as activities, and a live Chrome cannot be checkpointed across activity
        boundaries or worker restarts; without this guard the composition fails inside the first tool call instead.
        Detection matches `BaseDurabilityCapability`, the shared base of the bundled Temporal, DBOS and Prefect
        integrations, as `PlaywrightBrowser.for_agent` does.
        """
        siblings: list[AbstractCapability[AgentDepsT]] = []
        agent.root_capability.apply(siblings.append)
        if any(isinstance(sibling, BaseDurabilityCapability) for sibling in siblings):
            raise UserError(
                'CDPBrowser does not support durable execution (e.g. TemporalDurability): '
                'a live Chrome cannot survive activity replay or worker restart. '
                'Run the browser agent outside the durable agent, or remove the durability capability.'
            )
        return self

    def get_toolset(self) -> CDPBrowserToolset[AgentDepsT]:
        """Provide the `browser_cdp` tool."""
        return CDPBrowserToolset(
            headless=self.headless,
            chrome_path=self.chrome_path,
            chrome_sandbox=self.chrome_sandbox,
            allowed_methods=self.allowed_methods,
            policy=self._policy,
            id=self.id,
        )

    @classmethod
    def from_spec(
        cls,
        *,
        headless: bool = True,
        chrome_path: str | None = None,
        chrome_sandbox: bool = True,
        allowed_methods: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        block_private_addresses: bool = True,
        id: str | None = 'cdp_browser',
        description: str | None = None,
        defer_loading: bool = False,
    ) -> CDPBrowser[AgentDepsT]:
        """Construct the capability from serializable spec options; `policy` is left out, as it can carry code."""
        return cls(
            headless=headless,
            chrome_path=chrome_path,
            chrome_sandbox=chrome_sandbox,
            allowed_methods=DEFAULT_ALLOWED_CDP_METHODS if allowed_methods is None else allowed_methods,
            allowed_domains=allowed_domains,
            block_private_addresses=block_private_addresses,
            id=id,
            description=description,
            defer_loading=defer_loading,
        )
