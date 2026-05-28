"""
LogicNodes Integration for PydanticAI
======================================
Demonstrates how to register LogicNodes deterministic compute workers as
tools in a PydanticAI agent. LogicNodes provides 2,300+ cryptographically-
signed microservices: gas oracles, compliance sentries, identity verification,
ZK attestation, DeFi data, and more.

Install:
    pip install pydantic-ai requests

Usage:
    export OPENAI_API_KEY="sk-..."
    export LOGICNODES_API_KEY="your_key_from_https://logicnodes.io/checkout"
    python docs/logicnodes_example.py
"""

import asyncio
import os
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, Tool

LOGICNODES_API_KEY = os.environ.get("LOGICNODES_API_KEY", "")
LOGICNODES_BASE = "https://logicnodes.io"


def _ln_headers() -> dict:
    """Return authorization headers for LogicNodes API."""
    if LOGICNODES_API_KEY:
        return {"Authorization": f"Bearer {LOGICNODES_API_KEY}"}
    return {}


# ---------------------------------------------------------------------------
# Pydantic models for structured LogicNodes responses
# ---------------------------------------------------------------------------

class GasEstimate(BaseModel):
    chain: str
    base_fee_gwei: float | None = None
    priority_fee_gwei: float | None = None
    max_fee_gwei: float | None = None
    signature: str | None = None
    raw: dict = {}


class ComplianceResult(BaseModel):
    agent_id: str
    action: str
    permitted: bool | None = None
    reason: str | None = None
    attestation_hash: str | None = None
    raw: dict = {}


class EthPrice(BaseModel):
    price_usd: float | None = None
    timestamp: int | None = None
    signature: str | None = None
    raw: dict = {}


class ZkAttestation(BaseModel):
    content_hash: str | None = None
    tx_hash: str | None = None
    chain: str | None = None
    proof_url: str | None = None
    raw: dict = {}


class GraphScore(BaseModel):
    agent_id: str
    score: float | None = None
    risk_tier: str | None = None
    raw: dict = {}


# ---------------------------------------------------------------------------
# LogicNodes tool implementations
# ---------------------------------------------------------------------------

async def gas_oracle(ctx: RunContext[None], chain: str = "ethereum") -> GasEstimate:
    """
    Query the LogicNodes gas oracle for deterministic EIP-1559 gas estimates.

    Returns cryptographically-signed gas data for the specified chain.
    Supported chains: ethereum, base, polygon, arbitrum, optimism.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{LOGICNODES_BASE}/call/gas-oracle",
            json={"chain": chain},
            headers=_ln_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    return GasEstimate(
        chain=chain,
        base_fee_gwei=data.get("base_fee_gwei"),
        priority_fee_gwei=data.get("priority_fee_gwei"),
        max_fee_gwei=data.get("max_fee_gwei"),
        signature=data.get("signature"),
        raw=data,
    )


async def compliance_sentry(
    ctx: RunContext[None], agent_id: str, action: str, context: str = ""
) -> ComplianceResult:
    """
    Run an on-chain compliance check for an autonomous agent action.

    Returns a verifiable attestation indicating whether the action is
    permitted under the current regulatory and constitutional ruleset
    anchored by LogicNodes.

    Args:
        agent_id: Unique identifier (wallet address or DID) for the agent.
        action: Description of the action to compliance-check.
        context: Optional JSON string providing additional context.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{LOGICNODES_BASE}/call/compliance-sentry",
            json={"agent_id": agent_id, "action": action, "context": context},
            headers=_ln_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    return ComplianceResult(
        agent_id=agent_id,
        action=action,
        permitted=data.get("permitted"),
        reason=data.get("reason"),
        attestation_hash=data.get("attestation_hash"),
        raw=data,
    )


async def eth_price(ctx: RunContext[None]) -> EthPrice:
    """
    Fetch the current ETH/USD price from LogicNodes.
    Output is cryptographically signed — suitable for on-chain price verification.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{LOGICNODES_BASE}/call/eth-price",
            headers=_ln_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    return EthPrice(
        price_usd=data.get("price_usd"),
        timestamp=data.get("timestamp"),
        signature=data.get("signature"),
        raw=data,
    )


async def zk_attest(ctx: RunContext[None], content: str) -> ZkAttestation:
    """
    Anchor content on-chain via LogicNodes ZK attestation.

    Creates a verifiable proof-of-existence anchored to Base L2 via USDC x402.
    Useful for audit trails, decision logs, and compliance evidence.

    Args:
        content: Text or JSON string to anchor on-chain.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{LOGICNODES_BASE}/x402/zk-attest",
            json={"content": content},
            headers=_ln_headers(),
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    return ZkAttestation(
        content_hash=data.get("content_hash"),
        tx_hash=data.get("tx_hash"),
        chain=data.get("chain"),
        proof_url=data.get("proof_url"),
        raw=data,
    )


async def graph_score(ctx: RunContext[None], agent_id: str) -> GraphScore:
    """
    Retrieve the LogicNodes trust graph score for an agent.

    Returns a reputation score based on on-chain interaction history,
    dispute records, and attestation volume.

    Args:
        agent_id: Agent wallet address or DID.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{LOGICNODES_BASE}/graph/score/{agent_id}",
            headers=_ln_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    return GraphScore(
        agent_id=agent_id,
        score=data.get("score"),
        risk_tier=data.get("risk_tier"),
        raw=data,
    )


# ---------------------------------------------------------------------------
# Build the PydanticAI agent
# ---------------------------------------------------------------------------

logicnodes_agent = Agent(
    model="openai:gpt-4o",
    system_prompt=(
        "You are an autonomous on-chain agent powered by LogicNodes deterministic "
        "compute infrastructure. Your responsibilities:\n"
        "1. Always call compliance_sentry before recommending any on-chain action.\n"
        "2. Use gas_oracle to provide accurate transaction cost estimates.\n"
        "3. Use eth_price for current ETH valuation.\n"
        "4. Anchor critical decisions and audit trails with zk_attest.\n"
        "5. Check graph_score to assess counterparty reputation before interaction.\n\n"
        "All LogicNodes responses are cryptographically signed and verifiable on Base L2."
    ),
    tools=[
        Tool(gas_oracle, takes_ctx=True),
        Tool(compliance_sentry, takes_ctx=True),
        Tool(eth_price, takes_ctx=True),
        Tool(zk_attest, takes_ctx=True),
        Tool(graph_score, takes_ctx=True),
    ],
)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

async def main():
    print("=== LogicNodes + PydanticAI Integration Demo ===\n")

    result = await logicnodes_agent.run(
        "I want to send 0.5 ETH to counterparty 0xDEADBEEF. "
        "Please: (1) get the current ETH price, "
        "(2) get a gas estimate for Ethereum, "
        "(3) check compliance for agent 'agent-demo' performing 'send 0.5 ETH to 0xDEADBEEF', "
        "(4) check the graph score for 0xDEADBEEF. "
        "Based on these results, should I proceed?"
    )

    print("Agent response:\n")
    print(result.data)

    print("\n--- Conversation history ---")
    for msg in result.all_messages():
        print(f"[{msg.kind}]: {str(msg)[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
