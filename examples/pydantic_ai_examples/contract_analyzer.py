"""Legal Contract Analyzer — pydantic-ai example.

Analyzes a contract clause-by-clause, flags risky terms, identifies key
obligations and dates, and streams a plain-English risk report.

Run with::

    ANTHROPIC_API_KEY=your-key python -m pydantic_ai_examples.contract_analyzer

Or as a one-liner with uv::

    ANTHROPIC_API_KEY=your-key \\
      uv run --with "pydantic-ai[examples]" \\
      -m pydantic_ai_examples.contract_analyzer
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class RiskyClause(BaseModel):
    clause_number: str | None = None
    description: str
    risk_level: Literal["low", "medium", "high", "critical"]
    reason: str
    suggested_revision: str | None = None


class KeyDate(BaseModel):
    label: str  # e.g. "Payment due", "Contract expiry"
    date: date | None = None
    raw_text: str  # original text snippet if date is ambiguous


class Party(BaseModel):
    name: str
    role: Literal["client", "vendor", "employer", "employee", "licensor", "licensee", "other"]
    key_obligations: list[str]


class ContractAnalysis(BaseModel):
    contract_type: str  # e.g. "Software License Agreement"
    governing_law: str | None = None
    parties: list[Party]
    key_dates: list[KeyDate]
    risky_clauses: list[RiskyClause]
    overall_risk: Literal["low", "medium", "high", "critical"]
    summary: str = Field(..., description="One-paragraph plain-English overview")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


@dataclass
class AnalysisDeps:
    reviewer_role: str  # e.g. "vendor", "client" — whose perspective to flag risks from
    jurisdiction: str   # e.g. "Pakistan", "UK", "USA"


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

analysis_agent: Agent[AnalysisDeps, ContractAnalysis] = Agent(
    "anthropic:claude-sonnet-4-20250514",
    deps_type=AnalysisDeps,
    output_type=ContractAnalysis,
    system_prompt=(
        "You are an expert contract lawyer. "
        "Analyze the contract text provided and extract a structured analysis. "
        "Flag any clauses that are unusual, one-sided, or potentially harmful. "
        "Be concise but thorough."
    ),
)


@analysis_agent.system_prompt
async def inject_perspective(ctx: RunContext[AnalysisDeps]) -> str:
    return (
        f"Review from the perspective of the '{ctx.deps.reviewer_role}'. "
        f"Apply {ctx.deps.jurisdiction} legal context where relevant."
    )


report_agent: Agent[None, str] = Agent(
    "anthropic:claude-sonnet-4-20250514",
    output_type=str,
    system_prompt=(
        "You are a legal consultant writing plain-English risk reports for non-lawyers. "
        "Given a structured contract analysis, write a clear risk report. "
        "Use short paragraphs. Start with an executive summary, then list the top risks "
        "ordered from critical to low. End with 2-3 concrete recommendations."
    ),
)


# ---------------------------------------------------------------------------
# Sample contract text
# ---------------------------------------------------------------------------

SAMPLE_CONTRACT = """
SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 1, 2025,
between TechVendor Ltd. ("Vendor") and ClientCorp Inc. ("Client").

1. SERVICES
   Vendor shall provide software development services as described in Schedule A.

2. PAYMENT
   Client shall pay Vendor PKR 500,000 per month, due on the 1st of each month.
   Late payments shall incur a penalty of 5% per week compounded.

3. INTELLECTUAL PROPERTY
   All work product, inventions, and deliverables shall be the sole property of
   the Client. Vendor irrevocably assigns all IP rights to Client worldwide in
   perpetuity, including moral rights where waivable.

4. TERMINATION
   Client may terminate this Agreement at any time with 7 days notice.
   Vendor may only terminate with 90 days written notice and only for material
   breach by Client that remains uncured for 60 days.

5. LIABILITY
   Vendor's total liability shall not exceed one month's fees.
   Client's liability is unlimited.

6. CONFIDENTIALITY
   Vendor shall not disclose any information about Client or its business for
   a period of 10 years after termination.

7. GOVERNING LAW
   This Agreement shall be governed by the laws of England and Wales.

8. ENTIRE AGREEMENT
   This Agreement supersedes all prior agreements and representations.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    deps = AnalysisDeps(reviewer_role="vendor", jurisdiction="Pakistan")

    print("=" * 60)
    print("STEP 1 — Analyzing contract structure...")
    print("=" * 60)

    result = await analysis_agent.run(
        f"Analyze this contract:\n\n{SAMPLE_CONTRACT}",
        deps=deps,
    )
    analysis = result.output

    print(f"\nContract type   : {analysis.contract_type}")
    print(f"Governing law   : {analysis.governing_law}")
    print(f"Overall risk    : {analysis.overall_risk.upper()}")
    print(f"Parties         : {len(analysis.parties)}")
    print(f"Key dates       : {len(analysis.key_dates)}")
    print(f"Risky clauses   : {len(analysis.risky_clauses)}")

    print("\nRisky Clauses:")
    for clause in analysis.risky_clauses:
        print(f"  [{clause.risk_level.upper():8s}] {clause.description}")

    print("\n" + "=" * 60)
    print("STEP 2 — Streaming plain-English risk report...")
    print("=" * 60 + "\n")

    async with report_agent.run_stream(
        f"Write a risk report based on:\n{analysis.model_dump_json(indent=2)}"
    ) as stream:
        async for chunk in stream.stream_text(delta=True):
            print(chunk, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
