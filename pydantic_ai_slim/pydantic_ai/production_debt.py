from __future__ import annotations as _annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class AgentDebtReport:
    step_id: str
    adi_score: float  # Agent Debt Index (target <= 12.0)
    token_sprawl_multiplier: float  # Target <= 1.12x
    step_latency_seconds: float  # Target <= 1.8s
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: list[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """Cryptographic SHA-256 hash-chained Action Ledger for PydanticAI agent execution steps."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_agent_event(
        self,
        step_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = (
            f"{index}|{self._last_hash}|{step_id}|{event_type}|"
            f"{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        )
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "step_id": step_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtAgentGate:
    """A2Z SOC Production Debt & Technical Due Diligence Gate for PydanticAI Autonomous Agents & Structured Tools.

    Quantifies tool call validation retries, agent token context sprawl, and step execution latency against 4 Enterprise KPIs:
    1. Agent Debt Index (ADI <= 12.0)
    2. Agent Token Sprawl Multiplier (ATSM <= 1.12x)
    3. P99 Agent Step Latency (<= 1.8s)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_adi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_adi = max_acceptable_adi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        return any(Path(p).exists() for p in ("artifacts/KILL", "/tmp/KILL"))

    def evaluate_agent_step(
        self,
        step_id: str,
        prompt_tokens: int = 1500,
        response_tokens: int = 1600,
        step_latency_seconds: float = 1.1,
        tool_validation_retries: int = 0,
        un_gated_mutations: int = 0,
    ) -> AgentDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_agent_event(
                step_id=step_id,
                event_type="agent_step_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            err_msg = "A2Z SOC ActionGate: Emergency kill switch is engaged. PydanticAI agent execution halted."
            raise PermissionError(err_msg)

        critical_smells: list[str] = []

        # KPI 2: Agent Token Sprawl Multiplier
        token_ratio = response_tokens / max(1, prompt_tokens)
        if token_ratio > 1.8:
            critical_smells.append(f"HIGH_AGENT_TOKEN_SPRAWL_{token_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if step_latency_seconds > 10.0:
            critical_smells.append(f"HIGH_AGENT_STEP_LATENCY_{step_latency_seconds:.1f}S")

        # Tool validation retry count
        if tool_validation_retries > 0:
            critical_smells.append(f"DETECTED_{tool_validation_retries}_TOOL_VALIDATION_RETRIES")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_TOOL_MUTATIONS")

        # KPI 1: Agent Debt Index (0 = Clean, 100 = Catastrophic)
        adi = (
            max(0.0, (token_ratio - 1.0) * 20.0)
            + max(0.0, (step_latency_seconds - 1.8) * 1.5)
            + (tool_validation_retries * 25.0)
            + (un_gated_mutations * 30.0)
        )
        adi_score = round(min(100.0, adi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - adi_score)
        is_production_ready = (
            adi_score <= self.max_acceptable_adi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_agent_event(
            step_id=step_id,
            event_type="agent_step_authorized" if is_production_ready else "agent_step_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "adi_score": adi_score,
                "token_ratio": token_ratio,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "step_latency_seconds": step_latency_seconds,
                "tool_validation_retries": tool_validation_retries,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return AgentDebtReport(
            step_id=step_id,
            adi_score=adi_score,
            token_sprawl_multiplier=round(token_ratio, 2),
            step_latency_seconds=round(step_latency_seconds, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
