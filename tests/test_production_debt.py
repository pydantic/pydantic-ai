import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../pydantic_ai_slim/pydantic_ai/production_debt.py",
)
spec = importlib.util.spec_from_file_location("pydantic_ai_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["pydantic_ai_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtAgentGate = production_debt_mod.ProductionDebtAgentGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtAgentGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtAgentGate(
            never_equate_intent_to_approval=True,
            max_acceptable_adi=12.0,
        )

    def test_clean_agent_step_passes_readiness(self) -> None:
        report = self.gate.evaluate_agent_step(
            step_id="pydantic_ai_clean_step",
            prompt_tokens=1500,
            response_tokens=1600,
            step_latency_seconds=1.1,
            tool_validation_retries=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.adi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_agent_step_fails_debt(self) -> None:
        report = self.gate.evaluate_agent_step(
            step_id="uncalibrated_agent_step",
            prompt_tokens=1500,
            response_tokens=4200,  # 2.8x token context sprawl
            step_latency_seconds=14.0,  # High step latency
            tool_validation_retries=3,  # 3 validation retries
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.adi_score, 50.0)
        self.assertIn("HIGH_AGENT_TOKEN_SPRAWL_2.80X", report.critical_smells)
        self.assertIn("HIGH_AGENT_STEP_LATENCY_14.0S", report.critical_smells)
        self.assertIn("DETECTED_3_TOOL_VALIDATION_RETRIES", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_TOOL_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_agent_step("step-1")
        self.gate.evaluate_agent_step("step-2")
        self.gate.evaluate_agent_step("step-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
