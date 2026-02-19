"""Medical Triage System with Agent Delegation.

The `triage_agent` acts as the central decision-maker and orchestrator.
The instructions helps it to "call tools to consult specialists or a senior doctor."
It delegates the actual medical work (diagnosis or treatment planning) to other agents.

The two core functions act as the delegation mechanism:

- consult_specialist: This tool routes the complaint to a specific Specialist Agent
(cardiology_agent, neurology_agent, etc.). This is Level 1 Delegation: Routing to expertise.

- consult_senior_doctor: This tool routes the complaint to a Senior Agent (senior_doctor_agent).
This is Level 2 Delegation: Escalation for critical decision-making.

Demonstrates:
- Master agent coordinating specialized sub-agents
- Dynamic routing and delegation based on symptom analysis
- Structured output

Run with:

    uv run -m pydantic_ai_examples.medical_agent_delegation
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from pydantic_ai import Agent, ModelHTTPError, RunContext

MODEL = 'openai:gpt-4.1-mini'


# Structured Outputs
class Specialty(str, Enum):
    general = 'general'
    cardiology = 'cardiology'
    neurology = 'neurology'


class MedicalReport(BaseModel):
    diagnosis: list[str]
    differential: list[str]
    recommended_tests: list[str]
    immediate_actions: list[str]
    estimated_time_minutes: int


class TreatmentPlan(BaseModel):
    plan_summary: str = Field(
        description='The structured treatment plan from the senior doctor'
    )
    refer_to_specialist: Specialty | None = Field(
        description='Specialty to route the patient to for further treatment, if necessary'
    )
    follow_up_days: int


class TriageFinalOutput(BaseModel):
    """The final structured output containing the result of the entire flow."""

    specialty: Specialty | None = None
    final_report: MedicalReport | None = None
    treatment_plan: TreatmentPlan | None = None
    final_status: str = Field(
        ..., description="Status: 'resolved_by_specialist' or 'escalated'"
    )


# Shared Dependency
@dataclass
class PatientInfo:
    patient_id: str
    age: int
    known_conditions: list[str]


class TestPatient(TypedDict):
    complaint: str
    patient: PatientInfo


# Specialist and Senior Agents
gp_agent = Agent(
    MODEL,
    output_type=MedicalReport,
    deps_type=PatientInfo,
    system_prompt="""
    You are a general practitioner.
    """,
)

cardiology_agent = Agent(
    MODEL,
    output_type=MedicalReport,
    deps_type=PatientInfo,
    system_prompt="""
    You are a cardiology specialist.
    """,
)

neurology_agent = Agent(
    MODEL,
    output_type=MedicalReport,
    deps_type=PatientInfo,
    system_prompt="""
    You are a neurology specialist.
    """,
)

senior_doctor_agent = Agent(
    MODEL,
    output_type=TreatmentPlan,
    deps_type=PatientInfo,
    system_prompt="""
    You are a senior clinician overseeing complex or ambiguous cases.
    Integrate all prior findings to produce a clear treatment plan.
    """,
)

SPECIALIST_MAP = {
    'general': gp_agent,
    'cardiology': cardiology_agent,
    'neurology': neurology_agent,
}

# Agent-as-Orchestrator: triage_agent with Delegation Tools
triage_agent = Agent(
    MODEL,
    output_type=TriageFinalOutput,
    deps_type=PatientInfo,
    system_prompt="""
    You are a triage clinician coordinating medical workflow.
    You can call tools to consult specialists or a senior doctor.

    AVAILABLE SPECIALTIES:
    - "general": General practitioner for common issues
    - "cardiology": For heart, chest pain, cardiac symptoms
    - "neurology": For brain, nerve, stroke, headache symptoms

    Always produce a structured TriageFinalOutput.
    """,
)


@triage_agent.tool
async def consult_specialist(
    ctx: RunContext[PatientInfo],
    specialty: Specialty,
    question: str,
) -> TriageFinalOutput | str:
    """Consult the appropriate specialist for expert consultation."""
    specialist_agent = SPECIALIST_MAP.get(specialty)
    print(f'Proceed with specialist - {specialty}')
    if not specialist_agent:
        print('Selected specialist does not exists!')
        return f'No specialist found for {specialty.name}.'

    result = await specialist_agent.run(f'Consultation: {question}', deps=ctx.deps)
    report: MedicalReport = result.output

    return TriageFinalOutput(
        final_status='resolved_by_specialist',
        specialty=specialty,
        final_report=report,
    )


@triage_agent.tool
async def consult_senior_doctor(
    ctx: RunContext[PatientInfo], reason_for_escalation: str, initial_complaint: str
) -> TriageFinalOutput:
    """Consult senior doctor in case of escalation and emergency cases.

    Immediately escalates the case to the senior clinician for severe cases and for a final TreatmentPlan.
    Use this for high severity, critical, or ambiguous cases.

    Args:
        ctx: Pydantic AI agent RunContext
        reason_for_escalation: Summary of why the case must be escalated (e.g., "Severe pain, possible cardiac event").
        initial_complaint: The patient's original complaint.
    """
    patient = ctx.deps
    senior_note = f'Reason: {reason_for_escalation}\nComplaint and context:\n{initial_complaint}\nPatient: {patient.patient_id}, age {patient.age}\n'

    print('Direct escalation triggered by Triage LLM.')
    treatment_plan = None
    try:
        result = await senior_doctor_agent.run(
            f'Consultation for: {senior_note}', deps=ctx.deps
        )
        treatment_plan = result.output
    except ModelHTTPError as e:
        # Handle case where LLM fails to provide TreatmentPlan structure
        treatment_plan = TreatmentPlan(
            plan_summary=f'Consultation failed due to API error: {e.status_code}. Requires manual review.',
            refer_to_specialist=None,
            follow_up_days=1,
        )

    return TriageFinalOutput(
        final_status='escalated',
        treatment_plan=treatment_plan,
    )


# Coordinator System
class MedicalTriageSystem:
    """Coordinator that invokes triage_agent as the orchestrator."""

    def __init__(self):
        self.triage = triage_agent
        self.medical_history: list[dict[str, Any]] = []

    async def handle_patient(
        self, complaint: str, patient: PatientInfo
    ) -> dict[str, str]:
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        print(f'\n[{timestamp}] Processing complaint: {complaint}')

        triage_prompt = (
            f'Patient {patient.patient_id}, age {patient.age}\n'
            f'Complaint: {complaint}\n'
            f'Known conditions: {patient.known_conditions}\n'
            f'If necessary, use your tools to consult specialists or senior doctor.'
        )

        triage_result = await self.triage.run(triage_prompt, deps=patient)
        final_output: TriageFinalOutput = triage_result.output

        record = {
            'timestamp': timestamp,
            'patient_id': patient.patient_id,
            'path': final_output.final_status,
            'specialty': final_output.specialty,
            'report_summary': final_output.final_report.diagnosis
            if final_output.final_report
            else 'N/A',
            'treatment_summary': final_output.treatment_plan.plan_summary
            if final_output.treatment_plan
            else 'N/A',
        }
        self.medical_history.append(record)

        return final_output.model_dump()


async def demo_medical_triage():
    system = MedicalTriageSystem()

    test_patients: list[TestPatient] = [
        {
            'complaint': 'Sudden severe chest pain radiating to left arm and shortness of breath.',
            'patient': PatientInfo(
                patient_id='P001', age=64, known_conditions=['hypertension']
            ),
        },
        {
            'complaint': 'Intermittent headaches for 2 weeks, mild nausea, no weakness.',
            'patient': PatientInfo(patient_id='P002', age=34, known_conditions=[]),
        },
        {
            'complaint': "Sudden onset of the worst headache of my life, followed by blurry vision and now I can't feel my left leg. I took aspirin an hour ago.",
            'patient': PatientInfo(
                patient_id='P003',
                age=71,
                known_conditions=['Type 2 Diabetes', 'Chronic Migraines'],
            ),
        },
        {
            'complaint': 'Hard to breath and faint every few minutes.',
            'patient': PatientInfo(patient_id='P003', age=71, known_conditions=[]),
        },
    ]

    for entry in test_patients:
        print(f'Processing patient {entry["patient"].patient_id}')
        result = await system.handle_patient(entry['complaint'], entry['patient'])
        print('Result:', result)

    print('\nMEDICAL HISTORY SUMMARY:')
    for history in system.medical_history:
        print(
            f'- {history["timestamp"]} | Patient {history["patient_id"]} | Path: {history["path"]}'
        )


if __name__ == '__main__':
    asyncio.run(demo_medical_triage())
