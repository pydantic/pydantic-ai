"""AML (Anti-Money Laundering) Compliance Agent example.

This example demonstrates a multi-agent workflow for AML compliance:
1. Sanctions Screening - Check against sanctions lists
2. Risk Assessment - Evaluate customer risk profile
3. Transaction Monitoring - Detect suspicious patterns
4. Compliance Reporting - Generate regulatory reports

Run with:

    uv run -m pydantic_ai_examples.aml_compliance
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext


# Risk levels
RiskLevel = Literal['low', 'medium', 'high', 'critical']
ActionType = Literal['clear', 'review', 'block']


@dataclass
class AMLDependencies:
    """Dependencies for AML compliance checks."""
    customer_id: str
    customer_name: str
    country: str
    business_activity: str
    transaction_history: list[dict]


class ScreeningResult(BaseModel):
    """Result of sanctions screening."""
    risk_level: RiskLevel = Field(description='Risk level from sanctions check')
    action: ActionType = Field(description='Recommended action')
    details: str = Field(description='Explanation of screening results')
    matches_found: list[str] = Field(default_factory=list, description='Any sanctions matches')


class RiskAssessmentResult(BaseModel):
    """Result of customer risk assessment."""
    overall_risk: RiskLevel = Field(description='Overall customer risk level')
    review_frequency: Literal['annual', 'biannual', 'quarterly', 'monthly'] = Field(
        description='How often to review this customer'
    )
    justification: str = Field(description='Reasoning for risk assessment')
    factors: list[str] = Field(description='Risk factors identified')


class TransactionAlert(BaseModel):
    """Alert for suspicious transaction."""
    alert_type: str = Field(description='Type of suspicious activity detected')
    severity: RiskLevel = Field(description='Severity of the alert')
    description: str = Field(description='Detailed description of the issue')
    recommended_action: str = Field(description='Recommended action to take')


class ComplianceReport(BaseModel):
    """Final compliance report."""
    status: Literal['compliant', 'review_required', 'non_compliant'] = Field(
        description='Overall compliance status'
    )
    screening_result: str = Field(description='Summary of screening')
    risk_level: str = Field(description='Customer risk level')
    alerts_count: int = Field(description='Number of alerts generated')
    next_review_date: str = Field(description='When to review next')
    summary: str = Field(description='Executive summary')


# Agent 1: Sanctions Screening
screening_agent = Agent(
    'openai:gpt-4o',
    deps_type=AMLDependencies,
    output_type=ScreeningResult,
    system_prompt=(
        'You are a sanctions screening specialist. Analyze customer information '
        'and determine if they match any sanctions lists (OFAC, UN, EU). '
        'Check for Politically Exposed Persons (PEP) and adverse media. '
        'Return a structured assessment with risk level and recommended action.'
    ),
)


# Agent 2: Risk Assessment
risk_agent = Agent(
    'openai:gpt-4o',
    deps_type=AMLDependencies,
    output_type=RiskAssessmentResult,
    system_prompt=(
        'You are a risk assessment analyst. Evaluate customer risk based on: '
        'geographic location, business activity, transaction patterns, and '
        'screening results. Consider AML risk factors like high-risk jurisdictions, '
        'cash-intensive businesses, and unusual transaction patterns. '
        'Return a structured risk assessment with review frequency.'
    ),
)


# Agent 3: Transaction Monitoring
monitoring_agent = Agent(
    'openai:gpt-4o',
    deps_type=AMLDependencies,
    output_type=list[TransactionAlert],
    system_prompt=(
        'You are a transaction monitoring specialist. Analyze transaction history '
        'for suspicious patterns: structuring (smurfing), rapid movement of funds, '
        'transactions with high-risk jurisdictions, unusual amounts, and '
        'frequency anomalies. Return a list of alerts with severity and recommendations.'
    ),
)


# Agent 4: Compliance Reporting
report_agent = Agent(
    'openai:gpt-4o',
    deps_type=AMLDependencies,
    output_type=ComplianceReport,
    system_prompt=(
        'You are a compliance reporting specialist. Synthesize screening, '
        'risk assessment, and transaction monitoring results into a comprehensive '
        'compliance report. Determine overall compliance status and recommend '
        'next steps. Return a structured report suitable for regulatory filing.'
    ),
)


async def run_aml_compliance_check(deps: AMLDependencies) -> ComplianceReport:
    """Run full AML compliance workflow.
    
    Args:
        deps: Customer data and dependencies
        
    Returns:
        Complete compliance report
    """
    print(f'\n{"="*60}')
    print(f'AML Compliance Check')
    print(f'Customer: {deps.customer_name} (ID: {deps.customer_id})')
    print(f'{"="*60}\n')
    
    # Step 1: Sanctions Screening
    print('[1/4] Running sanctions screening...')
    screening = await screening_agent.run(
        f'Screen customer {deps.customer_name} from {deps.country} '
        f'engaged in {deps.business_activity}',
        deps=deps,
    )
    print(f'      Risk: {screening.output.risk_level}, Action: {screening.output.action}')
    
    # Step 2: Risk Assessment
    print('\n[2/4] Performing risk assessment...')
    risk = await risk_agent.run(
        f'Assess risk for {deps.customer_name} from {deps.country} '
        f'with screening result: {screening.output.risk_level}',
        deps=deps,
    )
    print(f'      Overall Risk: {risk.output.overall_risk}')
    print(f'      Review Frequency: {risk.output.review_frequency}')
    
    # Step 3: Transaction Monitoring
    print('\n[3/4] Analyzing transactions...')
    alerts = await monitoring_agent.run(
        f'Analyze {len(deps.transaction_history)} transactions for {deps.customer_name}',
        deps=deps,
    )
    print(f'      Alerts Generated: {len(alerts.output)}')
    for alert in alerts.output:
        print(f'      - {alert.alert_type} ({alert.severity})')
    
    # Step 4: Compliance Report
    print('\n[4/4] Generating compliance report...')
    report = await report_agent.run(
        f'Generate report for {deps.customer_name}. '
        f'Screening: {screening.output.risk_level}, '
        f'Risk: {risk.output.overall_risk}, '
        f'Alerts: {len(alerts.output)}',
        deps=deps,
    )
    
    print(f'\n{"="*60}')
    print(f'Compliance Status: {report.output.status.upper()}')
    print(f'{"="*60}\n')
    
    return report.output


async def main():
    """Run AML compliance demo."""
    # Demo customer 1: Low risk
    low_risk_customer = AMLDependencies(
        customer_id='CUST-001',
        customer_name='John Smith',
        country='US',
        business_activity='Software Consultant',
        transaction_history=[
            {'amount': 5000, 'date': '2026-04-01', 'jurisdiction': 'US'},
            {'amount': 3000, 'date': '2026-04-05', 'jurisdiction': 'US'},
            {'amount': 4500, 'date': '2026-04-10', 'jurisdiction': 'US'},
        ],
    )
    
    print('\n' + '='*70)
    print(' Demo 1: Low-Risk Customer')
    print('='*70)
    await run_aml_compliance_check(low_risk_customer)
    
    # Demo customer 2: Higher risk
    high_risk_customer = AMLDependencies(
        customer_id='CUST-002',
        customer_name='International Trading Corp',
        country='High-Risk Jurisdiction',
        business_activity='Import/Export',
        transaction_history=[
            {'amount': 50000, 'date': '2026-04-01', 'jurisdiction': 'Offshore'},
            {'amount': 48000, 'date': '2026-04-02', 'jurisdiction': 'Offshore'},
            {'amount': 52000, 'date': '2026-04-03', 'jurisdiction': 'Offshore'},
            {'amount': 10000, 'date': '2026-04-04', 'jurisdiction': 'US'},
        ],
    )
    
    print('\n' + '='*70)
    print(' Demo 2: Higher-Risk Customer')
    print('='*70)
    await run_aml_compliance_check(high_risk_customer)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
