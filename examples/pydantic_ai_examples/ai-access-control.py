"""PydanticAI demonstration of structured prompt/response of fine grained access control.

This demo uses Permit.io for fine-grained access control and PydanticAI for secure AI interactions.
"""

import os
from dataclasses import dataclass
from typing import Final, Literal, Optional, Union

from permit import Permit
from permit.exceptions import PermitApiError
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext

# Permit.io configuration from environment
PERMIT_KEY: Final[str] = os.environ.get('PERMIT_KEY', 'test_key')
if not PERMIT_KEY:
    raise ValueError('PERMIT_KEY environment variable not set')
PDP_URL: Final[str] = os.environ.get('PDP_URL', 'http://localhost:7766')


class SecurityError(Exception):
    """Custom exception for security-related errors."""

    pass


class UserContext(BaseModel):
    """User context containing identity and role information for permission checks."""

    user_id: str
    tier: Literal['opted_in_user', 'restricted_user', 'premium_user'] = Field(
        description="User's permission tier"
    )


class FinancialQuery(BaseModel):
    """Input model for financial queries with context for permission checks."""

    question: str
    context: UserContext


class FinancialResponse(BaseModel):
    """Output model for financial advice with compliance tracking."""

    answer: str
    includes_advice: bool = Field(
        default=False, description='Indicates if response contains financial advice'
    )
    disclaimer_added: bool = Field(
        default=False, description='Tracks if regulatory disclaimer was added'
    )
    metadata: Optional[dict[str, str]] = Field(
        default=None, description='Additional metadata about the response'
    )


@dataclass
class PermitDeps:
    """Dependencies for Permit.io integration."""

    permit: Permit
    user_id: str

    def __post_init__(self) -> None:
        if not self.permit:
            self.permit = Permit(
                token=PERMIT_KEY,
                pdp=PDP_URL,
            )


# Initialize the financial advisor agent with security focus
financial_agent = Agent[PermitDeps, FinancialResponse](
    'anthropic:claude-3-5-sonnet-latest',
    deps_type=PermitDeps,
    result_type=FinancialResponse,
    system_prompt='You are a financial advisor. Follow these steps in order:'
    '1. ALWAYS check user permissions first'
    '2. Only proceed with advice if user has opted into AI advice'
    '3. Only attempt document access if user has required permissions',
)


def classify_prompt_for_advice(question: str) -> bool:
    """Mock classifier that checks if the prompt is requesting financial advice.

    In a real implementation, this can be upgraded to use more sophisticated NLP/ML techniques.

    Args:
        question: The user's query text

    Returns:
        bool: True if the prompt is seeking financial advice, False if just information
    """
    # Simple keyword-based classification
    advice_keywords: Final[list[str]] = [
        'should i',
        'recommend',
        'advice',
        'suggest',
        'help me',
        "what's best",
        'what is best',
        'better option',
    ]

    question_lower = question.lower()
    return any(keyword in question_lower for keyword in advice_keywords)


@financial_agent.tool
async def validate_financial_query(
    ctx: RunContext[PermitDeps],
    query: FinancialQuery,
) -> Union[bool, str]:
    """Validates whether users have explicitly consented to receive AI-generated financial advice.

    Ensures compliance with financial regulations regarding automated advice systems.

    Key checks:
    - User has explicitly opted in to AI financial advice
    - Consent is properly recorded and verified
    - Classifies if the prompt is requesting advice

    Args:
        ctx: Context containing Permit client and user ID
        query: The financial query to validate

    Returns:
        bool: True if user has consented to AI advice, False otherwise
    """
    try:
        # Classify if the prompt is requesting advice
        is_seeking_advice = classify_prompt_for_advice(query.question)

        permitted = await ctx.deps.permit.check(  # type: ignore
            user=ctx.deps.user_id,
            action='receive',
            resource={
                'type': 'financial_advice',
                'attributes': {'is_ai_generated': is_seeking_advice},
            },
        )

        if not permitted:
            if is_seeking_advice:
                return 'User has not opted in to receive AI-generated financial advice'
            else:
                return 'User does not have permission to access this information'

        return True

    except PermitApiError as e:
        raise SecurityError(f'Permission check failed: {str(e)}')


def classify_response_for_advice(response_text: str) -> bool:
    """Mock classifier that checks if the response contains financial advice.

    In a real implementation, this could be upgraded to use:
    - NLP to detect advisory language patterns
    - ML models trained on financial advice datasets

    Args:
        response_text: The AI-generated response text

    Returns:
        bool: True if the response contains financial advice, False if just information
    """
    # Simple keyword-based classification
    advice_indicators: Final[list[str]] = [
        'recommend',
        'should',
        'consider',
        'advise',
        'suggest',
        'better to',
        'optimal',
        'best option',
        'strategy',
        'allocation',
    ]

    response_lower = response_text.lower()
    return any(indicator in response_lower for indicator in advice_indicators)


@financial_agent.tool
async def validate_financial_response(
    ctx: RunContext[PermitDeps], response: FinancialResponse
) -> FinancialResponse:
    """Ensures all financial advice responses meet regulatory requirements and include necessary disclaimers.

    Key features:
    - Automated advice detection using content classification
    - Regulatory disclaimer enforcement
    - Compliance verification and auditing

    Args:
        ctx: Context containing Permit client and user ID
        response: The financial response to validate

    Returns:
        FinancialResponse: Validated and compliant response
    """
    try:
        # Classify if response contains financial advice
        contains_advice = classify_response_for_advice(response.answer)

        # Check if user is allowed to receive this type of response
        permitted = await ctx.deps.permit.check(  # type: ignore
            ctx.deps.user_id,
            'requires_disclaimer',
            {
                'type': 'financial_response',
                'attributes': {'contains_advice': str(contains_advice)},
            },
        )

        if contains_advice and permitted:
            disclaimer = (
                '\n\nIMPORTANT DISCLAIMER: This is AI-generated financial advice. '
                'This information is for educational purposes only and should not be '
                'considered as professional financial advice. Always consult with a '
                'qualified financial advisor before making investment decisions.'
            )
            response.answer += disclaimer
            response.disclaimer_added = True
            response.includes_advice = True

        return response

    except PermitApiError as e:
        raise SecurityError(f'Failed to check response content: {str(e)}')


# Example usage
async def main() -> None:
    """Run example usage of the financial advisor agent."""
    # Initialize Permit client
    permit: Permit = Permit(
        token=PERMIT_KEY,
        pdp=PDP_URL,
    )

    # Create security context for the user
    deps: PermitDeps = PermitDeps(permit=permit, user_id='user@example.com')

    try:
        # Example: Process a financial query
        # Expected response:
        # - If user doesn't have permission:
        #   Secure response: answer="I apologize, but I cannot provide investment strategy suggestions at this time because you have not explicitly opted in to receive AI-generated financial advice. This is a regulatory requirement designed to protect consumers. \n\nTo receive investment advice, you would need to:\n1. Explicitly opt in to receive AI-generated financial advice\n2. Acknowledge the associated disclaimers and risks\n 3. Update your user preferences accordingly\n\Please contact your financial institution or platform administrator to update your preference s if you wish to receive AI-generated investment advice. Once you have opted in, I'll be happy to provide basic investment strategy suggesti ons." includes_advice=False disclaimer_added=False
        result = await financial_agent.run(
            'Can you suggest some basic investment strategies for beginners?',
            deps=deps,
        )
        print(f'Secure response: {result.data}')

    except SecurityError as e:
        print(f'Security check failed: {str(e)}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
