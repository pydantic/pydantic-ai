# type: ignore
"""PydanticAI demonstration of the AI Access Control Four Perimeters Framework.

This implementation demonstrates the four access control perimeters framework with a practical example of a financial advisor agent:
1. Prompt Filtering: Ensures queries comply with financial advice regulations
2. Data Protection: Controls access to different levels of financial documentation
3. Secure External Access: Manages access to financial analysis capabilities
4. Response Enforcement: Enforces compliance requirements in responses

This demo uses Permit.io for fine-grained access control and PydanticAI for secure AI interactions.
"""

import os
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext

try:
    from permit import Permit
    from permit.exceptions import PermitApiError
except ImportError:
    pass

# Permit.io configuration from environment
PERMIT_KEY = os.environ.get('PERMIT_KEY', 'testkey')
if not PERMIT_KEY:
    raise ValueError('PERMIT_KEY environment variable not set')
PDP_URL = os.environ.get('PDP_URL', 'http://localhost:7766')


class SecurityError(Exception):
    """Custom exception for security-related errors."""

    pass


class UserContext(BaseModel):
    """User context containing identity and role information for permission checks."""

    user_id: str
    tier: str = Field(
        description="User's permission tier (opted_in_user, restricted_user, premium_user)"
    )


class FinancialDocument(BaseModel):
    """Model for financial documents with classification levels."""

    id: str
    type: str = Field(
        ..., description="Document type (e.g., 'investment', 'tax', 'retirement')"
    )
    content: str
    classification: str = Field(
        ...,
        description='Document classification level (public, restricted, confidential)',
    )


class FinancialQuery(BaseModel):
    """Input model for financial queries with context for permission checks."""

    question: str
    context: UserContext
    documents: Optional[list[FinancialDocument]] = None


class FinancialResponse(BaseModel):
    """Output model for financial advice with compliance tracking."""

    answer: str
    includes_advice: bool = Field(
        default=False, description='Indicates if response contains financial advice'
    )
    disclaimer_added: bool = Field(
        default=False, description='Tracks if regulatory disclaimer was added'
    )


@dataclass
class PermitDeps:
    """Dependencies for Permit.io integration."""

    permit: 'Permit | None'
    user_id: str

    def __post_init__(self):
        if not self.permit and Permit is not None:
            self.permit = Permit(
                token=PERMIT_KEY,
                pdp=PDP_URL,
            )


# Initialize the financial advisor agent with security focus
financial_agent = Agent(
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

    In a real implementation, this would use more sophisticated NLP/ML techniques.

    Args:
        question: The user's query text

    Returns:
        bool: True if the prompt is seeking financial advice, False if just information
    """
    # Simple keyword-based classification
    advice_keywords = [
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
) -> str | bool:
    """SECURITY PERIMETER 1: Prompt Filtering.

    Validates whether users have explicitly consented to receive AI-generated financial advice.
    Ensures compliance with financial regulations regarding automated advice systems.

    Key checks:
    - User has explicitly opted in to AI financial advice
    - Consent is properly recorded and verified
    - Classifies if the prompt is requesting advice

    Expected users in Permit.io:
    - opted_in_user (ai_advice_opted_in: true) - Has opted in to receive AI advice

    Args:
        ctx: Context containing Permit client and user ID
        query: The financial query to validate

    Returns:
        bool: True if user has consented to AI advice, False otherwise
    """
    try:
        # Classify if the prompt is requesting advice
        is_seeking_advice = classify_prompt_for_advice(query.question)

        permitted = await ctx.deps.permit.check(
            # The user object with their attributes
            {
                'key': ctx.deps.user_id,
            },
            # The action being performed
            'receive',
            # The resource being accessed
            {
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


@financial_agent.tool
async def access_financial_knowledge(
    ctx: RunContext[PermitDeps], usr: UserContext, documents: list[FinancialDocument]
) -> list[FinancialDocument]:
    """SECURITY PERIMETER 2: Data Protection.

    Controls access to financial knowledge base and documentation based on user permissions
    and document classification levels. Implements information barriers and content restrictions.

    Key aspects:
    - Document classification levels (public, restricted, confidential)
    - User clearance level verification
    - Regulatory compliance for information access

    Args:
        ctx: Context containing Permit client and user ID
        usr: User context
        documents: List of financial documents to filter

    Returns:
        List[FinancialDocument]: Filtered list of documents user is allowed to access
    """
    try:
        # Create resource instances for each document
        resources = [
            {
                'id': doc.id,
                'type': 'financial_document',
                'attributes': {
                    'doc_type': doc.type,
                    'classification': doc.classification,
                },
            }
            for doc in documents
        ]

        # Use Permit's filter_objects to get allowed documents
        allowed_docs = await ctx.deps.permit.filter_objects(
            ctx.deps.user_id, 'read', {}, resources
        )
        # Return only the documents that were allowed
        allowed_ids = {doc['id'] for doc in allowed_docs}
        return [doc for doc in documents if doc.id in allowed_ids]

    except PermitApiError as e:
        raise SecurityError(f'Failed to filter documents: {str(e)}')


@financial_agent.tool
async def check_action_permissions(
    ctx: RunContext[PermitDeps], action: str, context: UserContext, portfolio_id: str
) -> bool:
    """SECURITY PERIMETER 3: Secure External Access.

    Controls permissions for sensitive financial operations, particularly portfolio
    modifications. Ensures only authorized users can perform account-level changes.

    Key checks:
    - Portfolio ownership verification
    - External API access control


    Args:
        ctx: Context containing Permit client and user ID
        action: The type of action being verified
        context: User context containing identity and tier information
        portfolio_id: Identifier of the portfolio to update

    Returns:
        bool: True if user is authorized to update portfolio, False otherwise
    """
    try:
        return await ctx.deps.permit.check(
            ctx.deps.user_id,
            'update',
            {
                'type': 'portfolio',
            },
        )
    except PermitApiError as e:
        raise SecurityError(f'Failed to check portfolio update permission: {str(e)}')


def classify_response_for_advice(response_text: str) -> bool:
    """Mock classifier that checks if the response contains financial advice.

    In a real implementation, this would use:
    - NLP to detect advisory language patterns
    - ML models trained on financial advice datasets


    Args:
        response_text: The AI-generated response text

    Returns:
        bool: True if the response contains financial advice, False if just information
    """
    # Simple keyword-based classification
    advice_indicators = [
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
    """SECURITY PERIMETER 4: Response Enforcement.

    Ensures all financial advice responses meet regulatory requirements and include
    necessary disclaimers.

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
        permitted = await ctx.deps.permit.check(
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


# Initialize example documents
SAMPLE_DOCUMENTS = {
    'inv_001': FinancialDocument(
        id='inv_001',
        type='investment',
        content='Tech Growth Fund performance analysis shows a 15% YoY return...',
        classification='confidential',
    ),
    'tax_001': FinancialDocument(
        id='tax_001',
        type='tax',
        content='Tax optimization strategies for high-income investors...',
        classification='restricted',
    ),
    'ret_001': FinancialDocument(
        id='ret_001',
        type='retirement',
        content='401(k) contribution strategies and employer matching...',
        classification='public',
    ),
    'inv_002': FinancialDocument(
        id='inv_002',
        type='investment',
        content='ESG Fund analysis and sustainable investment opportunities...',
        classification='public',
    ),
}


# Example usage
async def main():
    # Initialize Permit client
    permit = Permit(
        token=PERMIT_KEY,
        pdp=PDP_URL,
    )

    # Create security context for the user (this user has been created during setup)
    deps = PermitDeps(permit=permit, user_id='user@example.com')

    try:
        # Example: Process a financial query
        result = await financial_agent.run(
            'Can you suggest some basic investment strategies for beginners?',
            deps=deps,
        )
        print(f'Secure response: {result.data}')

        # Example: Access to protected documentation
        docs_result = await financial_agent.run(
            "Please check my access level for tax documents and tell me what I'm permitted to see.",
            deps=deps,
        )
        print(f'Protected document access: {docs_result.data}')

    except SecurityError as e:
        print(f'Security check failed: {str(e)}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
