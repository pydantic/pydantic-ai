"""
Test script to verify the code examples in the blog article work correctly.
This script tests the structure and imports, not the actual LLM calls.
"""

import asyncio
from typing import Any


def test_example_1_basic_dataset():
    """Test: Basic LLMJudge usage from Section 2."""
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import LLMJudge

    dataset = Dataset(
        cases=[
            Case(
                name='password_reset_help',
                inputs='How do I reset my password?',
            ),
        ],
        evaluators=[
            LLMJudge(rubric='The response provides clear, actionable instructions'),
        ],
    )

    assert len(dataset.cases) == 1
    assert dataset.cases[0].name == 'password_reset_help'
    print('Example 1 (basic dataset): PASS')


def test_example_2_one_size_fits_all():
    """Test: One-size-fits-all evaluator from Section 4."""
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import LLMJudge

    dataset = Dataset(
        cases=[
            Case(name='billing_question', inputs={'query': 'Why was I charged twice?'}),
            Case(name='feature_request', inputs={'query': 'Can you add dark mode?'}),
            Case(name='bug_report', inputs={'query': 'The app crashes when I upload photos'}),
        ],
        evaluators=[
            LLMJudge(
                rubric='Response is professional, empathetic, and does not blame the user',
                include_input=True,
            ),
        ],
    )

    assert len(dataset.cases) == 3
    assert len(dataset.evaluators) == 1
    print('Example 2 (one-size-fits-all): PASS')


def test_example_3_case_specific():
    """Test: Case-specific evaluators from Section 4."""
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import LLMJudge

    dataset = Dataset(
        cases=[
            Case(
                name='vegetarian_recipe',
                inputs={'request': 'I need a vegetarian dinner recipe'},
                evaluators=[
                    LLMJudge(
                        rubric='''
                        Recipe must NOT contain:
                        - Meat (beef, pork, chicken, fish, etc.)
                        - Meat-based broths or stocks
                        - Gelatin or other animal-derived ingredients
                        PASS only if the recipe is fully vegetarian.
                        ''',
                        include_input=True,
                    ),
                ],
            ),
            Case(
                name='quick_weeknight_meal',
                inputs={'request': 'I need something I can make in under 30 minutes'},
                evaluators=[
                    LLMJudge(
                        rubric='''
                        Recipe must:
                        - Have total prep + cook time under 30 minutes
                        - Use commonly available ingredients
                        - Not require specialized equipment
                        FAIL if the recipe would realistically take longer than 30 minutes.
                        ''',
                        include_input=True,
                    ),
                ],
            ),
            Case(
                name='allergy_safe',
                inputs={'request': 'Nut-free dessert for a school event'},
                evaluators=[
                    LLMJudge(
                        rubric='''
                        Recipe must NOT contain:
                        - Tree nuts (almonds, walnuts, pecans, etc.)
                        - Peanuts or peanut-derived ingredients
                        - "May contain nuts" warnings should be noted as a concern
                        This is for a child with allergies. FAIL if any nut risk exists.
                        ''',
                        include_input=True,
                    ),
                ],
            ),
        ],
        evaluators=[
            LLMJudge(
                rubric='Recipe instructions are clear and easy to follow',
                include_input=True,
            ),
        ],
    )

    assert len(dataset.cases) == 3
    # Check that each case has its own evaluator
    assert len(dataset.cases[0].evaluators) == 1
    assert len(dataset.cases[1].evaluators) == 1
    assert len(dataset.cases[2].evaluators) == 1
    # Check dataset-level evaluator
    assert len(dataset.evaluators) == 1
    print('Example 3 (case-specific): PASS')


def test_example_4_multi_criteria():
    """Test: Multi-criteria evaluation from Section 5."""
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import LLMJudge

    dataset = Dataset(
        cases=[
            Case(
                name='explain_concept',
                inputs='Explain how photosynthesis works',
            ),
        ],
        evaluators=[
            LLMJudge(
                rubric='The explanation is scientifically accurate',
                assertion={'evaluation_name': 'accuracy', 'include_reason': True},
            ),
            LLMJudge(
                rubric='The explanation is understandable to a middle school student',
                assertion={'evaluation_name': 'clarity', 'include_reason': True},
            ),
            LLMJudge(
                rubric='The explanation is concise without sacrificing completeness',
                assertion={'evaluation_name': 'conciseness', 'include_reason': True},
            ),
        ],
    )

    assert len(dataset.evaluators) == 3
    print('Example 4 (multi-criteria): PASS')


def test_example_5_reference_based():
    """Test: Reference-based evaluation (hallucination check) from Section 5."""
    from pydantic import BaseModel
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import LLMJudge

    class RAGInput(BaseModel):
        question: str
        retrieved_context: str

    dataset = Dataset(
        cases=[
            Case(
                name='policy_question',
                inputs=RAGInput(
                    question='What is the return policy?',
                    retrieved_context='''
                    Our return policy:
                    - Items can be returned within 30 days of purchase
                    - Items must be unused and in original packaging
                    - Refunds are processed within 5-7 business days
                    - Sale items are final sale and cannot be returned
                    ''',
                ),
                evaluators=[
                    LLMJudge(
                        rubric='''
                        The response must ONLY contain information from the retrieved context.
                        Check for:
                        1. Any facts not present in the context (hallucination)
                        2. Any numbers or timeframes that differ from the context
                        3. Any policies or claims not stated in the context
                        FAIL if the response includes ANY information not in the context.
                        ''',
                        include_input=True,
                        assertion={'evaluation_name': 'grounded', 'include_reason': True},
                    ),
                ],
            ),
        ],
    )

    assert len(dataset.cases) == 1
    assert isinstance(dataset.cases[0].inputs, RAGInput)
    print('Example 5 (reference-based): PASS')


def test_example_6_deterministic_combined():
    """Test: Combining deterministic and LLM evaluators from Section 7."""
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import LLMJudge, IsInstance

    dataset = Dataset(
        cases=[
            Case(
                name='refund_request',
                inputs='I want a refund for my order',
            ),
        ],
        evaluators=[
            IsInstance(type_name='str'),
            LLMJudge(
                rubric='Response acknowledges the refund request and provides next steps',
                include_input=True,
            ),
        ],
    )

    assert len(dataset.evaluators) == 2
    print('Example 6 (deterministic + LLM): PASS')


def test_example_7_rubric_config():
    """Test: Rubric configuration options from Section 8."""
    from pydantic_ai.settings import ModelSettings
    from pydantic_evals.evaluators import LLMJudge

    # Basic assertion
    judge1 = LLMJudge(rubric='Response is helpful')

    # Good rubric with criteria
    judge2 = LLMJudge(
        rubric='''
        The response must:
        1. Directly address the user's question (not a tangential topic)
        2. Provide at least one actionable next step
        3. Not require the user to ask follow-up questions to get basic information

        PASS if all three criteria are met.
        FAIL if any criterion is not met, and explain which one(s).
        ''',
        include_input=True,
        assertion={'evaluation_name': 'helpful', 'include_reason': True},
    )

    # Binary assertion
    judge3 = LLMJudge(
        rubric='Response follows company guidelines',
        assertion={'evaluation_name': 'compliant', 'include_reason': True},
        score=False,
    )

    # Numeric score
    judge4 = LLMJudge(
        rubric='Rate the helpfulness of the response from 0 to 1',
        score={'evaluation_name': 'helpfulness', 'include_reason': True},
        assertion=False,
    )

    # Both score and assertion
    judge5 = LLMJudge(
        rubric='''
        Rate the accuracy from 0 to 1.
        PASS if accuracy >= 0.8, FAIL otherwise.
        ''',
        score={'evaluation_name': 'accuracy_score'},
        assertion={'evaluation_name': 'accuracy_pass'},
    )

    # With model settings
    judge6 = LLMJudge(
        rubric='Test rubric',
        model='openai:gpt-4o',
        model_settings=ModelSettings(temperature=0.0),
    )

    print('Example 7 (rubric config): PASS')


def test_example_8_context_config():
    """Test: Context configuration from Section 8."""
    from pydantic_evals.evaluators import LLMJudge

    # Output only
    judge1 = LLMJudge(
        rubric='Response uses professional language without slang or emojis',
        include_input=False,
        include_expected_output=False,
    )

    # Input + Output
    judge2 = LLMJudge(
        rubric='Response directly addresses the question that was asked',
        include_input=True,
    )

    # Input + Output + Expected
    judge3 = LLMJudge(
        rubric='''
        Compare the response to the expected answer.
        PASS if they are semantically equivalent (same meaning, different wording OK).
        FAIL if they contradict or the response is missing key information.
        ''',
        include_input=True,
        include_expected_output=True,
    )

    print('Example 8 (context config): PASS')


def test_example_9_style_evaluators():
    """Test: Style evaluators from Section 6."""
    from pydantic_evals.evaluators import LLMJudge

    style_evaluators = [
        LLMJudge(
            rubric='Response uses second person ("you", "your") instead of third person ("the user", "they")',
            assertion={'evaluation_name': 'uses_second_person'},
        ),
        LLMJudge(
            rubric='Response does not start with "I" or make the assistant the subject of the first sentence',
            assertion={'evaluation_name': 'not_self_focused'},
        ),
        LLMJudge(
            rubric='''
            Response avoids corporate buzzwords and jargon including:
            - "leverage", "synergy", "paradigm", "holistic"
            - "circle back", "move the needle", "low-hanging fruit"
            - Unnecessary acronyms without explanation
            ''',
            assertion={'evaluation_name': 'no_jargon'},
        ),
    ]

    assert len(style_evaluators) == 3
    print('Example 9 (style evaluators): PASS')


def test_example_10_dataset_building():
    """Test: Building a dataset with metadata from Section 9."""
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import LLMJudge

    dataset = Dataset(
        cases=[
            Case(
                name='password_reset',
                inputs={'query': 'How do I reset my password?'},
                metadata={'category': 'account', 'priority': 'high'},
                evaluators=[
                    LLMJudge(
                        rubric='''
                        Response must:
                        1. Provide password reset instructions OR link
                        2. Mention email verification step
                        NOT acceptable:
                        - Asking for current password (security risk)
                        ''',
                        include_input=True,
                    ),
                ],
            ),
            Case(
                name='billing_dispute',
                inputs={'query': 'I was charged twice for my subscription'},
                metadata={'category': 'billing', 'priority': 'high'},
                evaluators=[
                    LLMJudge(
                        rubric='''
                        Response must:
                        1. Acknowledge the double charge concern
                        2. Offer to investigate or escalate
                        3. Provide a timeline for resolution
                        NOT acceptable:
                        - Dismissing the concern
                        - Asking the customer to wait without a timeline
                        ''',
                        include_input=True,
                    ),
                ],
            ),
        ],
        evaluators=[
            LLMJudge(rubric='Response is professional and empathetic'),
        ],
    )

    assert len(dataset.cases) == 2
    assert dataset.cases[0].metadata == {'category': 'account', 'priority': 'high'}
    print('Example 10 (dataset building): PASS')


def test_example_11_user_feedback_cases():
    """Test: User feedback cases from Section 11."""
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import LLMJudge

    feedback_cases = [
        Case(
            name='refund_eligibility_false_negative',
            inputs={
                'query': 'I bought this 2 weeks ago and it broke. Can I get a refund?',
            },
            metadata={
                'source': 'user_feedback',
                'ticket_id': '12345',
                'issue': 'incorrectly denied refund',
            },
            evaluators=[
                LLMJudge(
                    rubric='''
                    Context: Purchase was 2 weeks ago (within 30-day return window).
                    Product is defective.

                    Response MUST:
                    1. Confirm the customer is eligible for a refund
                    2. Provide return/refund instructions

                    FAIL if refund is denied or eligibility is unclear.
                    ''',
                    include_input=True,
                ),
            ],
        ),
        Case(
            name='order_number_recognition',
            inputs={
                'query': 'My order number is ABC-12345. Where is my package?',
            },
            metadata={
                'source': 'user_feedback',
                'ticket_id': '12389',
                'issue': 'failed to recognize order number',
            },
            evaluators=[
                LLMJudge(
                    rubric='''
                    The user has provided their order number (ABC-12345).

                    Response MUST:
                    1. Acknowledge the order number
                    2. Provide tracking or status information

                    Response must NOT:
                    - Ask for the order number again
                    - Say the order number is invalid without checking

                    FAIL if the bot asks for information already provided.
                    ''',
                    include_input=True,
                ),
            ],
        ),
        Case(
            name='empathetic_delay_response',
            inputs={
                'query': 'Why is my order taking so long? This is ridiculous.',
            },
            metadata={
                'source': 'user_feedback',
                'ticket_id': '12401',
                'issue': 'response perceived as rude',
            },
            evaluators=[
                LLMJudge(
                    rubric='''
                    The customer is frustrated about a shipping delay.

                    Response MUST:
                    1. Acknowledge the frustration empathetically
                    2. Apologize for the delay
                    3. Provide information about the order status or next steps

                    Response must NOT:
                    1. Be defensive or dismissive
                    2. Blame the customer or external factors without offering help
                    3. Use curt or cold language

                    FAIL if the response lacks empathy or feels dismissive.
                    ''',
                    include_input=True,
                ),
            ],
        ),
    ]

    dataset = Dataset(cases=feedback_cases)
    assert len(dataset.cases) == 3
    print('Example 11 (user feedback cases): PASS')


def test_example_12_complete_agent_dataset():
    """Test: Complete agent dataset structure from Section 12."""
    from pydantic import BaseModel
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import LLMJudge, IsInstance

    class SupportQuery(BaseModel):
        query: str
        user_id: str
        account_tier: str

    class SupportResponse(BaseModel):
        message: str
        suggested_articles: list[str]
        escalate_to_human: bool

    support_dataset = Dataset(
        cases=[
            Case(
                name='enterprise_escalation',
                inputs=SupportQuery(
                    query='My integration is broken and I need help urgently',
                    user_id='user-123',
                    account_tier='enterprise',
                ),
                evaluators=[
                    LLMJudge(
                        rubric='''
                        This is an enterprise user with an urgent issue.
                        Response MUST offer to escalate to human support.
                        FAIL if escalation is not offered.
                        ''',
                        include_input=True,
                    ),
                ],
            ),
            Case(
                name='free_tier_phone_support',
                inputs=SupportQuery(
                    query='Can I get phone support?',
                    user_id='user-456',
                    account_tier='free',
                ),
                evaluators=[
                    LLMJudge(
                        rubric='''
                        This is a free tier user asking about phone support.
                        Phone support is only available for Pro and Enterprise.

                        Response MUST:
                        - Explain that phone support requires an upgrade
                        - Be polite and not dismissive
                        - Offer alternative support options

                        FAIL if it implies free users can get phone support.
                        ''',
                        include_input=True,
                    ),
                ],
            ),
            Case(
                name='order_status_inquiry',
                inputs=SupportQuery(
                    query='Where is my order ORD-789? It was supposed to arrive yesterday.',
                    user_id='user-789',
                    account_tier='pro',
                ),
                evaluators=[
                    LLMJudge(
                        rubric='''
                        User is asking about order ORD-789.

                        Response MUST:
                        - Reference the specific order number
                        - Provide status information
                        - Acknowledge the delay concern empathetically

                        FAIL if the order is not looked up or the response is generic.
                        ''',
                        include_input=True,
                    ),
                ],
            ),
        ],
        evaluators=[
            IsInstance(type_name='SupportResponse'),
            LLMJudge(
                rubric='Response is friendly, professional, and helpful',
                assertion={'evaluation_name': 'professional_tone'},
            ),
            LLMJudge(
                rubric='Response does not reveal sensitive account information without verification',
                assertion={'evaluation_name': 'security_conscious'},
            ),
        ],
    )

    assert len(support_dataset.cases) == 3
    assert len(support_dataset.evaluators) == 3
    print('Example 12 (complete agent dataset): PASS')


def test_set_default_judge_model():
    """Test: Setting default judge model."""
    from pydantic_evals.evaluators.llm_as_a_judge import set_default_judge_model

    # This should not raise
    set_default_judge_model('anthropic:claude-sonnet-4-5')
    print('set_default_judge_model: PASS')


if __name__ == '__main__':
    print('Testing blog article code examples...\n')

    test_example_1_basic_dataset()
    test_example_2_one_size_fits_all()
    test_example_3_case_specific()
    test_example_4_multi_criteria()
    test_example_5_reference_based()
    test_example_6_deterministic_combined()
    test_example_7_rubric_config()
    test_example_8_context_config()
    test_example_9_style_evaluators()
    test_example_10_dataset_building()
    test_example_11_user_feedback_cases()
    test_example_12_complete_agent_dataset()
    test_set_default_judge_model()

    print('\nAll examples passed!')
