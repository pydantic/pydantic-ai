"""Example of integrating StackOne tools with Pydantic AI.

This example demonstrates how to use StackOne's unified API platform
to access HRIS, ATS, CRM, and other business systems through Pydantic AI agents.
"""

import asyncio
import os

from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.ext.stackone import StackOneToolset, tool_from_stackone


# Example 1: Using a single StackOne tool
def single_tool_example():
    """Example using a single StackOne tool."""
    print('=== Single Tool Example ===')

    # Create a single tool for listing employees
    employee_tool = tool_from_stackone(
        'hris_list_employees',
        account_id=os.getenv('STACKONE_ACCOUNT_ID', 'demo-account'),
        api_key=os.getenv('STACKONE_API_KEY', 'demo-key'),
    )

    # Create an agent with the single tool
    agent = Agent(
        'openai:gpt-4o-mini',  # Use a smaller model for examples
        tools=[employee_tool],
        system_prompt='You are an HR assistant. Help users with employee-related queries using StackOne HRIS tools.',
    )

    try:
        result = agent.run_sync('List all employees in the system')
        print(f'Result: {result.output}')
    except Exception as e:
        print(f'Error: {e}')


# Example 2: Using multiple StackOne tools with glob patterns
def multiple_tools_example():
    """Example using multiple StackOne tools with patterns."""
    print('\n=== Multiple Tools Example ===')

    # Create a toolset with glob patterns
    toolset = StackOneToolset(
        ['hris_*', '!hris_delete_*'],  # Include all HRIS tools except delete operations
        account_id=os.getenv('STACKONE_ACCOUNT_ID', 'demo-account'),
        api_key=os.getenv('STACKONE_API_KEY', 'demo-key'),
    )

    # Create an agent with the toolset
    agent = Agent(
        'openai:gpt-4o-mini',
        toolsets=[toolset],
        system_prompt="""
        You are an HR assistant with access to comprehensive HRIS tools.
        You can list employees, get employee details, create new employees,
        update existing employees, and manage departments.
        Always be helpful and provide detailed information when available.
        """,
    )

    try:
        result = agent.run_sync(
            'Get information about all employees and then show me details about the first employee'
        )
        print(f'Result: {result.output}')
    except Exception as e:
        print(f'Error: {e}')


# Example 3: Using StackOne with structured output
class EmployeeReport(BaseModel):
    total_employees: int
    departments: list[str]
    summary: str


def structured_output_example():
    """Example using StackOne tools with structured output."""
    print('\n=== Structured Output Example ===')

    # Create a toolset for HRIS operations
    toolset = StackOneToolset(
        'hris_*',  # All HRIS tools
        account_id=os.getenv('STACKONE_ACCOUNT_ID', 'demo-account'),
        api_key=os.getenv('STACKONE_API_KEY', 'demo-key'),
    )

    # Create an agent that returns structured data
    agent = Agent(
        'openai:gpt-4o-mini',
        toolsets=[toolset],
        output_type=EmployeeReport,
        system_prompt="""
        You are an HR analytics assistant. Use the StackOne HRIS tools to gather 
        employee data and provide structured reports.
        """,
    )

    try:
        result = agent.run_sync(
            'Create a report showing the total number of employees, '
            'list of departments, and a brief summary of the HR data'
        )
        print(f'Structured Result: {result.output}')
        print(f'Total Employees: {result.output.total_employees}')
        print(f'Departments: {result.output.departments}')
        print(f'Summary: {result.output.summary}')
    except Exception as e:
        print(f'Error: {e}')


# Example 4: Async usage with StackOne tools
async def async_example():
    """Example using StackOne tools asynchronously."""
    print('\n=== Async Example ===')

    toolset = StackOneToolset(
        ['hris_list_employees', 'hris_get_employee'],
        account_id=os.getenv('STACKONE_ACCOUNT_ID', 'demo-account'),
        api_key=os.getenv('STACKONE_API_KEY', 'demo-key'),
    )

    agent = Agent(
        'openai:gpt-4o-mini',
        toolsets=[toolset],
        system_prompt='You are an HR assistant. Help with employee queries efficiently.',
    )

    try:
        result = await agent.run(
            'Find the employee with the most recent hire date and show their details'
        )
        print(f'Async Result: {result.output}')
    except Exception as e:
        print(f'Async Error: {e}')


# Example 5: Using StackOne with different business systems
def multi_system_example():
    """Example using StackOne tools across different business systems."""
    print('\n=== Multi-System Example ===')

    # Create toolsets for different systems
    hris_toolset = StackOneToolset(
        'hris_*',
        account_id=os.getenv('STACKONE_HRIS_ACCOUNT_ID', 'demo-hris-account'),
        api_key=os.getenv('STACKONE_API_KEY', 'demo-key'),
    )

    ats_toolset = StackOneToolset(
        'ats_*',
        account_id=os.getenv('STACKONE_ATS_ACCOUNT_ID', 'demo-ats-account'),
        api_key=os.getenv('STACKONE_API_KEY', 'demo-key'),
    )

    # Create an agent with multiple toolsets
    agent = Agent(
        'openai:gpt-4o-mini',
        toolsets=[hris_toolset, ats_toolset],
        system_prompt="""
        You are a comprehensive HR and recruiting assistant with access to both 
        HRIS (Human Resources) and ATS (Applicant Tracking System) tools.
        You can help with employee management, recruitment, and hiring processes.
        """,
    )

    try:
        result = agent.run_sync(
            'Show me the current employees and any open job positions'
        )
        print(f'Multi-System Result: {result.output}')
    except Exception as e:
        print(f'Multi-System Error: {e}')


def main():
    """Run all examples."""
    print('StackOne Integration Examples')
    print('=' * 40)

    # Check if API key is set
    if not os.getenv('STACKONE_API_KEY'):
        print('Warning: STACKONE_API_KEY environment variable not set.')
        print('Using demo values - actual API calls may fail.')
        print()

    # Run synchronous examples
    single_tool_example()
    multiple_tools_example()
    structured_output_example()
    multi_system_example()

    # Run async example
    print('\nRunning async example...')
    asyncio.run(async_example())

    print('\n' + '=' * 40)
    print('All examples completed!')


if __name__ == '__main__':
    main()
