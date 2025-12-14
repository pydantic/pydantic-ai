"""Minimal example to test SearchableToolset functionality.

Run with: uv run python test_searchable_example.py
Make sure you have ANTHROPIC_API_KEY set in your environment.
"""

import asyncio
import logging
import sys

# Configure logging to print to stdout
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Silence noisy loggers
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore.connection').setLevel(logging.WARNING)
logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
logging.getLogger('anthropic._base_client').setLevel(logging.WARNING)

from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.searchable import SearchableToolset


# Create a toolset with various tools
toolset = FunctionToolset()


@toolset.tool(defer_loading=True)
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: The name of the city to get weather for.
    """
    return f"The weather in {city} is sunny and 72°F"


@toolset.tool(defer_loading=True)
def calculate_sum(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


@toolset.tool(defer_loading=True)
def calculate_product(a: float, b: float) -> float:
    """Multiply two numbers together.

    Args:
        a: The first number.
        b: The second number.
    """
    return a * b


@toolset.tool(defer_loading=True)
def fetch_user_data(user_id: int) -> dict:
    """Fetch user data from the database.

    Args:
        user_id: The ID of the user to fetch.
    """
    return {"id": user_id, "name": "John Doe", "email": "john@example.com"}


@toolset.tool(defer_loading=True)
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    Args:
        recipient: The email address of the recipient.
        subject: The subject line of the email.
        body: The body content of the email.
    """
    return f"Email sent to {recipient} with subject '{subject}'"


@toolset.tool(defer_loading=True)
def list_database_tables() -> list[str]:
    """List all tables in the database."""
    return ["users", "orders", "products", "reviews"]


# Wrap the toolset with SearchableToolset
searchable_toolset = SearchableToolset(toolset=toolset)

# Create an agent with the searchable toolset
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    toolsets=[searchable_toolset],
    system_prompt=(
        "You are a helpful assistant."
    ),
)


async def main():
    print("=" * 60)
    print("Testing SearchableToolset")
    print("=" * 60)
    print()

    # Test 1: Ask something that requires searching for calculation tools
    print("Test 1: Calculation task")
    print("-" * 60)
    result = await agent.run("What is 123 multiplied by 456?")
    print(f"Result: {result.output}")
    print()

    # Test 2: Ask something that requires searching for database tools
    print("\nTest 2: Database task")
    print("-" * 60)
    result = await agent.run("Can you list the database tables and then fetch user 42?")
    print(f"Result: {result.output}")
    print()

    # Test 3: Ask something that requires weather tool
    print("\nTest 3: Weather task")
    print("-" * 60)
    result = await agent.run("What's the weather like in San Francisco? Search for a weather tool")
    print(f"Result: {result.output}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
