"""Shared fixtures and test tools for code mode tests."""

from __future__ import annotations

from typing import Any, TypedDict


# Define return type as TypedDict for better type hints in signatures
# Note: We return a plain dict at runtime for Monty compatibility
class WeatherResult(TypedDict):
    """Weather data for a city."""

    city: str
    temp: float
    unit: str
    conditions: str


# Simulated weather data for test cities
_WEATHER_DATA: dict[str, dict[str, Any]] = {
    'London': {'temp': 15.0, 'conditions': 'cloudy'},
    'Paris': {'temp': 18.0, 'conditions': 'sunny'},
    'Tokyo': {'temp': 22.0, 'conditions': 'rainy'},
    'New York': {'temp': 12.0, 'conditions': 'windy'},
    'Sydney': {'temp': 25.0, 'conditions': 'sunny'},
}


def get_weather(city: str) -> WeatherResult:
    """Get weather for a city.

    Args:
        city: Name of the city to get weather for.

    Returns:
        Weather data including temperature and conditions.
    """
    data = _WEATHER_DATA.get(city, {'temp': 20.0, 'conditions': 'unknown'})
    return {'city': city, 'temp': data['temp'], 'unit': 'celsius', 'conditions': data['conditions']}
