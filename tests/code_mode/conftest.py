"""Shared fixtures and test tools for code mode tests."""

from __future__ import annotations

from typing import Any, TypedDict


# Define return type as TypedDict for better type hints in signatures
# Note: We return a plain dict at runtime for Monty compatibility
class WeatherResult(TypedDict):
    """Weather data for a city."""

    city: str
    temperature: float
    unit: str
    conditions: str


# Simulated weather data for test cities
_WEATHER_DATA: dict[str, dict[str, Any]] = {
    'London': {'temperature': 15.0, 'conditions': 'cloudy'},
    'Paris': {'temperature': 18.0, 'conditions': 'sunny'},
    'Tokyo': {'temperature': 22.0, 'conditions': 'rainy'},
    'New York': {'temperature': 12.0, 'conditions': 'windy'},
    'Sydney': {'temperature': 25.0, 'conditions': 'sunny'},
}


def get_weather(city: str) -> WeatherResult:
    """Get weather for a city.

    Args:
        city: Name of the city to get weather for.

    Returns:
        Weather data including temperature and conditions.
    """
    data = _WEATHER_DATA.get(city, {'temperature': 20.0, 'conditions': 'unknown'})
    return {'city': city, 'temperature': data['temperature'], 'unit': 'celsius', 'conditions': data['conditions']}
