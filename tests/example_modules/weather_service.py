from __future__ import annotations as _annotations

from datetime import date
from typing import Any


class WeatherService:
    def get_historic_weather(self, location: str, forecast_date: date) -> str:
        return 'Sunny with a chance of rain'

    def get_forecast(self, location: str, forecast_date: date) -> str:
        return 'Rainy with a chance of sun'

    async def __aenter__(self) -> WeatherService:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
