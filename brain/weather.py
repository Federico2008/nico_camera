import logging
import requests
import config

logger = logging.getLogger(__name__)


def get_weather(city: str | None = None) -> str | None:
    city = city or config.WEATHER_CITY
    if not config.OPENWEATHER_API_KEY or not city:
        return None
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": config.OPENWEATHER_API_KEY,
                "units": "metric",
                "lang": "it",
            },
            timeout=5,
        )
        resp.raise_for_status()
        d = resp.json()
        temp     = round(d["main"]["temp"])
        feels    = round(d["main"]["feels_like"])
        desc     = d["weather"][0]["description"]
        humidity = d["main"]["humidity"]
        return f"{desc.capitalize()}, {temp}°C (percepiti {feels}°C), umidità {humidity}%"
    except Exception as exc:
        logger.warning("Meteo non disponibile: %s", exc)
        return None
