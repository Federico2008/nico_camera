import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def build_briefing_text() -> str:
    import memory.db as db
    import config
    from brain.weather import get_weather

    hour = datetime.now().hour
    if hour < 12:
        greeting = "Buongiorno"
    elif hour < 18:
        greeting = "Buon pomeriggio"
    else:
        greeting = "Buonasera"

    parts = [greeting]

    weather = get_weather()
    if weather:
        parts.append(f"Il meteo oggi a {config.WEATHER_CITY}: {weather}")

    reminders = db.get_upcoming_reminders(limit=20)
    today = datetime.now().date()
    today_rems = [r for r in reminders if datetime.fromisoformat(r["trigger_time"]).date() == today]
    if today_rems:
        rem_texts = [r["text"] for r in today_rems[:3]]
        parts.append(f"Hai {len(today_rems)} promemoria per oggi: {', '.join(rem_texts)}")
    else:
        parts.append("Nessun promemoria per oggi")

    notes = db.get_notes(limit=3)
    if notes:
        note_texts = [n["text"][:60] for n in notes]
        parts.append(f"Ultime note: {', '.join(note_texts)}")

    return ". ".join(parts) + "."
