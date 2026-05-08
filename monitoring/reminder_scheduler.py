import logging
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta

import audio.tts as tts
import config
import memory.db as db

logger = logging.getLogger(__name__)

_CHECK_INTERVAL_S = 30

_REPEAT_DELTAS = {
    "daily":  timedelta(days=1),
    "weekly": timedelta(weeks=1),
}

_recently_fired: deque = deque(maxlen=30)
_fired_lock = threading.Lock()
_briefing_done_for: date | None = None


def get_recently_fired(since_s: int = 120) -> list[dict]:
    cutoff = time.time() - since_s
    with _fired_lock:
        return [dict(r) for r in _recently_fired if r.get("fired_at", 0) >= cutoff]


def _reschedule(r: dict) -> None:
    repeat = r.get("repeat", "none")
    delta = _REPEAT_DELTAS.get(repeat)
    if delta is None:
        db.mark_reminder_done(r["id"])
        return
    next_trigger = datetime.fromisoformat(r["trigger_time"]) + delta
    db.add_reminder(
        text=r["text"],
        trigger_time=next_trigger,
        repeat=repeat,
        category=r.get("category"),
        tags=r.get("tags"),
    )
    db.mark_reminder_done(r["id"])
    logger.info("Reminder riprogrammato (%s): '%s' → %s", repeat, r["text"], next_trigger.strftime("%Y-%m-%d %H:%M"))


_STALE_THRESHOLD_S = 3600
_ALARM_REPEATS     = 4
_ALARM_PAUSE_S     = 2.5


def _check_reminders() -> None:
    now = datetime.now()
    due = db.get_due_reminders()
    for r in due:
        age_s = (now - datetime.fromisoformat(r["trigger_time"])).total_seconds()
        if age_s > _STALE_THRESHOLD_S:
            logger.info("Reminder stantio ignorato (%.0fh): id=%d '%s'", age_s / 3600, r["id"], r["text"])
            _reschedule(r)
            continue

        logger.info("Reminder scattato: id=%d '%s'", r["id"], r["text"])

        with _fired_lock:
            _recently_fired.append({**r, "fired_at": time.time()})

        if r.get("category") == "sveglia":
            for i in range(_ALARM_REPEATS):
                tts.speak(r["text"])
                if i < _ALARM_REPEATS - 1:
                    time.sleep(_ALARM_PAUSE_S)
        else:
            tts.speak(r["text"])

        db.mark_reminder_spoken(r["id"])
        _reschedule(r)


def _maybe_morning_briefing() -> None:
    global _briefing_done_for
    now = datetime.now()
    if now.hour < config.BRIEFING_HOUR:
        return
    today = now.date()
    if _briefing_done_for == today:
        return
    _briefing_done_for = today
    try:
        from brain.briefing import build_briefing_text
        tts.speak(build_briefing_text())
        logger.info("Briefing mattutino parlato.")
    except Exception:
        logger.exception("Errore nel briefing mattutino")


def _loop() -> None:
    while True:
        try:
            _check_reminders()
            _maybe_morning_briefing()
        except Exception:
            logger.exception("Errore nel reminder scheduler")
        time.sleep(_CHECK_INTERVAL_S)


def start() -> None:
    t = threading.Thread(target=_loop, daemon=True, name="reminder-scheduler")
    t.start()
    logger.info("Reminder scheduler avviato (intervallo %ds)", _CHECK_INTERVAL_S)
