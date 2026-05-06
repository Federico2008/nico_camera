import logging
import threading
import time
from datetime import datetime, timedelta

import audio.tts as tts
import memory.db as db

logger = logging.getLogger(__name__)

_CHECK_INTERVAL_S = 60

_REPEAT_DELTAS = {
    "daily":  timedelta(days=1),
    "weekly": timedelta(weeks=1),
}


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


_STALE_THRESHOLD_S = 3600  # reminder più vecchi di 1h → scartati silenziosamente


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
        tts.speak(r["text"])
        db.mark_reminder_spoken(r["id"])
        _reschedule(r)


def _loop() -> None:
    while True:
        try:
            _check_reminders()
        except Exception:
            logger.exception("Errore nel reminder scheduler")
        time.sleep(_CHECK_INTERVAL_S)


def start() -> None:
    t = threading.Thread(target=_loop, daemon=True, name="reminder-scheduler")
    t.start()
    logger.info("Reminder scheduler avviato (intervallo %ds)", _CHECK_INTERVAL_S)
