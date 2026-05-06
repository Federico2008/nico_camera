"""
Aggrega eventi grezzi in sessioni.

Viene eseguito ogni ora in un thread daemon.
Legge eventi con id > watermark (preference "aggregator_watermark"),
raggruppa sequenze continue di in_room=True in sessioni,
scrive sessioni complete nella tabella sessions.
"""

import logging
import threading
import time
from collections import Counter
from datetime import datetime

import config
from memory.db import (
    get_preference, set_preference,
    end_session, start_session,
)

logger = logging.getLogger(__name__)

_WATERMARK_KEY = "aggregator_watermark"   # ultimo event.id processato


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------

def run_once() -> int:
    """
    Esegue un ciclo di aggregazione.
    Restituisce il numero di sessioni scritte.
    """
    import sqlite3
    import config as _cfg

    watermark = int(get_preference(_WATERMARK_KEY, default="0"))

    with sqlite3.connect(_cfg.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT id, timestamp, in_room, at_desk, at_pc, activity_label
               FROM events
               WHERE id > ?
               ORDER BY timestamp ASC""",
            (watermark,),
        ).fetchall()

    if not rows:
        return 0

    events = [dict(r) for r in rows]
    sessions = _group_into_sessions(events)

    written = 0
    for s in sessions:
        sid = start_session(s["activity"])
        end_session(sid, context_note=s["context_note"])
        # Override start/end timestamps since start_session uses now()
        _rewrite_session_times(sid, s["start"], s["end"], s["duration_min"])
        written += 1
        logger.debug("Sessione aggregata: %s (%s, %.0f min)", s["activity"], s["start"][:16], s["duration_min"])

    max_id = events[-1]["id"]
    set_preference(_WATERMARK_KEY, str(max_id), source="aggregator")
    logger.info("Aggregator: %d sessioni scritte (eventi %d→%d).", written, watermark + 1, max_id)
    return written


def start_background() -> threading.Thread:
    """Avvia il job di aggregazione oraria in un thread daemon."""
    def _loop():
        while True:
            try:
                run_once()
            except Exception as exc:
                logger.error("Aggregator errore: %s", exc)
            time.sleep(config.AGGREGATOR_RUN_EVERY_S)

    t = threading.Thread(target=_loop, name="aggregator", daemon=True)
    t.start()
    logger.info("Aggregator avviato (ogni %ds).", config.AGGREGATOR_RUN_EVERY_S)
    return t


# ---------------------------------------------------------------------------
# session detection
# ---------------------------------------------------------------------------

def _group_into_sessions(events: list[dict]) -> list[dict]:
    """
    Raggruppa eventi in sessioni basandosi sulla continuità di in_room=True.

    Regole:
    - Un gap > AGGREGATOR_GAP_S tra due eventi in_room=True chiude la sessione.
    - Sessioni con durata < AGGREGATOR_MIN_DURATION_S vengono scartate (rumore).
    - L'ultima sessione non viene chiusa se l'ultimo evento è in_room=True
      (potrebbe essere ancora in corso — aggreghiamo solo sessioni "complete").
    """
    sessions: list[dict] = []
    current: list[dict] = []
    last_ts: datetime | None = None

    for ev in events:
        ts = datetime.fromisoformat(ev["timestamp"])
        in_room = bool(ev["in_room"]) if ev["in_room"] is not None else False

        if in_room:
            if last_ts is not None and (ts - last_ts).total_seconds() > config.AGGREGATOR_GAP_S:
                _flush(current, sessions)
                current = []
            current.append(ev)
            last_ts = ts
        else:
            if current and last_ts:
                gap = (ts - last_ts).total_seconds()
                if gap > config.AGGREGATOR_GAP_S:
                    _flush(current, sessions)
                    current = []
                    last_ts = None

    # non chiudiamo l'ultima sessione se ancora aperta
    return sessions


def _flush(events: list[dict], sessions: list[dict]) -> None:
    if not events:
        return
    start = datetime.fromisoformat(events[0]["timestamp"])
    end   = datetime.fromisoformat(events[-1]["timestamp"])
    dur   = (end - start).total_seconds()

    if dur < config.AGGREGATOR_MIN_DURATION_S:
        return

    labels = [e["activity_label"] for e in events if e.get("activity_label")]
    activity = Counter(labels).most_common(1)[0][0] if labels else "sconosciuta"

    sessions.append({
        "start":        events[0]["timestamp"],
        "end":          events[-1]["timestamp"],
        "activity":     activity,
        "duration_min": round(dur / 60, 1),
        "context_note": f"aggregated from {len(events)} events",
    })


def _rewrite_session_times(
    session_id: int, start: str, end: str, duration_min: float
) -> None:
    import sqlite3
    import config as _cfg
    with sqlite3.connect(_cfg.DB_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET start=?, end=?, duration_min=? WHERE id=?",
            (start, end, duration_min, session_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import timedelta
    from memory.db import init_db, get_recent_sessions, log_event

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    init_db()

    # ------------------------------------------------------------------
    # 1. unit test _group_into_sessions
    # ------------------------------------------------------------------
    print("=== 1. rilevamento sessioni ===")

    def _make_events(sequence):
        """sequence: lista di (in_room, seconds_from_start)"""
        base = datetime(2026, 5, 5, 9, 0, 0)
        return [
            {
                "id": i + 1,
                "timestamp": (base + timedelta(seconds=t)).isoformat(),
                "in_room": 1 if in_room else 0,
                "at_desk": 1,
                "at_pc": 1,
                "activity_label": "lavoro_pc" if in_room else None,
            }
            for i, (in_room, t) in enumerate(sequence)
        ]

    # Scenario A: una sessione di 20 minuti.
    # Il gap tra l'ultimo True (1190s) e il primo False deve essere > GAP_THRESHOLD (300s).
    seq_a = [(True, i * 10) for i in range(120)]         # 20min: 0-1190s
    seq_a += [(False, 1510 + i * 10) for i in range(5)]  # gap=320s > 300s → chiude
    sessions_a = _group_into_sessions(_make_events(seq_a))
    print(f"  Scenario A (20min continui): {len(sessions_a)} sessione(i)  "
          f"{'OK' if len(sessions_a) == 1 else 'FAIL'}")
    if sessions_a:
        print(f"    activity={sessions_a[0]['activity']}  dur={sessions_a[0]['duration_min']}min")

    # Scenario B: due sessioni separate.
    # Sessione 1 finisce a 590s, gap di 320s, sessione 2 inizia a 1200s.
    # Sessione 2 finisce a 1790s, gap di 320s → chiude anche la seconda.
    seq_b  = [(True, i * 10) for i in range(60)]           # sess.1: 0-590s
    seq_b += [(False, 910 + i * 10) for i in range(5)]     # gap=320s → chiude sess.1
    seq_b += [(True, 1200 + i * 10) for i in range(60)]    # sess.2: 1200-1790s
    seq_b += [(False, 2110 + i * 10) for i in range(3)]    # gap=320s → chiude sess.2
    sessions_b = _group_into_sessions(_make_events(seq_b))
    print(f"  Scenario B (2 sessioni distinte): {len(sessions_b)} sessione(i)  "
          f"{'OK' if len(sessions_b) == 2 else 'FAIL'}")

    # Scenario C: sessione troppo corta (< 2min) — deve essere scartata
    seq_c  = [(True, i * 10) for i in range(5)]     # solo 40 secondi
    seq_c += [(False, 50 + i * 10) for i in range(3)]
    sessions_c = _group_into_sessions(_make_events(seq_c))
    print(f"  Scenario C (40s, troppo corta): {len(sessions_c)} sessione(i)  "
          f"{'OK' if len(sessions_c) == 0 else 'FAIL'}")

    # Scenario D: sessione ancora aperta — non deve essere chiusa
    seq_d = [(True, i * 10) for i in range(60)]   # 10min, nessun evento False finale
    sessions_d = _group_into_sessions(_make_events(seq_d))
    print(f"  Scenario D (ancora aperta): {len(sessions_d)} sessioni chiuse  "
          f"{'OK' if len(sessions_d) == 0 else 'FAIL'}")

    # ------------------------------------------------------------------
    # 2. integrazione run_once con DB
    # ------------------------------------------------------------------
    print("\n=== 2. run_once integrazione DB ===")

    # reset watermark
    from memory.db import set_preference
    set_preference(_WATERMARK_KEY, "0")

    base = datetime(2026, 5, 5, 10, 0, 0)
    for i in range(90):          # 15 minuti in_room
        log_event(True, True, True, "lavoro_pc", 0.95,
                  inference_time_ms=3000)
    for i in range(5):           # fine sessione
        log_event(False, None, None, None, None)

    n_sessions_before = len(get_recent_sessions(50))
    written = run_once()
    n_sessions_after = len(get_recent_sessions(50))

    print(f"  Sessioni scritte: {written}")
    print(f"  Sessioni in DB prima: {n_sessions_before}  dopo: {n_sessions_after}")
    assert written >= 1, "Almeno 1 sessione attesa"
    assert n_sessions_after > n_sessions_before

    sessions = get_recent_sessions(3)
    s = sessions[0]
    print(f"  Ultima sessione: activity={s['activity']}  dur={s['duration_min']}min")
    print("  OK")

    print("\n=== Tutti i test superati ===")
