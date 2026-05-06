from datetime import datetime

from brain.router import RequestClass
from memory.db import (
    get_recent_events,
    get_recent_sessions,
    get_patterns,
    get_all_preferences,
    get_upcoming_reminders,
)

# Quanti dati recenti includere nel contesto
_EVENTS_WINDOW = 12        # ultimi N eventi passivi (~2min a 10s/frame)
_SESSIONS_WINDOW = 5       # ultime N sessioni
_PATTERNS_WINDOW = 5       # top N pattern per confidenza


def build(request_class: RequestClass, user_text: str = "") -> str:
    """
    Assembla il contesto da SQLite da includere nel system prompt di GPT-4o.
    Il contenuto varia in base alla classe della richiesta.

    Class A → preferenze + pattern (nessun dato visivo)
    Class B → A + stato corrente dal monitoraggio passivo + sessioni recenti
    Class C → segnala che seguiranno immagini live; GPT non ha bisogno di dati passivi
    """
    sections: list[str] = []

    prefs = _format_preferences()
    if prefs:
        sections.append(prefs)

    reminders = _format_upcoming_reminders()
    if reminders:
        sections.append(reminders)

    if request_class == RequestClass.C:
        sections.append("[Visione] Frame live allegati alla richiesta.")
        return "\n\n".join(sections)

    patterns = _format_patterns()
    if patterns:
        sections.append(patterns)

    if request_class == RequestClass.B:
        current = _format_current_state()
        if current:
            sections.append(current)
        sessions = _format_recent_sessions()
        if sessions:
            sections.append(sessions)

    return "\n\n".join(sections) if sections else "(nessun contesto disponibile)"


# ---------------------------------------------------------------------------
# formatters
# ---------------------------------------------------------------------------

_INTERNAL_PREFS = {
    "aggregator_watermark",
    "question_pool",
    "learner_last_gen_at",
    "learner_last_suspend_at",
}


def _format_preferences() -> str:
    rows = [r for r in get_all_preferences() if r["category"] not in _INTERNAL_PREFS]
    if not rows:
        return ""
    lines = [f"  {r['category']}: {r['value']}" for r in rows]
    return "[Preferenze]\n" + "\n".join(lines)


def _format_patterns() -> str:
    rows = get_patterns()[:_PATTERNS_WINDOW]
    if not rows:
        return ""
    lines = []
    for p in rows:
        conf = f"{p['confidence']:.0%}" if p["confidence"] is not None else "?"
        span = ""
        if p["typical_start"] and p["typical_end"]:
            span = f" ({p['typical_start']}–{p['typical_end']})"
        lines.append(f"  {p['pattern_type']}{span} — conf {conf}")
    return "[Pattern abituali]\n" + "\n".join(lines)


def _format_current_state() -> str:
    events = get_recent_events(_EVENTS_WINDOW)
    if not events:
        return ""

    latest = events[0]
    ts = latest["timestamp"][:16].replace("T", " ")

    state_parts = []
    if latest["in_room"] is not None:
        state_parts.append("in stanza" if latest["in_room"] else "fuori stanza")
    if latest["at_desk"] is not None:
        state_parts.append("alla scrivania" if latest["at_desk"] else "non alla scrivania")
    if latest["at_pc"] is not None:
        state_parts.append("al pc" if latest["at_pc"] else "non al pc")
    if latest["activity_label"]:
        state_parts.append(latest["activity_label"])

    state_line = ", ".join(state_parts) if state_parts else "stato sconosciuto"

    # stima durata sessione corrente: cerca quanti eventi consecutivi
    # hanno lo stesso stato "in stanza" partendo dall'ultimo
    same_count = sum(
        1 for e in events
        if e["in_room"] == latest["in_room"] and e["at_desk"] == latest["at_desk"]
    )
    approx_min = round(same_count * 10 / 60, 1)

    lines = [
        f"  Ultimo aggiornamento: {ts}",
        f"  Stato: {state_line}",
        f"  Durata stimata sessione corrente: ~{approx_min} min",
    ]

    if latest["extra_questions"]:
        for q, a in latest["extra_questions"].items():
            lines.append(f"  {q}: {a}")

    avg_infer = _avg_inference_from_events(events)
    if avg_infer:
        lines.append(f"  Latenza media moondream: {avg_infer}ms")

    return "[Stato corrente (monitoraggio passivo)]\n" + "\n".join(lines)


def _format_recent_sessions() -> str:
    rows = get_recent_sessions(_SESSIONS_WINDOW)
    if not rows:
        return ""
    lines = []
    for s in rows:
        start = s["start"][:16].replace("T", " ")
        dur = f"{s['duration_min']:.0f}min" if s["duration_min"] else "in corso"
        act = s["activity"] or "—"
        lines.append(f"  {start}  {act}  {dur}")
    return "[Sessioni recenti]\n" + "\n".join(lines)


def _format_upcoming_reminders() -> str:
    rows = get_upcoming_reminders(limit=5)
    if not rows:
        return ""
    lines = []
    for r in rows:
        t = r["trigger_time"][:16].replace("T", " ")
        rep = f" [{r['repeat']}]" if r.get("repeat") and r["repeat"] != "none" else ""
        lines.append(f"  {t}{rep}  {r['text']}")
    return "[Promemoria in arrivo]\n" + "\n".join(lines)


def _avg_inference_from_events(events: list[dict]) -> int | None:
    times = [e["inference_time_ms"] for e in events if e["inference_time_ms"] is not None]
    if not times:
        return None
    return round(sum(times) / len(times))


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from memory.db import init_db, log_event, upsert_pattern, set_preference, start_session, end_session
    from brain.router import RequestClass

    init_db()

    # seed dati di test
    set_preference("lingua", "italiano", source="config")
    set_preference("nome_utente", "Federico", source="appreso")
    upsert_pattern("mattina_scrivania", "09:00", "13:00", frequency=10,
                   avg_duration_min=220, confidence=0.8)

    for i in range(6):
        log_event(in_room=True, at_desk=True, at_pc=(i % 2 == 0),
                  inference_time_ms=3200 + i * 100)

    sid = start_session("lavoro")
    end_session(sid, context_note="test build contesto")

    print("=== Class A ===")
    print(build(RequestClass.A))

    print("\n=== Class B ===")
    print(build(RequestClass.B))

    print("\n=== Class C ===")
    print(build(RequestClass.C))
