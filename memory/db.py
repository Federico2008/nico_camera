import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime

import config


@contextmanager
def _conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT    NOT NULL,
                in_room           INTEGER,
                at_desk           INTEGER,
                at_pc             INTEGER,
                activity_label    TEXT,
                confidence        REAL,
                extra_questions   TEXT,         -- JSON {"domanda": "risposta"}
                inference_time_ms INTEGER       -- latenza moondream (performance monitoring)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                start         TEXT NOT NULL,
                end           TEXT,
                activity      TEXT,
                duration_min  REAL,
                interruptions INTEGER DEFAULT 0,
                context_note  TEXT
            );

            CREATE TABLE IF NOT EXISTS patterns (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type     TEXT NOT NULL UNIQUE,
                typical_start    TEXT,
                typical_end      TEXT,
                frequency        INTEGER DEFAULT 0,
                avg_duration_min REAL,
                confidence       REAL,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS preferences (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                category   TEXT NOT NULL UNIQUE,
                value      TEXT NOT NULL,
                learned_on TEXT,
                source     TEXT
            );

            CREATE TABLE IF NOT EXISTS reminders (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                text         TEXT NOT NULL,
                trigger_time DATETIME NOT NULL,
                repeat       TEXT DEFAULT 'none',
                done         BOOLEAN DEFAULT 0,
                spoken       BOOLEAN DEFAULT 0,
                category     TEXT,
                tags         TEXT
            );

            CREATE TABLE IF NOT EXISTS notes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                text       TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                category   TEXT,
                tags       TEXT
            );
        """)


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

def log_event(
    in_room: bool | None,
    at_desk: bool | None,
    at_pc: bool | None,
    activity_label: str | None = None,
    confidence: float | None = None,
    extra_questions: dict | None = None,
    inference_time_ms: int | None = None,
) -> int:
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO events
               (timestamp, in_room, at_desk, at_pc, activity_label, confidence,
                extra_questions, inference_time_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                None if in_room is None else int(in_room),
                None if at_desk is None else int(at_desk),
                None if at_pc is None else int(at_pc),
                activity_label,
                confidence,
                json.dumps(extra_questions, ensure_ascii=False) if extra_questions else None,
                inference_time_ms,
            ),
        )
        return cur.lastrowid


def get_recent_events(n: int = 50) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
    result = []
    for r in rows:
        row = dict(r)
        if row["extra_questions"]:
            row["extra_questions"] = json.loads(row["extra_questions"])
        result.append(row)
    return result


def count_events() -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]


def avg_inference_time_ms(last_n: int = 100) -> float | None:
    with _conn() as conn:
        row = conn.execute(
            """SELECT AVG(inference_time_ms) FROM (
                   SELECT inference_time_ms FROM events
                   WHERE inference_time_ms IS NOT NULL
                   ORDER BY timestamp DESC LIMIT ?
               )""",
            (last_n,),
        ).fetchone()
    return row[0]


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

def start_session(activity: str | None = None) -> int:
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (start, activity) VALUES (?, ?)",
            (datetime.now().isoformat(), activity),
        )
        return cur.lastrowid


def end_session(
    session_id: int,
    interruptions: int = 0,
    context_note: str | None = None,
) -> None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT start FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return
        start = datetime.fromisoformat(row["start"])
        end = datetime.now()
        duration = (end - start).total_seconds() / 60
        conn.execute(
            """UPDATE sessions
               SET end = ?, duration_min = ?, interruptions = ?, context_note = ?
               WHERE id = ?""",
            (end.isoformat(), round(duration, 2), interruptions, context_note, session_id),
        )


def get_recent_sessions(n: int = 20) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY start DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------

def upsert_pattern(
    pattern_type: str,
    typical_start: str | None = None,
    typical_end: str | None = None,
    frequency: int = 0,
    avg_duration_min: float | None = None,
    confidence: float | None = None,
) -> None:
    now = datetime.now().isoformat()
    with _conn() as conn:
        conn.execute(
            """INSERT INTO patterns
               (pattern_type, typical_start, typical_end, frequency,
                avg_duration_min, confidence, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(pattern_type) DO UPDATE SET
                   typical_start    = excluded.typical_start,
                   typical_end      = excluded.typical_end,
                   frequency        = excluded.frequency,
                   avg_duration_min = excluded.avg_duration_min,
                   confidence       = excluded.confidence,
                   updated_at       = excluded.updated_at""",
            (pattern_type, typical_start, typical_end, frequency,
             avg_duration_min, confidence, now, now),
        )


def get_patterns() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM patterns ORDER BY confidence DESC NULLS LAST"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# preferences
# ---------------------------------------------------------------------------

def set_preference(category: str, value: str, source: str | None = None) -> None:
    now = datetime.now().isoformat()
    with _conn() as conn:
        conn.execute(
            """INSERT INTO preferences (category, value, learned_on, source)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(category) DO UPDATE SET
                   value      = excluded.value,
                   learned_on = excluded.learned_on,
                   source     = excluded.source""",
            (category, str(value), now, source),
        )


def get_preference(category: str, default: str | None = None) -> str | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT value FROM preferences WHERE category = ?", (category,)
        ).fetchone()
    return row["value"] if row else default


def get_all_preferences() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM preferences").fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# reminders
# ---------------------------------------------------------------------------

def add_reminder(
    text: str,
    trigger_time: datetime,
    repeat: str = "none",
    category: str | None = None,
    tags: str | None = None,
) -> int:
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO reminders (text, trigger_time, repeat, category, tags)
               VALUES (?, ?, ?, ?, ?)""",
            (text, trigger_time.isoformat(), repeat, category, tags),
        )
        return cur.lastrowid


def get_due_reminders() -> list[dict]:
    now = datetime.now().isoformat()
    with _conn() as conn:
        rows = conn.execute(
            """SELECT * FROM reminders
               WHERE trigger_time <= ? AND spoken = 0 AND done = 0
               ORDER BY trigger_time""",
            (now,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_reminder_spoken(reminder_id: int) -> None:
    with _conn() as conn:
        conn.execute("UPDATE reminders SET spoken = 1 WHERE id = ?", (reminder_id,))


def mark_reminder_done(reminder_id: int) -> None:
    with _conn() as conn:
        conn.execute("UPDATE reminders SET done = 1 WHERE id = ?", (reminder_id,))


def get_upcoming_reminders(limit: int = 10) -> list[dict]:
    now = datetime.now().isoformat()
    with _conn() as conn:
        rows = conn.execute(
            """SELECT * FROM reminders
               WHERE trigger_time > ? AND done = 0
               ORDER BY trigger_time LIMIT ?""",
            (now, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------

def add_note(
    text: str,
    category: str | None = None,
    tags: str | None = None,
) -> int:
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO notes (text, created_at, category, tags) VALUES (?, ?, ?, ?)",
            (text, datetime.now().isoformat(), category, tags),
        )
        return cur.lastrowid


def get_notes(
    category: str | None = None,
    tag: str | None = None,
    keyword: str | None = None,
    limit: int = 20,
) -> list[dict]:
    clauses: list[str] = []
    params: list = []
    if category:
        clauses.append("category LIKE ?")
        params.append(f"%{category}%")
    if tag:
        clauses.append("tags LIKE ?")
        params.append(f"%{tag}%")
    if keyword:
        clauses.append("(text LIKE ? OR tags LIKE ? OR category LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM notes {where} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# sessions (extended query)
# ---------------------------------------------------------------------------

def get_sessions_last_days(n_days: int = 7) -> list[dict]:
    cutoff = datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    from datetime import timedelta
    cutoff -= timedelta(days=n_days - 1)
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE start >= ? ORDER BY start DESC",
            (cutoff.isoformat(),),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    print("DB inizializzato.")

    event_id = log_event(
        in_room=True, at_desk=True, at_pc=True,
        activity_label="scrivania", confidence=0.9,
        extra_questions={"Sta bevendo?": "no"},
        inference_time_ms=3200,
    )
    print(f"Evento scritto: id={event_id}")

    events = get_recent_events(5)
    print(f"Ultimi eventi ({len(events)}):")
    for e in events:
        print(f"  [{e['timestamp']}] in_room={e['in_room']} at_desk={e['at_desk']} "
              f"at_pc={e['at_pc']} infer={e['inference_time_ms']}ms")

    sid = start_session("lavoro")
    end_session(sid, context_note="test")
    sessions = get_recent_sessions(3)
    print(f"\nSessioni ({len(sessions)}):")
    for s in sessions:
        print(f"  [{s['start']}] {s['activity']} — {s['duration_min']} min")

    upsert_pattern("mattina_scrivania", typical_start="09:00", typical_end="13:00",
                   frequency=5, avg_duration_min=240, confidence=0.7)
    patterns = get_patterns()
    print(f"\nPattern ({len(patterns)}):")
    for p in patterns:
        print(f"  {p['pattern_type']} conf={p['confidence']}")

    set_preference("lingua", "italiano", source="config")
    val = get_preference("lingua")
    print(f"\nPreferenza lingua: {val}")

    avg = avg_inference_time_ms()
    print(f"\nMedia inferenza: {avg:.0f}ms" if avg else "\nNessun dato inferenza")
