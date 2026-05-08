import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from enum import Enum
from typing import Any

import config

_FUZZY_THRESHOLD = 0.78   # similarità minima per match fonetico (0-1)


_FIRST_WORD_THRESHOLD = 0.60  # primo word deve matchare almeno 60% — previene falsi positivi su suffix comuni (es. "ore sono" ≁ "come sono")


def _fuzzy_contains(text: str, keyword: str) -> bool:
    """True se keyword (o variante fonetica simile) compare in text."""
    kw_words = keyword.split()
    text_words = text.split()
    n = len(kw_words)
    kw_str = " ".join(kw_words)
    for i in range(len(text_words) - n + 1):
        window_words = text_words[i : i + n]
        if n > 1 and SequenceMatcher(None, kw_words[0], window_words[0]).ratio() < _FIRST_WORD_THRESHOLD:
            continue
        window = " ".join(window_words)
        if SequenceMatcher(None, kw_str, window).ratio() >= _FUZZY_THRESHOLD:
            return True
    return False


class RequestClass(str, Enum):
    A = "A"  # no camera — risponde con dati statici o memoria storica
    B = "B"  # usa dati dal monitoraggio passivo già accumulati
    C = "C"  # attiva tier2 "guardami" — cattura frame live


@dataclass(frozen=True)
class RoutingResult:
    cls: RequestClass
    reason: str


# Keyword aggiuntive per Class B (domande sullo stato tracciato passivamente)
# config.CLASS_C_KEYWORDS e CLASS_A_KEYWORDS sono già in config.py
_CLASS_B_KEYWORDS: list[str] = [
    "da quanto",
    "quanto tempo",
    "da quando",
    "sono alla scrivania",
    "sono al computer",
    "sono al pc",
    "ero alla scrivania",
    "quante volte",
    "mi sono alzato",
    "ho fatto pausa",
    "ho lavorato",
    "sessione",
    "routine",
    "di solito a quest'ora",
    "stamattina ho",
    "oggi ho passato",
    "quanto ho",
]


def classify(text: str) -> RoutingResult:
    """
    Classifica una richiesta vocale in A / B / C.

    Ordine di priorità:
      1. C — qualsiasi keyword visiva esplicita → attiva camera live
      2. B — keyword che referenziano dati del monitoraggio passivo
      3. A — tutto il resto (default)

    La classificazione è puramente keyword-based: veloce e deterministica,
    nessuna chiamata LLM in questo stadio.
    """
    low = text.lower().strip()

    # -- Class C: trigger visivi espliciti (ha la precedenza assoluta) --
    for kw in config.CLASS_C_KEYWORDS:
        if kw in low or _fuzzy_contains(low, kw):
            return RoutingResult(cls=RequestClass.C, reason=f"keyword visiva: '{kw}'")

    # -- Class B: domande sul monitoraggio passivo --
    for kw in _CLASS_B_KEYWORDS:
        if kw in low or _fuzzy_contains(low, kw):
            return RoutingResult(cls=RequestClass.B, reason=f"keyword passivo: '{kw}'")

    # -- Class A: default --
    return RoutingResult(cls=RequestClass.A, reason="nessun trigger visivo o passivo")


# ---------------------------------------------------------------------------
# intent detection (layer sopra classify — azioni specifiche)
# ---------------------------------------------------------------------------

class IntentType(str, Enum):
    REMINDER        = "reminder"         # "ricordami di X alle Y"
    ALARM           = "alarm"            # "imposta sveglia alle Y"
    ADD_NOTE        = "add_note"         # "segna nota: X" / "ricorda che X"
    SEARCH_NOTES    = "search_notes"     # "che idee avevo su X"
    SEARCH_SHOPPING = "search_shopping"  # "ricordami cosa dovevo comprare per X"
    WEEKLY_SUMMARY  = "weekly_summary"   # "riassumi la mia settimana"
    STUDY_PLAN      = "study_plan"       # "fammi piano studio per domani"
    NONE            = "none"


@dataclass
class IntentResult:
    intent: IntentType
    response: str | None = None        # risposta diretta senza GPT
    gpt_prompt: str | None = None      # prompt da passare a gpt.chat()
    gpt_context: str | None = None     # contesto dati formattato
    data: dict[str, Any] = field(default_factory=dict)


_DAYS_RE = r"(?:lunedì?|martedì?|mercoledì?|giovedì?|venerdì?|sabato|domenica)"
_TIME_EXPR_RE = (
    r"(domani\s+alle\s+\d{1,2}(?::\d{2})?"
    + r"|domani"
    + r"|" + _DAYS_RE + r"\s+alle\s+\d{1,2}(?::\d{2})?"
    + r"|" + _DAYS_RE
    + r"|(?:alle|per\s+le)\s+\d{1,2}(?::\d{2})?"
    + r"|tra\s+\d+\s+(?:ore?|minut[io]))"
)
_REMINDER_PATTERNS = [
    r"ricordami\s+di\s+(.+?)\s+" + _TIME_EXPR_RE,
    r"ricordami\s+" + _TIME_EXPR_RE + r"\s+di\s+(.+)",
]
_ALARM_PATTERNS = [
    r"(?:imposta|metti)\s+(?:una\s+)?sveglia\s+(?:alle|per le)\s+(\d{1,2})(?::(\d{2}))?",
    r"svegliami\s+(?:alle|per le)\s+(\d{1,2})(?::(\d{2}))?",
]
_ADD_NOTE_PATTERNS = [
    r"segna\s+(?:una\s+)?nota[:\s]+(.+)",
    r"ricorda\s+che\s+(.+)",
    r"annota\s+(?:che\s+)?(.+)",
    r"aggiungi\s+(?:una\s+)?nota[:\s]+(.+)",
    r"salva\s+(?:questa\s+)?idea[:\s]+(.+)",
]
_NOTES_SEARCH_PATTERNS = [
    r"che\s+idee\s+avevo\s+su\s+(.+)",
    r"cosa\s+avevo\s+pensato\s+(?:per|su|di)\s+(.+)",
    r"mostrami\s+(?:le\s+)?note\s+su\s+(.+)",
]
_SHOPPING_PATTERNS = [
    r"(?:ricordami\s+)?cosa\s+(?:dovevo|devo)\s+comprare\s+(?:per\s+(.+)|dalla\s+parte\s+acquisti)?",
    r"lista\s+(?:della\s+)?spesa",
    r"cosa\s+(?:c'è|ho)\s+nella\s+lista\s+(?:della\s+)?spesa",
]
_WEEKLY_PATTERNS = [
    r"riassumi\s+(?:la\s+mia\s+)?settimana",
    r"com'è\s+andata\s+(?:la\s+settimana|questa\s+settimana)",
    r"cosa\s+ho\s+fatto\s+(?:questa|la)\s+settimana",
    r"resoconto\s+(?:della\s+)?settimana",
]
_STUDY_PATTERNS = [
    r"(?:fammi|crea|prepara)\s+(?:un\s+)?piano\s+(?:di\s+)?studio\s+per\s+domani",
    r"piano\s+studio\s+(?:per\s+)?domani",
    r"come\s+studiare\s+domani",
]


def _parse_time(hour: str, minute: str | None, base: datetime | None = None) -> datetime:
    base = base or datetime.now()
    h = int(hour)
    m = int(minute) if minute else 0
    target = base.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= datetime.now():
        target += timedelta(days=1)
    return target


_DAY_NAMES_IT: dict[str, int] = {
    "lunedì": 0, "lunedi": 0,
    "martedì": 1, "martedi": 1,
    "mercoledì": 2, "mercoledi": 2,
    "giovedì": 3, "giovedi": 3,
    "venerdì": 4, "venerdi": 4,
    "sabato": 5,
    "domenica": 6,
}


def _next_weekday_dt(weekday_num: int) -> datetime:
    now = datetime.now()
    days_ahead = weekday_num - now.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return now + timedelta(days=days_ahead)


def _parse_reminder_time(expr: str) -> datetime:
    """Parse Italian time/date expression. Fallback: domani 9:00."""
    t = expr.strip().lower()
    now = datetime.now()

    m = re.match(r"tra\s+(\d+)\s+minut[io]", t)
    if m:
        return now + timedelta(minutes=int(m.group(1)))

    m = re.match(r"tra\s+(\d+)\s+or[ae]", t)
    if m:
        return now + timedelta(hours=int(m.group(1)))

    m = re.match(r"domani(?:\s+alle\s+(\d{1,2})(?::(\d{2}))?)?", t)
    if m:
        base = now + timedelta(days=1)
        if m.group(1):
            return _parse_time(m.group(1), m.group(2), base)
        return base.replace(hour=9, minute=0, second=0, microsecond=0)

    for day_name, day_num in _DAY_NAMES_IT.items():
        if t.startswith(day_name):
            rest = t[len(day_name):].strip()
            base = _next_weekday_dt(day_num)
            m2 = re.match(r"alle\s+(\d{1,2})(?::(\d{2}))?", rest)
            if m2:
                return _parse_time(m2.group(1), m2.group(2), base)
            return base.replace(hour=9, minute=0, second=0, microsecond=0)

    m = re.match(r"(?:alle|per\s+le)\s+(\d{1,2})(?::(\d{2}))?", t)
    if m:
        return _parse_time(m.group(1), m.group(2))

    return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)


def _format_sessions_summary(sessions: list[dict]) -> str:
    if not sessions:
        return "(nessuna sessione negli ultimi 7 giorni)"
    lines = []
    for s in sessions:
        start = s["start"][:16].replace("T", " ")
        dur = f"{s['duration_min']:.0f}min" if s.get("duration_min") else "in corso"
        act = s.get("activity") or "—"
        intr = s.get("interruptions") or 0
        lines.append(f"  {start}  attività:{act}  durata:{dur}  interruzioni:{intr}")
    return "\n".join(lines)


def _format_patterns_context(patterns: list[dict]) -> str:
    if not patterns:
        return "(nessun pattern di studio rilevato)"
    lines = []
    for p in patterns:
        span = ""
        if p.get("typical_start") and p.get("typical_end"):
            span = f" ({p['typical_start']}–{p['typical_end']})"
        lines.append(f"  {p['pattern_type']}{span} — freq:{p.get('frequency',0)}")
    return "\n".join(lines)


def detect_intent(text: str) -> IntentResult:
    """
    Rileva intent specifici nel testo vocale.
    Restituisce IntentResult con NONE se nessun intent corrisponde.
    Da chiamare prima di classify() nel main loop.
    """
    import memory.db as db

    low = text.lower().strip()

    # -- ADD NOTE --
    for pat in _ADD_NOTE_PATTERNS:
        m = re.search(pat, low)
        if m:
            note_text = m.group(1).strip().capitalize()
            note_id = db.add_note(text=note_text, category="voce")
            return IntentResult(
                intent=IntentType.ADD_NOTE,
                response=f"Nota salvata: {note_text}.",
                data={"note_id": note_id},
            )

    # -- ALARM --
    for pat in _ALARM_PATTERNS:
        m = re.search(pat, low)
        if m:
            groups = [g for g in m.groups() if g is not None]
            hour = groups[0]
            minute = groups[1] if len(groups) > 1 else None
            trigger = _parse_time(hour, minute)
            reminder_id = db.add_reminder(
                text="Sveglia!",
                trigger_time=trigger,
                category="sveglia",
            )
            t_str = trigger.strftime("%H:%M")
            return IntentResult(
                intent=IntentType.ALARM,
                response=f"Sveglia impostata per le {t_str}.",
                data={"reminder_id": reminder_id, "trigger_time": trigger.isoformat()},
            )

    # -- REMINDER --
    for i, pat in enumerate(_REMINDER_PATTERNS):
        m = re.search(pat, low)
        if m:
            if i == 0:
                reminder_text, time_expr = m.group(1).strip(), m.group(2).strip()
            else:
                time_expr, reminder_text = m.group(1).strip(), m.group(2).strip()
            trigger = _parse_reminder_time(time_expr)
            reminder_id = db.add_reminder(
                text=reminder_text.capitalize(),
                trigger_time=trigger,
                category="promemoria",
            )
            t_str = trigger.strftime("%d/%m alle %H:%M")
            return IntentResult(
                intent=IntentType.REMINDER,
                response=f"Promemoria impostato per {t_str}: {reminder_text}.",
                data={"reminder_id": reminder_id, "trigger_time": trigger.isoformat()},
            )

    # -- SEARCH NOTES --
    for pat in _NOTES_SEARCH_PATTERNS:
        m = re.search(pat, low)
        if m:
            keyword = m.group(1).strip()
            notes = db.get_notes(keyword=keyword, limit=5)
            if not notes:
                return IntentResult(
                    intent=IntentType.SEARCH_NOTES,
                    response=f"Non ho trovato note su '{keyword}'.",
                    data={"keyword": keyword},
                )
            lines = [f"- {n['text']}" for n in notes]
            summary = f"Ho trovato {len(notes)} nota{'e' if len(notes)>1 else ''} su '{keyword}':\n" + "\n".join(lines)
            return IntentResult(
                intent=IntentType.SEARCH_NOTES,
                response=summary,
                data={"keyword": keyword, "notes": notes},
            )

    # -- SEARCH SHOPPING --
    for pat in _SHOPPING_PATTERNS:
        m = re.search(pat, low)
        if m:
            context_arg = m.group(1).strip() if m.lastindex and m.group(1) else None
            notes = db.get_notes(category="acquisti", keyword=context_arg, limit=10)
            if not notes:
                return IntentResult(
                    intent=IntentType.SEARCH_SHOPPING,
                    response="Non ho trovato nulla nella lista acquisti.",
                )
            lines = [f"- {n['text']}" for n in notes]
            return IntentResult(
                intent=IntentType.SEARCH_SHOPPING,
                response="Lista acquisti:\n" + "\n".join(lines),
                data={"notes": notes},
            )

    # -- WEEKLY SUMMARY --
    for pat in _WEEKLY_PATTERNS:
        if re.search(pat, low):
            sessions = db.get_sessions_last_days(7)
            ctx = _format_sessions_summary(sessions)
            return IntentResult(
                intent=IntentType.WEEKLY_SUMMARY,
                gpt_prompt=(
                    "L'utente vuole un riassunto della sua settimana. "
                    "Basati sulle sessioni qui sotto e fai un riassunto parlato, "
                    "conciso e naturale. Evidenzia pattern, ore di lavoro, pause."
                ),
                gpt_context=f"[Sessioni ultimi 7 giorni]\n{ctx}",
                data={"sessions": sessions},
            )

    # -- STUDY PLAN --
    for pat in _STUDY_PATTERNS:
        if re.search(pat, low):
            from memory.db import get_patterns
            patterns = [p for p in get_patterns() if "stud" in (p.get("pattern_type") or "").lower()]
            sessions = db.get_sessions_last_days(14)
            study_sessions = [s for s in sessions if "stud" in (s.get("activity") or "").lower()]
            ctx_parts = []
            if patterns:
                ctx_parts.append("[Pattern studio]\n" + _format_patterns_context(patterns))
            if study_sessions:
                ctx_parts.append("[Sessioni studio recenti]\n" + _format_sessions_summary(study_sessions))
            ctx = "\n\n".join(ctx_parts) if ctx_parts else "(nessun dato di studio disponibile)"
            return IntentResult(
                intent=IntentType.STUDY_PLAN,
                gpt_prompt=(
                    "L'utente vuole un piano di studio per domani. "
                    "Basati sui suoi pattern e sessioni precedenti. "
                    "Proponi orari concreti, pause, obiettivi. Risposta parlata, senza elenchi o markdown."
                ),
                gpt_context=ctx,
                data={"patterns": patterns, "study_sessions": study_sessions},
            )

    return IntentResult(intent=IntentType.NONE)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        # (testo, classe attesa, descrizione)
        ("che ore sono",                    "A", "ora corrente"),
        ("metti un timer di 5 minuti",      "A", "timer"),
        ("com'è il meteo oggi",             "A", "meteo"),
        ("quanto fa 15 per 7",              "A", "calcolo"),
        ("dimmi una barzelletta",           "A", "domanda generica"),
        ("da quanto sono alla scrivania",   "B", "durata sessione scrivania"),
        ("quanto ho lavorato stamattina",   "B", "ore lavoro"),
        ("quante volte mi sono alzato",     "B", "conteggio movimenti"),
        ("di solito a quest'ora cosa faccio","B","routine"),
        ("guardami",                        "C", "trigger visivo diretto"),
        ("cosa vedi adesso",                "C", "trigger visivo"),
        ("descrivi cosa sto facendo",       "C", "descrizione attività visiva"),
        ("cosa c'è sulla mia scrivania",    "C", "oggetti visivi"),
        ("come sono vestito",               "C", "aspetto fisico"),
    ]

    print(f"{'Testo':<40} {'Atteso'} {'Ottenuto'} {'OK'}")
    print("-" * 65)
    all_ok = True
    for text, expected_cls, desc in cases:
        result = classify(text)
        ok = result.cls.value == expected_cls
        if not ok:
            all_ok = False
        flag = "OK" if ok else f"FAIL (reason: {result.reason})"
        print(f"{text:<40} {expected_cls}      {result.cls.value}        {flag}")

    print()
    print("Tutti OK" if all_ok else "ERRORI presenti — rivedi le keyword")
