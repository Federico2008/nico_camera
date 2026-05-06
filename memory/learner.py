"""
Evoluzione adattiva del pool di domande per il monitoraggio passivo.

Lifecycle di una domanda:
  fissa (config) → sempre attiva, non modificabile
  candidata      → generata da GPT, validata su QUESTIONS_VALIDATE_ON osservazioni
  attiva         → promossa se le risposte sono variate (non monotone)
  sospesa        → risposta sempre uguale per QUESTIONS_SUSPEND_STATIC cicli consecutivi
                   → ricontrollata dopo QUESTIONS_SUSPEND_STATIC ulteriori osservazioni

Il pool è serializzato come JSON nel preference "question_pool".
"""

import json
import logging
from datetime import datetime

import config
from memory.db import (
    count_events, get_preference, set_preference, get_recent_events,
)

logger = logging.getLogger(__name__)

_POOL_KEY             = "question_pool"
_LAST_GEN_KEY         = "learner_last_gen_at"
_LAST_SUSPEND_KEY     = "learner_last_suspend_at"


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------

def get_active_questions() -> list[str]:
    """
    Domande da inviare a moondream: fisse + quelle custom validate.
    Chiamato da passive_loop._active_questions().
    """
    pool = _load_pool()
    return list(config.PASSIVE_QUESTIONS_FIXED) + pool.get("active", [])


def maybe_evolve(recent_answers: dict[str, str] | None = None) -> None:
    """
    Chiama dopo ogni ciclo passivo.
    Gestisce (in ordine):
      1. Aggiorna contatori candidati con le risposte di questo ciclo
      2. Promuove i candidati validati → attivi
      3. Sospende domande attive con risposta sempre uguale
      4. Risveglia domande sospese dopo QUESTIONS_SUSPEND_STATIC cicli
      5. Genera nuovi candidati ogni QUESTIONS_EVOLVE_AFTER eventi (via GPT)
    """
    pool = _load_pool()
    n    = count_events()

    if recent_answers:
        _update_candidates(pool, recent_answers)

    _promote_candidates(pool)
    _suspend_static(pool, n)
    _revive_suspended(pool, n)

    _save_pool(pool)

    # genera nuovi candidati solo se siamo a un multiplo esatto
    last_gen = int(get_preference(_LAST_GEN_KEY, "0"))
    if n >= last_gen + config.QUESTIONS_EVOLVE_AFTER and n > 0:
        _generate_and_add_candidates(pool)
        set_preference(_LAST_GEN_KEY, str(n), source="learner")


def pool_status() -> dict:
    """Restituisce lo stato corrente del pool (per debug e context_builder)."""
    pool = _load_pool()
    return {
        "fixed":      list(config.PASSIVE_QUESTIONS_FIXED),
        "active":     pool.get("active", []),
        "candidates": [c["text"] for c in pool.get("candidates", [])],
        "suspended":  [s["text"] for s in pool.get("suspended", [])],
    }


# ---------------------------------------------------------------------------
# pool persistence
# ---------------------------------------------------------------------------

def _load_pool() -> dict:
    raw = get_preference(_POOL_KEY)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {"active": [], "candidates": [], "suspended": []}


def _save_pool(pool: dict) -> None:
    set_preference(_POOL_KEY, json.dumps(pool, ensure_ascii=False), source="learner")


# ---------------------------------------------------------------------------
# pool operations
# ---------------------------------------------------------------------------

def _update_candidates(pool: dict, answers: dict[str, str]) -> None:
    for candidate in pool.get("candidates", []):
        q = candidate["text"]
        if q in answers:
            candidate.setdefault("responses", []).append(answers[q])
            candidate["count"] = candidate.get("count", 0) + 1


def _promote_candidates(pool: dict) -> None:
    remaining = []
    for candidate in pool.get("candidates", []):
        if candidate.get("count", 0) >= config.QUESTIONS_VALIDATE_ON:
            responses = candidate.get("responses", [])
            unique    = set(responses)
            if len(unique) > 1:
                pool.setdefault("active", []).append(candidate["text"])
                logger.info("Domanda promossa → attiva: '%s'", candidate["text"])
            else:
                logger.info("Domanda scartata (risposta monotona '%s'): '%s'",
                            unique, candidate["text"])
        else:
            remaining.append(candidate)
    pool["candidates"] = remaining


def _suspend_static(pool: dict, current_count: int) -> None:
    """Sospende domande attive custom la cui risposta è sempre uguale negli ultimi N eventi."""
    active    = pool.get("active", [])
    suspended = pool.get("suspended", [])
    still_active = []

    events = get_recent_events(config.QUESTIONS_SUSPEND_STATIC)
    for question in active:
        answers_for_q = [
            e["extra_questions"][question]
            for e in events
            if e.get("extra_questions") and question in (e.get("extra_questions") or {})
        ]
        if len(answers_for_q) >= config.QUESTIONS_SUSPEND_STATIC:
            unique = set(answers_for_q)
            if len(unique) == 1:
                suspended.append({
                    "text":        question,
                    "suspended_at": datetime.now().isoformat(),
                    "revive_after": current_count + config.QUESTIONS_SUSPEND_STATIC,
                })
                logger.info("Domanda sospesa (sempre '%s'): '%s'",
                            next(iter(unique)), question)
                continue
        still_active.append(question)

    pool["active"]    = still_active
    pool["suspended"] = suspended


def _revive_suspended(pool: dict, current_count: int) -> None:
    """Riporta in candidatura le domande sospese dopo QUESTIONS_SUSPEND_STATIC cicli."""
    still_suspended = []
    for item in pool.get("suspended", []):
        if current_count >= item.get("revive_after", 0):
            pool.setdefault("candidates", []).append({
                "text":      item["text"],
                "count":     0,
                "responses": [],
            })
            logger.info("Domanda risvegliata → candidata: '%s'", item["text"])
        else:
            still_suspended.append(item)
    pool["suspended"] = still_suspended


def _generate_and_add_candidates(pool: dict) -> None:
    """Chiama GPT per generare nuove domande candidate basate sugli ultimi N eventi."""
    try:
        from brain.gpt import chat
    except ImportError:
        logger.warning("brain.gpt non disponibile — generazione candidati saltata.")
        return

    events  = get_recent_events(config.QUESTIONS_EVOLVE_AFTER)
    summary = _summarise_events(events)
    existing = get_active_questions() + [c["text"] for c in pool.get("candidates", [])]

    prompt = (
        "Analizza queste osservazioni di monitoraggio passivo e suggerisci 2 nuove "
        "domande sì/no in inglese da fare a un modello di visione per capire meglio "
        "le routine dell'utente. Le domande devono essere visibili dalla camera, "
        "non duplicare quelle esistenti e avere risposta yes/no.\n\n"
        f"Domande esistenti: {', '.join(existing)}\n\n"
        f"Osservazioni recenti:\n{summary}\n\n"
        "Rispondi SOLO con le domande, una per riga, senza numerazione né spiegazioni."
    )

    try:
        raw = chat(prompt)
    except Exception as exc:
        logger.error("Errore GPT generazione candidati: %s", exc)
        return

    new_questions = [
        line.strip().rstrip("?") + "?"
        for line in raw.splitlines()
        if line.strip() and "?" in line
    ][:3]  # max 3 nuove candidature alla volta

    for q in new_questions:
        if q not in existing:
            pool.setdefault("candidates", []).append({"text": q, "count": 0, "responses": []})
            logger.info("Nuova domanda candidata: '%s'", q)


def _summarise_events(events: list[dict]) -> str:
    if not events:
        return "(nessun evento)"
    labels = [e.get("activity_label") or "?" for e in events]
    from collections import Counter
    counts = Counter(labels)
    total  = len(events)
    return "  " + "\n  ".join(
        f"{label}: {n}/{total} ({n*100//total}%)" for label, n in counts.most_common()
    )


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import config as _cfg
    from memory.db import init_db, log_event, set_preference as _set_pref

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_db()

    # reset pool
    _set_pref(_POOL_KEY, json.dumps({"active": [], "candidates": [], "suspended": []}))
    _set_pref(_LAST_GEN_KEY, "0")

    print("=== 1. get_active_questions (solo fisse) ===")
    qs = get_active_questions()
    assert qs == list(_cfg.PASSIVE_QUESTIONS_FIXED), f"attese solo fisse, got {qs}"
    print(f"  {len(qs)} domande fisse  OK")

    print("\n=== 2. promozione candidato ===")
    pool = _load_pool()
    candidate = {"text": "Is the person wearing headphones?", "count": 0, "responses": []}
    pool["candidates"].append(candidate)
    _save_pool(pool)

    # simula QUESTIONS_VALIDATE_ON risposte variate
    for i in range(_cfg.QUESTIONS_VALIDATE_ON):
        answers = {"Is the person wearing headphones?": "yes" if i % 2 == 0 else "no"}
        maybe_evolve(recent_answers=answers)

    status = pool_status()
    print(f"  Attive:     {status['active']}")
    print(f"  Candidati:  {status['candidates']}")
    promoted = "Is the person wearing headphones?" in status["active"]
    print(f"  Promossa:   {'OK' if promoted else 'FAIL'}")

    print("\n=== 3. scarto candidato monotono ===")
    pool = _load_pool()
    pool["candidates"].append({"text": "Is there a plant in the room?", "count": 0, "responses": []})
    _save_pool(pool)

    for _ in range(_cfg.QUESTIONS_VALIDATE_ON):
        maybe_evolve(recent_answers={"Is there a plant in the room?": "yes"})

    status = pool_status()
    discarded = "Is there a plant in the room?" not in status["active"] and \
                "Is there a plant in the room?" not in status["candidates"]
    print(f"  Candidato monotono scartato: {'OK' if discarded else 'FAIL'}")

    print("\n=== 4. sospensione domanda statica ===")
    pool = _load_pool()
    pool["active"].append("Is the window open?")
    _save_pool(pool)

    # patch config per test rapido
    orig_suspend = _cfg.QUESTIONS_SUSPEND_STATIC
    _cfg.QUESTIONS_SUSPEND_STATIC = 5

    for _ in range(5):
        log_event(True, True, True, extra_questions={"Is the window open?": "no"})
    maybe_evolve()

    status = pool_status()
    suspended = "Is the window open?" in status["suspended"]
    print(f"  Domanda sospesa (risposta sempre 'no'): {'OK' if suspended else 'FAIL'}")

    _cfg.QUESTIONS_SUSPEND_STATIC = orig_suspend

    print("\n=== Tutti i test superati ===")
