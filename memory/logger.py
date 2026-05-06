"""
Strato di scrittura eventi arricchito.

Il passive_loop chiama write_observation() invece di log_event() direttamente:
ottiene activity_label e confidence calcolati automaticamente, senza duplicare
la logica di parsing nelle due parti del codice.
"""

import config
from memory.db import log_event


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------

def write_observation(
    answers: dict[str, str],
    inference_time_ms: int,
) -> int:
    """
    Traduce le risposte di moondream in un evento DB arricchito.
    Restituisce l'ID dell'evento scritto.

    answers deve contenere le chiavi di config.PASSIVE_QUESTIONS_FIXED più
    eventuali domande extra dal learner.
    """
    q_room, q_desk, q_pc = config.PASSIVE_QUESTIONS_FIXED

    in_room = _parse_bool(answers.get(q_room))
    at_desk = _parse_bool(answers.get(q_desk))
    at_pc   = _parse_bool(answers.get(q_pc))

    activity_label = _derive_activity(in_room, at_desk, at_pc)
    confidence     = _compute_confidence(answers)

    extra = {
        q: a for q, a in answers.items()
        if q not in config.PASSIVE_QUESTIONS_FIXED
    } or None

    return log_event(
        in_room=in_room,
        at_desk=at_desk,
        at_pc=at_pc,
        activity_label=activity_label,
        confidence=confidence,
        extra_questions=extra,
        inference_time_ms=inference_time_ms,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_bool(answer: str | None) -> bool | None:
    if answer is None:
        return None
    words = answer.lower().strip().split()
    if not words:
        return None
    tok = words[0].strip(".,;:!?")
    if tok in ("yes", "true", "1", "sì", "si"):
        return True
    if tok in ("no", "false", "0"):
        return False
    return None


def _derive_activity(
    in_room: bool | None,
    at_desk: bool | None,
    at_pc:   bool | None,
) -> str | None:
    if in_room is False:
        return "fuori_stanza"
    if in_room is True:
        if at_desk and at_pc:
            return "lavoro_pc"
        if at_desk:
            return "scrivania"
        return "in_stanza"
    return None  # in_room sconosciuto


def _compute_confidence(answers: dict[str, str]) -> float:
    """Frazione di risposte che risultano chiaramente yes/no (non ambigue)."""
    if not answers:
        return 0.0
    clear = sum(1 for a in answers.values() if _parse_bool(a) is not None)
    return round(clear / len(answers), 2)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from memory.db import init_db, get_recent_events

    init_db()

    print("=== _derive_activity ===")
    cases = [
        (True,  True,  True,  "lavoro_pc"),
        (True,  True,  False, "scrivania"),
        (True,  False, False, "in_stanza"),
        (False, None,  None,  "fuori_stanza"),
        (None,  True,  True,  None),
    ]
    for in_room, at_desk, at_pc, expected in cases:
        got = _derive_activity(in_room, at_desk, at_pc)
        ok  = "OK" if got == expected else f"FAIL (got {got})"
        print(f"  in_room={str(in_room):<5} at_desk={str(at_desk):<5} "
              f"at_pc={str(at_pc):<5} → {str(got):<14} {ok}")

    print("\n=== _compute_confidence ===")
    conf_cases = [
        ({"a": "yes", "b": "no", "c": "yes"},         1.0),
        ({"a": "yes", "b": "maybe", "c": "yes"},       0.67),
        ({"a": "perhaps", "b": "unclear"},              0.0),
        ({},                                            0.0),
    ]
    for answers, expected in conf_cases:
        got = _compute_confidence(answers)
        ok  = "OK" if abs(got - expected) < 0.02 else f"FAIL (got {got})"
        print(f"  answers={str(answers):<45} conf={got:.2f}  {ok}")

    print("\n=== write_observation (integrazione DB) ===")
    n_before = len(get_recent_events(100))
    answers_full = {
        config.PASSIVE_QUESTIONS_FIXED[0]: "yes",
        config.PASSIVE_QUESTIONS_FIXED[1]: "yes",
        config.PASSIVE_QUESTIONS_FIXED[2]: "no",
        "Is the person drinking coffee?": "yes",
    }
    event_id = write_observation(answers_full, inference_time_ms=2800)
    events = get_recent_events(1)
    e = events[0]
    print(f"  event_id={event_id}")
    print(f"  activity_label: {e['activity_label']}")
    print(f"  confidence:     {e['confidence']}")
    print(f"  extra_questions:{e['extra_questions']}")
    assert e["activity_label"] == "scrivania"
    assert e["confidence"] == 1.0   # tutte e 4 le risposte sono yes/no → 4/4
    print("  OK")
