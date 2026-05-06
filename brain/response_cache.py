import re
import time
from datetime import datetime

_DAYS_IT   = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
_MONTHS_IT = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
               "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]


def _now_time() -> str:
    return f"Sono le {datetime.now().strftime('%H:%M')}."

def _now_date() -> str:
    n = datetime.now()
    return f"Oggi è {_DAYS_IT[n.weekday()]}, {n.day} {_MONTHS_IT[n.month - 1]} {n.year}."


_DIRECT: list[tuple[str, callable, int]] = [
    (r"che ore sono|che ora è|dimmi l.?ora|ora è", _now_time, 30),
    (r"che (giorno|data) (è|abbiamo)|quanti ne abbiamo|che giorno", _now_date, 3600),
]

_cache: dict[str, tuple[str, float]] = {}
_hits = 0


def try_direct_answer(text: str) -> str | None:
    low = text.lower().strip()
    for pattern, fn, _ in _DIRECT:
        if re.search(pattern, low):
            return fn()
    return None


def get_cached(text: str) -> str | None:
    global _hits
    entry = _cache.get(text.lower().strip())
    if entry and time.monotonic() < entry[1]:
        _hits += 1
        return entry[0]
    return None


def cache_response(text: str, response: str, ttl_s: float = 300.0) -> None:
    _cache[text.lower().strip()] = (response, time.monotonic() + ttl_s)
    _evict()


def cache_stats() -> dict:
    return {"entries": len(_cache), "hits": _hits}


def _evict() -> None:
    now = time.monotonic()
    for k in [k for k, v in _cache.items() if v[1] <= now]:
        del _cache[k]
