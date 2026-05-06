import base64
import io
import time

import requests
from PIL import Image

import config


class OllamaError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _to_base64(image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _call_ollama(b64_image: str, prompt: str) -> tuple[str, int]:
    """POST a ollama /api/generate. Restituisce (risposta, ms_impiegati)."""
    payload = {
        "model": config.MOONDREAM_MODEL,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
    except requests.ConnectionError as exc:
        raise OllamaError(
            f"ollama non raggiungibile su {config.OLLAMA_BASE_URL}. "
            "Avvialo con: ollama serve"
        ) from exc
    except requests.HTTPError as exc:
        raise OllamaError(f"ollama HTTP {resp.status_code}: {resp.text[:200]}") from exc

    ms = int((time.monotonic() - t0) * 1000)
    answer = resp.json().get("response", "").strip()
    return answer, ms


def _parse_numbered(raw: str, questions: list[str]) -> dict[str, str]:
    """
    Parsa risposte numerate. Accetta i separatori che moondream produce in pratica:
        1. yes  |  1- yes  |  1) yes  |  1 yes
    Fallback: usa la riga i-esima se nessun prefisso combacia.
    """
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    answers: dict[str, str] = {}
    for i, q in enumerate(questions):
        n = str(i + 1)
        match: str | None = None
        for line in lines:
            for sep in (".", "-", ")", " "):
                prefix = n + sep
                if line.startswith(prefix):
                    match = line[len(prefix):].strip().lstrip("- ")
                    break
            if match is not None:
                break
        if match is None:
            match = lines[i] if i < len(lines) else ""
        answers[q] = match
    return answers


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def ask_frame(image: Image.Image, question: str) -> str:
    """Domanda singola su un frame. Firma richiesta dal passive_loop."""
    answer, _ = _call_ollama(_to_base64(image), question)
    return answer


def ask_frame_timed(image: Image.Image, question: str) -> tuple[str, int]:
    """Come ask_frame, restituisce anche il tempo di inferenza in ms."""
    b64 = _to_base64(image)
    return _call_ollama(b64, question)


def ask_frame_batch(
    image: Image.Image,
    questions: list[str],
) -> tuple[dict[str, str], int]:
    """
    Invia tutte le domande in una sola chiamata ollama per ridurre la latenza.
    Restituisce ({domanda: risposta}, ms_totali).

    Il passive_loop usa questa funzione per le 3+ domande fisse + quelle adattive.
    Una singola codifica dell'immagine, un solo round-trip HTTP.
    """
    if not questions:
        return {}, 0
    if len(questions) == 1:
        answer, ms = ask_frame_timed(image, questions[0])
        return {questions[0]: answer}, ms

    b64 = _to_base64(image)
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    prompt = (
        "Answer each question about the image with a short response "
        "(yes, no, or a few words). Reply only with the number and answer:\n"
        + numbered
    )
    raw, ms = _call_ollama(b64, prompt)
    return _parse_numbered(raw, questions), ms


def is_available() -> bool:
    """Controlla che ollama sia up e moondream sia presente nella lista modelli."""
    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(config.MOONDREAM_MODEL in m for m in models)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print(f"ollama URL: {config.OLLAMA_BASE_URL}")
    print(f"Modello:    {config.MOONDREAM_MODEL}")
    print(f"Disponibile: {is_available()}")

    if not is_available():
        print(
            "\nollama/moondream non disponibile.\n"
            "  1. Installa ollama: curl -fsSL https://ollama.com/install.sh | sh\n"
            "  2. Scarica modello: ollama pull moondream\n"
            "  3. Verifica:        ollama serve (in un altro terminale)"
        )
        sys.exit(1)

    # usa il frame di test catturato da camera.py se esiste, altrimenti crea immagine sintetica
    import os
    test_img_path = "/tmp/nico_test.jpg"
    if os.path.exists(test_img_path):
        image = Image.open(test_img_path)
        print(f"\nImmagine: {test_img_path} ({image.size[0]}x{image.size[1]})")
    else:
        print("\nNessun frame da camera trovato — uso immagine sintetica 640x480")
        image = Image.new("RGB", (640, 480), color=(100, 120, 140))

    # --- test ask_frame ---
    print("\n[1] ask_frame (domanda singola)...")
    q = "Is there a person visible in the image?"
    t0 = time.monotonic()
    answer = ask_frame(image, q)
    ms = int((time.monotonic() - t0) * 1000)
    print(f"  Q: {q}")
    print(f"  A: {answer}")
    print(f"  Tempo: {ms}ms")

    # --- test ask_frame_batch ---
    print("\n[2] ask_frame_batch (3 domande fisse)...")
    answers, ms_batch = ask_frame_batch(image, config.PASSIVE_QUESTIONS_FIXED)
    print(f"  Tempo totale: {ms_batch}ms")
    for q, a in answers.items():
        print(f"  Q: {q[:55]:<55}  A: {a}")

    print(
        f"\nRisultato: inferenza batch completata in {ms_batch}ms. "
        f"Il loop passivo userà max({config.PASSIVE_LOOP_MIN_INTERVAL_S}s, "
        f"{ms_batch/1000:.1f}s + {config.PASSIVE_LOOP_BUFFER_S}s) come intervallo."
    )
