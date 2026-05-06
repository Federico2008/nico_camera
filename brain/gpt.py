import logging
from collections.abc import Iterator
from typing import Sequence

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

import config

logger = logging.getLogger(__name__)

_SYSTEM_BASE = """Sei Nico, un assistente AI personale che gira su Raspberry Pi 5.
Parli sempre in italiano. Rispondi in modo conciso e naturale — le tue risposte \
vengono lette ad alta voce, quindi evita elenchi puntati, markdown e simboli.
Hai accesso alla memoria delle routine dell'utente e, quando richiesto, alla camera."""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY non impostata nel file .env")
        _client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
        )
    return _client


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def chat(user_text: str, context: str = "") -> str:
    """
    Risposta testuale — Class A e B.
    context è il blocco prodotto da context_builder.build().
    """
    system = _build_system(context)
    messages = [{"role": "user", "content": user_text}]
    return _complete(config.BRAIN_MODEL, system, messages)


def chat_stream(user_text: str, context: str = "") -> Iterator[str]:
    """
    Versione streaming di chat() — yield token per token.
    Il chiamante accumula in frasi e passa a tts.speak() man mano.
    Errori di rete vengono yielded come stringa di fallback.
    """
    system = _build_system(context)
    client = _get_client()
    try:
        stream = client.chat.completions.create(
            model=config.BRAIN_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user_text}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except RateLimitError:
        yield "Ho bisogno di un momento, riprova tra poco."
    except APIConnectionError:
        yield "Non riesco a connettermi. Controlla la rete."
    except APIError as exc:
        logger.error("Errore API streaming: %s", exc)
        yield "Si è verificato un errore. Riprova."


def chat_with_vision(
    user_text: str,
    images_b64: Sequence[str],
    context: str = "",
) -> str:
    """
    Risposta con immagini allegate — Class C ("guardami").
    images_b64: lista di JPEG codificati in base64 (max 5 frame).
    """
    system = _build_system(context)
    content: list[dict] = [{"type": "text", "text": user_text}]
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    messages = [{"role": "user", "content": content}]
    return _complete(config.VISION_MODEL, system, messages)


# ---------------------------------------------------------------------------
# internal
# ---------------------------------------------------------------------------

def _build_system(context: str) -> str:
    if not context or context == "(nessun contesto disponibile)":
        return _SYSTEM_BASE
    return f"{_SYSTEM_BASE}\n\n{context}"


def _complete(model: str, system: str, messages: list[dict]) -> str:
    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
        )
        text = resp.choices[0].message.content or ""
        return text.strip()
    except RateLimitError:
        logger.warning("Rate limit OpenRouter raggiunto.")
        return "Ho bisogno di un momento, riprova tra poco."
    except APIConnectionError as exc:
        logger.error("Connessione a OpenRouter fallita: %s", exc)
        return "Non riesco a connettermi. Controlla la rete."
    except APIError as exc:
        logger.error("Errore API OpenRouter: %s", exc)
        return "Si è verificato un errore. Riprova."


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ------------------------------------------------------------------
    # 1. test costruzione system prompt
    # ------------------------------------------------------------------
    print("=== 1. system prompt ===")
    sys_no_ctx = _build_system("")
    sys_with_ctx = _build_system("[Preferenze]\n  lingua: italiano")
    assert _SYSTEM_BASE in sys_no_ctx
    assert "[Preferenze]" in sys_with_ctx
    assert _SYSTEM_BASE in sys_with_ctx
    print("  _build_system  OK")

    # ------------------------------------------------------------------
    # 2. test formato messaggi vision
    # ------------------------------------------------------------------
    print("\n=== 2. formato vision message ===")
    import base64
    from PIL import Image
    import io

    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    content: list[dict] = [{"type": "text", "text": "cosa vedi?"}]
    for frame_b64 in [b64, b64]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
        })

    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert len(content) == 3
    print(f"  Messaggio vision: {len(content)} parti (1 testo + 2 immagini)  OK")

    # ------------------------------------------------------------------
    # 3. test errore mancanza API key
    # ------------------------------------------------------------------
    print("\n=== 3. errore API key mancante ===")
    import brain.gpt as _gpt_mod
    orig_key = config.OPENROUTER_API_KEY
    orig_client = _gpt_mod._client
    config.OPENROUTER_API_KEY = ""
    _gpt_mod._client = None
    try:
        _get_client()
        print("  FAIL — avrebbe dovuto sollevare RuntimeError")
    except RuntimeError as e:
        print(f"  RuntimeError sollevata: OK")
    finally:
        config.OPENROUTER_API_KEY = orig_key
        _gpt_mod._client = orig_client

    # ------------------------------------------------------------------
    # 4. chiamata API reale (richiede OPENROUTER_API_KEY valida)
    # ------------------------------------------------------------------
    print("\n=== 4. chiamata API reale ===")
    if not config.OPENROUTER_API_KEY:
        print("  SKIP — OPENROUTER_API_KEY non impostata")
    else:
        import time
        t0 = time.time()
        try:
            reply = chat("Dimmi solo la parola 'funziona' senza aggiungere altro.")
            elapsed = time.time() - t0
            print(f"  Risposta: '{reply}'")
            print(f"  Tempo:    {elapsed:.1f}s")
            print(f"  OK" if reply else "  FAIL — risposta vuota")
        except Exception as exc:
            print(f"  Errore: {exc}")

    print("\n=== Test gpt.py completati ===")
