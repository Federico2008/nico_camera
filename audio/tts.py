import subprocess
import logging

import config

logger = logging.getLogger(__name__)


def is_available() -> bool:
    return config.PIPER_BIN.exists() and config.PIPER_VOICE.exists()


def speak(text: str) -> None:
    """Sintetizza text con Piper e riproduce con aplay.

    Il testo viene passato via stdin (non via shell echo) per evitare
    injection su caratteri speciali come virgolette o backtick.
    """
    if not text.strip():
        return

    if not is_available():
        logger.error(
            "Piper non trovato.\n  BIN:   %s\n  VOICE: %s",
            config.PIPER_BIN, config.PIPER_VOICE,
        )
        return

    try:
        piper = subprocess.run(
            [
                str(config.PIPER_BIN),
                "--model", str(config.PIPER_VOICE),
                "--output_file", str(config.PIPER_OUTPUT_FILE),
            ],
            input=text.encode("utf-8"),
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        if piper.returncode != 0:
            logger.error("Piper uscito con codice %d", piper.returncode)
            return

        subprocess.run(
            ["aplay", str(config.PIPER_OUTPUT_FILE)],
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except FileNotFoundError as exc:
        logger.error("Binario non trovato: %s", exc)
    except subprocess.TimeoutExpired:
        logger.error("TTS timeout superato.")


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Piper BIN:   {config.PIPER_BIN}")
    print(f"Piper VOICE: {config.PIPER_VOICE}")
    print(f"Disponibile: {is_available()}")

    if not is_available():
        print("\nPiper non trovato — verifica i path in .env")
    else:
        print("\nTest riproduzione audio...")
        speak("Ciao, sono Nico. Sistema audio operativo.")
        print("Completato. Hai sentito l'audio?")
