#!/usr/bin/env python3
"""
Nico — entry point principale.

Thread model:
  main thread      blocca su signal.pause(), gestisce SIGINT/SIGTERM
  wake-word        daemon — chiama _on_wake_word() su attivazione
  passive-loop     daemon — cattura frame ogni ~10s
  stt-preload      daemon one-shot — carica Whisper al boot
"""

import logging
import signal
import sys
import threading

import config
from audio.stt import STT
from audio.tts import speak
from audio.wake_word import WakeWordDetector
from brain.context_builder import build as build_context
from brain.gpt import chat, chat_stream, chat_with_vision
from brain.response_cache import cache_response, get_cached, try_direct_answer
from brain.router import IntentType, RequestClass, classify, detect_intent
from memory.aggregator import start_background as start_aggregator
from memory.db import init_db
from monitoring.reminder_scheduler import start as start_reminder_scheduler
from web.dashboard import start as start_dashboard
from monitoring.passive_loop import PassiveLoop
from privacy.controller import PrivacyController

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# stato condiviso tra thread
# ---------------------------------------------------------------------------

_stt = STT()
_interaction_lock = threading.Lock()   # serializza le interazioni vocali
_camera_lock      = threading.Lock()   # una sola istanza Camera attiva alla volta

_passive_loop:  PassiveLoop        | None = None
_wake_detector: WakeWordDetector   | None = None
_privacy:       PrivacyController  | None = None


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _passive_loop, _wake_detector, _privacy

    _setup_logging()

    _privacy = PrivacyController()
    _privacy.setup(on_kill=_trigger_shutdown)

    _setup_signals()

    logger.info("Nico — avvio sistema...")
    init_db()
    start_aggregator()
    start_reminder_scheduler()
    start_dashboard()

    _passive_loop = PassiveLoop(
        camera_lock=_camera_lock,
        is_allowed_fn=_privacy.is_monitoring_allowed,
    )
    _passive_loop.start()

    threading.Thread(target=_preload_stt, daemon=True, name="stt-preload").start()

    _wake_detector = WakeWordDetector()
    _wake_detector.start_background(on_detect=_on_wake_word)

    _privacy.set_monitoring(True)
    speak("Sistema avviato. Dimmi Nico quando sei pronto.")
    logger.info("Sistema operativo — in ascolto.")

    try:
        signal.pause()   # blocca il main thread, il sistema gira nei daemon thread
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — spegnimento.")
        _trigger_shutdown()


# ---------------------------------------------------------------------------
# gestione wake word → interazione
# ---------------------------------------------------------------------------

def _on_wake_word() -> None:
    """Chiamata dal thread wake-word. Un'interazione alla volta via lock."""
    if not _interaction_lock.acquire(blocking=False):
        logger.debug("Interazione già in corso — wake word ignorata.")
        return
    try:
        _interact()
    finally:
        _interaction_lock.release()


def _stream_to_voice(token_iter, min_chars: int = 20) -> str:
    """
    Consuma uno stream di token GPT, parla frase per frase non appena
    si raggiunge un boundary (. ! ?) dopo min_chars caratteri.
    Restituisce il testo completo per il caching.
    """
    buf = ""
    spoken: list[str] = []
    for tok in token_iter:
        buf += tok
        while len(buf) >= min_chars:
            end = -1
            for i in range(min_chars - 1, len(buf)):
                if buf[i] in ".!?" and (i + 1 >= len(buf) or buf[i + 1] in " \n"):
                    end = i
                    break
            if end == -1:
                break
            sentence = buf[: end + 1].strip()
            if sentence:
                speak(sentence)
                spoken.append(sentence)
            buf = buf[end + 1 :].lstrip()
    if buf.strip():
        speak(buf.strip())
        spoken.append(buf.strip())
    return " ".join(spoken)


def _interact() -> None:
    if _privacy:
        _privacy.led_blink()
    try:
        for turn in range(6):
            speak("Dimmi.")

            text = _stt.record_and_transcribe(max_seconds=8.0)
            if not text:
                if turn == 0:
                    speak("Non ho sentito niente.")
                return

            logger.info("[richiesta] %s", text)

            direct = try_direct_answer(text)
            if direct:
                speak(direct)
                continue

            cached = get_cached(text)
            if cached:
                logger.info("[cache] hit")
                speak(cached)
                continue

            intent = detect_intent(text)
            if intent.intent != IntentType.NONE:
                logger.info("[intent] %s", intent.intent.value)
                if intent.response is not None:
                    speak(intent.response)
                    continue
                if intent.gpt_prompt is not None:
                    _stream_to_voice(chat_stream(intent.gpt_prompt, intent.gpt_context or ""))
                    continue

            result = classify(text)
            logger.info("[routing] Classe %s — %s", result.cls.value, result.reason)

            context = build_context(result.cls, user_text=text)

            if result.cls == RequestClass.C:
                _handle_vision(text, context)
            else:
                reply = _stream_to_voice(chat_stream(text, context))
                cache_response(text, reply)
    finally:
        if _privacy and _privacy.is_monitoring_allowed():
            _privacy.led_on()


def _handle_vision(text: str, context: str) -> None:
    """
    Classe C — cattura 3 frame live e li manda a GPT-4o Vision.
    Acquisisce il camera_lock per non collidere con il passive loop.
    """
    from vision.camera import Camera, CameraError

    speak("Un attimo, sto guardando.")

    if not _camera_lock.acquire(timeout=8.0):
        speak("La camera è occupata, riprova tra poco.")
        return
    try:
        with Camera() as cam:
            frames_b64 = cam.capture_frames_base64(n=3, interval_s=1.5)
    except CameraError as exc:
        logger.error("Errore camera tier2: %s", exc)
        speak("Non riesco ad accedere alla camera.")
        return
    finally:
        _camera_lock.release()

    reply = chat_with_vision(text, frames_b64, context)
    speak(reply)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _preload_stt() -> None:
    """Carica Whisper in background al boot — evita il lag alla prima trascrizione."""
    try:
        STT._get_model()
        logger.info("Whisper precaricato.")
    except Exception as exc:
        logger.warning("Precaricamento Whisper non riuscito: %s", exc)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-22s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _trigger_shutdown() -> None:
    """Punto unico di spegnimento — chiamato da SIGINT/SIGTERM e dal kill switch hardware."""
    try:
        if _wake_detector:
            _wake_detector.stop()
    except Exception:
        pass
    try:
        if _passive_loop:
            _passive_loop.stop()
    except Exception:
        pass
    try:
        if _privacy:
            _privacy.set_monitoring(False)
            _privacy.shutdown()
    except Exception:
        pass
    try:
        speak("Arrivederci.")
    except Exception:
        pass
    import os
    os._exit(0)


def _setup_signals() -> None:
    def _shutdown(signum, _frame):
        logger.info("Segnale %d ricevuto — spegnimento.", signum)
        _trigger_shutdown()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)


# ---------------------------------------------------------------------------
# self-test (mock di tutti i sottosistemi hardware)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    if "--test" not in sys.argv:
        main()
        sys.exit(0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    from PIL import Image
    from memory.db import init_db

    init_db()

    # _this è questo stesso modulo — patchiamo i nomi a livello di modulo
    # così _interact() e _handle_vision() vedono le versioni mock
    _this = sys.modules[__name__]

    spoken: list[str] = []

    _this.speak            = lambda text: spoken.append(text)
    _this.chat             = lambda text, ctx="": f"[risposta a: {text[:30]}]"
    _this.chat_stream      = lambda text, ctx="": iter([f"[risposta a: {text[:30]}]"])
    _this.chat_with_vision = lambda text, imgs, ctx="": f"[visione: {len(imgs)} frame]"

    _stt_queries: list[str] = []

    class _MockSTT:
        def record_and_transcribe(self, max_seconds=8.0):
            return _stt_queries.pop(0) if _stt_queries else None

    _this._stt = _MockSTT()

    class _MockCamera:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def capture_frames_base64(self, n=3, interval_s=1.5, quality=85):
            return ["fake_b64"] * n

    import vision.camera as _cam_mod
    _cam_mod.Camera = _MockCamera

    # ------------------------------------------------------------------
    # 1. interazione Classe A
    # ------------------------------------------------------------------
    print("=== 1. interazione Classe A (nessuna camera) ===")
    spoken.clear()
    _stt_queries.append("dimmi una barzelletta")
    _interact()
    print(f"  speak() chiamato {len(spoken)} volte: {spoken}")
    assert any("risposta" in s for s in spoken), f"risposta attesa, got {spoken}"
    print("  OK")

    # ------------------------------------------------------------------
    # 2. interazione Classe B (dati passivi)
    # ------------------------------------------------------------------
    print("\n=== 2. interazione Classe B (dati passivi) ===")
    spoken.clear()
    _stt_queries.append("da quanto sono alla scrivania")
    _interact()
    print(f"  speak() chiamato {len(spoken)} volte: {spoken}")
    assert any("risposta" in s for s in spoken)
    print("  OK")

    # ------------------------------------------------------------------
    # 3. interazione Classe C (guardami → vision tier2)
    # ------------------------------------------------------------------
    print("\n=== 3. interazione Classe C (guardami) ===")
    spoken.clear()
    _stt_queries.append("guardami")
    _interact()
    print(f"  speak() chiamato {len(spoken)} volte: {spoken}")
    assert any("visione" in s for s in spoken), f"risposta vision attesa, got {spoken}"
    print("  OK")

    # ------------------------------------------------------------------
    # 4. lock: seconda wake word ignorata durante interazione attiva
    # ------------------------------------------------------------------
    print("\n=== 4. interazioni concorrenti bloccate dal lock ===")
    entered: list[bool] = []

    def _count_entry():
        entered.append(True)
        _stt_queries.append("che ore sono")
        _interact()

    _this._interaction_lock.acquire()
    t = threading.Thread(target=_on_wake_word)
    t.start()
    t.join(timeout=0.3)
    _this._interaction_lock.release()
    t.join(timeout=0.5)

    assert not entered, f"_interact non avrebbe dovuto eseguire, invece: {entered}"
    print("  Lock ha bloccato la seconda interazione  OK")

    # ------------------------------------------------------------------
    # 5. camera_lock condiviso: nessun deadlock tra passive loop e tier2
    # ------------------------------------------------------------------
    print("\n=== 5. camera_lock condiviso passive loop ↔ tier2 ===")
    config.PASSIVE_LOOP_MIN_INTERVAL_S = 0.5
    config.PASSIVE_LOOP_BUFFER_S       = 0.1

    class _MockCamCapture:
        def capture_frame(self):
            return Image.new("RGB", (4, 4))

    def _mock_ask(img, qs):
        time.sleep(0.05)
        return {q: "yes" for q in qs}, 50

    cam_lock = threading.Lock()
    loop = PassiveLoop(camera=_MockCamCapture(), ask_fn=_mock_ask,
                       camera_lock=cam_lock)
    loop.start()
    time.sleep(0.3)

    cam_lock.acquire()    # simula tier2 che occupa la camera
    time.sleep(0.2)
    cam_lock.release()

    time.sleep(0.3)
    loop.stop(timeout=2.0)
    assert loop.cycles_completed >= 1, "nessun ciclo completato"
    print(f"  Cicli completati senza deadlock: {loop.cycles_completed}  OK")

    # ------------------------------------------------------------------
    # 6. privacy controller (senza GPIO hardware)
    # ------------------------------------------------------------------
    print("\n=== 6. privacy controller (senza GPIO) ===")
    from privacy.controller import PrivacyController

    priv = PrivacyController()
    priv.setup(on_kill=lambda: None)
    priv.set_monitoring(True)
    assert priv.is_monitoring_allowed()

    priv.enable_guest_mode()
    assert not priv.is_monitoring_allowed()

    priv.disable_guest_mode()
    assert priv.is_monitoring_allowed()

    priv.led_blink(hz=4)
    time.sleep(0.1)
    priv.led_off()
    priv.shutdown()
    print("  PrivacyController: guest_mode, monitoring, led_blink  OK")

    # ------------------------------------------------------------------
    # 7. passive loop rispetta is_allowed_fn (guest mode)
    # ------------------------------------------------------------------
    print("\n=== 7. passive loop + is_allowed_fn ===")
    config.PASSIVE_LOOP_MIN_INTERVAL_S = 0.2
    config.PASSIVE_LOOP_BUFFER_S       = 0.0

    captured: list[bool] = []

    class _MockCamGuest:
        def capture_frame(self):
            captured.append(True)
            return Image.new("RGB", (4, 4))

    allowed = threading.Event()
    allowed.set()

    loop2 = PassiveLoop(
        camera=_MockCamGuest(),
        ask_fn=_mock_ask,
        is_allowed_fn=allowed.is_set,
    )
    loop2.start()
    time.sleep(0.4)
    n_before_pause = len(captured)

    allowed.clear()          # guest mode ON → nessuna cattura
    captured_before = len(captured)
    time.sleep(0.5)
    n_during_pause = len(captured) - captured_before

    allowed.set()            # guest mode OFF → riprende
    time.sleep(0.4)
    loop2.stop(timeout=1.0)

    assert n_before_pause >= 1, "nessuna cattura prima del guest mode"
    assert n_during_pause == 0, f"cattura durante guest mode: {n_during_pause}"
    print(f"  Catture pre-pausa: {n_before_pause}  durante pausa: {n_during_pause}  OK")

    # ------------------------------------------------------------------
    # 8. aggregator thread daemon avviato
    # ------------------------------------------------------------------
    print("\n=== 8. aggregator background ===")
    from memory.aggregator import start_background as _start_agg

    agg_thread = _start_agg()
    assert agg_thread.is_alive(), "thread aggregator non avviato"
    print(f"  Thread '{agg_thread.name}' vivo  OK")

    print("\n=== Tutti i test superati ===")
