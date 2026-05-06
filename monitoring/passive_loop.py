import logging
import threading
import time

from PIL import Image

import config
from memory.logger import write_observation
from memory.learner import get_active_questions as _learner_questions, maybe_evolve

logger = logging.getLogger(__name__)


class PassiveLoop:
    """
    Monitoraggio passivo non-bloccante.

    Gira in un thread daemon indipendente dal main loop (wake word, voce).
    Ogni ciclo:
      1. Cattura un frame dalla camera
      2. Invia le domande attive a moondream2 in un'unica chiamata batch
      3. Loga il risultato + tempo inferenza in SQLite
      4. Dorme per max(MIN_INTERVAL, elapsed + BUFFER) - elapsed
         → nessun backlog: il ciclo successivo parte sempre dopo la fine del precedente

    Dipendenze iniettabili (camera, ask_fn) per permettere i test senza hardware.
    """

    def __init__(self, camera=None, ask_fn=None, camera_lock=None, is_allowed_fn=None):
        self._ext_camera = camera          # None → la loop crea e gestisce la sua Camera
        self._ask_fn = ask_fn or _default_ask
        self._camera_lock: threading.Lock | None = camera_lock  # condiviso con tier2
        self._is_allowed_fn = is_allowed_fn  # callable() → bool; None = sempre permesso

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_inference_ms: int | None = None
        self._cycles_completed: int = 0

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="passive-loop", daemon=True
        )
        self._thread.start()
        logger.info("PassiveLoop avviato.")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("PassiveLoop fermato dopo %d cicli.", self._cycles_completed)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_inference_ms(self) -> int | None:
        return self._last_inference_ms

    @property
    def cycles_completed(self) -> int:
        return self._cycles_completed

    # ------------------------------------------------------------------
    # thread entry point
    # ------------------------------------------------------------------

    def _run(self) -> None:
        # modalità test: camera esterna iniettata
        if self._ext_camera is not None:
            while not self._stop_event.is_set():
                self._cycle(self._ext_camera)
            return

        # modalità produzione: apri/chiudi camera ogni ciclo con lock condiviso.
        # Il lock è condiviso con il tier2 "guardami" in main.py: garantisce che
        # passive loop e tier2 non usino la camera simultaneamente.
        from vision.camera import Camera, CameraError
        while not self._stop_event.is_set():
            self._cycle_own_camera(Camera, CameraError)

    def _cycle_own_camera(self, Camera, CameraError) -> None:
        if self._is_allowed_fn is not None and not self._is_allowed_fn():
            self._sleep(config.PASSIVE_LOOP_MIN_INTERVAL_S)
            return

        lock = self._camera_lock
        acquired = lock.acquire(timeout=6.0) if lock else True
        if not acquired:
            logger.warning("Camera lock timeout — ciclo passivo saltato.")
            self._sleep(config.PASSIVE_LOOP_MIN_INTERVAL_S)
            return

        t_start = time.monotonic()
        frame = None
        try:
            with Camera() as cam:
                frame = cam.capture_frame()
        except CameraError as exc:
            logger.error("Camera: %s", exc)
        finally:
            if lock:
                lock.release()

        if frame is None:
            self._sleep(config.PASSIVE_LOOP_MIN_INTERVAL_S)
            return

        self._process_frame(frame, t_start)

    # ------------------------------------------------------------------
    # single cycle (modalità test con camera iniettata)
    # ------------------------------------------------------------------

    def _cycle(self, cam) -> None:
        if self._is_allowed_fn is not None and not self._is_allowed_fn():
            self._sleep(config.PASSIVE_LOOP_MIN_INTERVAL_S)
            return
        t_start = time.monotonic()
        try:
            frame = cam.capture_frame()
        except Exception as exc:
            logger.error("Errore cattura frame: %s", exc)
            self._sleep(config.PASSIVE_LOOP_MIN_INTERVAL_S)
            return
        self._process_frame(frame, t_start)

    def _process_frame(self, frame, t_start: float) -> None:
        # 1. inferenza batch — tutte le domande in una sola chiamata
        questions = self._active_questions()
        try:
            answers, ms = self._ask_fn(frame, questions)
        except Exception as exc:
            logger.error("Errore inferenza moondream: %s", exc)
            self._sleep(config.PASSIVE_LOOP_MIN_INTERVAL_S)
            return

        self._last_inference_ms = ms
        elapsed = time.monotonic() - t_start

        q_room, q_desk, q_pc = config.PASSIVE_QUESTIONS_FIXED
        in_room = answers.get(q_room, "?")
        at_desk = answers.get(q_desk, "?")
        at_pc   = answers.get(q_pc,   "?")

        # 2. log in SQLite (parsing + activity_label + confidence via logger)
        try:
            write_observation(answers, inference_time_ms=ms)
        except Exception as exc:
            logger.error("Errore write_observation: %s", exc)

        # 3. evoluzione pool domande (non-bloccante su errore)
        try:
            maybe_evolve(recent_answers=answers)
        except Exception as exc:
            logger.warning("Errore maybe_evolve: %s", exc)

        self._cycles_completed += 1
        logger.debug(
            "[ciclo %d] in_room=%s at_desk=%s at_pc=%s | infer=%dms elapsed=%.1fs",
            self._cycles_completed, in_room, at_desk, at_pc, ms, elapsed,
        )

        # 4. sleep adattivo
        target = max(
            config.PASSIVE_LOOP_MIN_INTERVAL_S,
            elapsed + config.PASSIVE_LOOP_BUFFER_S,
        )
        self._sleep(max(0.0, target - elapsed))

    # ------------------------------------------------------------------
    # question pool
    # ------------------------------------------------------------------

    def _active_questions(self) -> list[str]:
        """Fisse + domande custom validate dal learner."""
        try:
            return _learner_questions()
        except Exception:
            return list(config.PASSIVE_QUESTIONS_FIXED)

    # ------------------------------------------------------------------
    # interruptible sleep
    # ------------------------------------------------------------------

    def _sleep(self, seconds: float) -> None:
        """Sleep interrompibile dallo stop_event con granularità 0.25s."""
        deadline = time.monotonic() + seconds
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.25, remaining))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _default_ask(
    image: Image.Image, questions: list[str]
) -> tuple[dict[str, str], int]:
    from vision.tier1 import ask_frame_batch
    return ask_frame_batch(image, questions)


# ---------------------------------------------------------------------------
# self-test (mock hardware — nessuna camera né ollama richiesti)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    from memory.db import init_db, count_events, get_recent_events, avg_inference_time_ms

    # ------------------------------------------------------------------
    # 1. timing adattivo
    # ------------------------------------------------------------------
    print("=== 1. timing adattivo ===")
    min_i = config.PASSIVE_LOOP_MIN_INTERVAL_S   # 10
    buf   = config.PASSIVE_LOOP_BUFFER_S          # 2
    scenarios = [
        (1,  max(0, max(min_i, 1  + buf) - 1)),   # veloce  → sleep 9s
        (5,  max(0, max(min_i, 5  + buf) - 5)),   # media   → sleep 5s
        (15, max(0, max(min_i, 15 + buf) - 15)),  # lenta   → sleep 2s
        (30, max(0, max(min_i, 30 + buf) - 30)),  # lentiss.→ sleep 2s
    ]
    for elapsed, expected_sleep in scenarios:
        target = max(min_i, elapsed + buf)
        sleep  = max(0.0, target - elapsed)
        print(f"  inferenza {elapsed:2}s → target={target:2}s  sleep={sleep:.0f}s  {'OK' if abs(sleep - expected_sleep) < 0.01 else 'FAIL'}")

    # ------------------------------------------------------------------
    # 2. integrazione loop con mock (cicli veloci)
    # ------------------------------------------------------------------
    print("\n=== 2. loop integrato (mock camera + mock moondream) ===")

    # patch config per cicli veloci nel test
    config.PASSIVE_LOOP_MIN_INTERVAL_S = 1
    config.PASSIVE_LOOP_BUFFER_S       = 0.2
    MOCK_INFERENCE_MS = 300

    class MockCamera:
        def capture_frame(self):
            return Image.new("RGB", (64, 64), color=(80, 120, 160))

    def mock_ask(image, questions):
        time.sleep(MOCK_INFERENCE_MS / 1000)
        return {
            config.PASSIVE_QUESTIONS_FIXED[0]: "yes",
            config.PASSIVE_QUESTIONS_FIXED[1]: "yes",
            config.PASSIVE_QUESTIONS_FIXED[2]: "no",
        }, MOCK_INFERENCE_MS

    # maybe_evolve può fare chiamate GPT reali — no-op nel test.
    # globals() modifica il dizionario di __main__ (questo stesso file),
    # che è il namespace dove _process_frame cerca 'maybe_evolve' a runtime.
    globals()["maybe_evolve"] = lambda *a, **kw: None

    init_db()
    n_before = count_events()

    loop = PassiveLoop(camera=MockCamera(), ask_fn=mock_ask)
    loop.start()
    assert loop.is_running, "thread non avviato"

    time.sleep(3.8)
    loop.stop(timeout=3.0)
    assert not loop.is_running, "thread non terminato"

    n_after    = count_events()
    new_events = n_after - n_before
    print(f"\n  Cicli completati: {loop.cycles_completed}")
    print(f"  Events in SQLite: {new_events}")
    print(f"  Ultimo inference: {loop.last_inference_ms}ms")

    events = get_recent_events(max(1, new_events))
    print(f"\n  Ultimi {len(events)} eventi:")
    for e in reversed(events[-3:]):
        ts = e["timestamp"][11:19]
        print(f"    [{ts}] in_room={e['in_room']} at_desk={e['at_desk']} "
              f"at_pc={e['at_pc']}  infer={e['inference_time_ms']}ms")

    avg = avg_inference_time_ms(last_n=max(1, new_events))
    if avg:
        print(f"\n  Media inferenza (ultimi {new_events}): {avg:.0f}ms")

    # verifica struttura: almeno 1 evento scritto con i campi corretti
    assert new_events >= 1, f"Nessun evento scritto"
    assert loop.last_inference_ms == MOCK_INFERENCE_MS
    last_e = get_recent_events(1)[0]
    assert last_e["in_room"] == 1
    assert last_e["at_desk"] == 1
    assert last_e["at_pc"]   == 0
    assert last_e["inference_time_ms"] == MOCK_INFERENCE_MS
    assert last_e["activity_label"] == "scrivania"   # at_desk=True, at_pc=False

    print("\n=== Tutti i test superati ===")
