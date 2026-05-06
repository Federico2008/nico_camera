"""
Wake word detection.

Backend selezionato automaticamente:
  1. Porcupine  — se PORCUPINE_ACCESS_KEY e PORCUPINE_MODEL_PATH sono in .env
  2. openWakeWord — fallback locale (wake word "alexa" built-in)

Setup Porcupine con "Nico" personalizzata:
  1. console.picovoice.ai → crea account gratuito
  2. "AccessKey" tab → crea chiave → copia in .env: PORCUPINE_ACCESS_KEY=...
  3. "Wake Word" tab → "Train" → digita "Nico" → seleziona "Linux (ARM64)" → Download
  4. Salva in assets/nico_linux.ppn
  5. .env: PORCUPINE_MODEL_PATH=/home/federico/Desktop/nico_camera/assets/nico_linux.ppn
"""

import logging
import threading
import time

import numpy as np

import config

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False

try:
    from openwakeword.model import Model as _OWWModel
    _OWW_OK = True
except ImportError:
    _OWW_OK = False

try:
    import pvporcupine
    _PORCUPINE_OK = True
except ImportError:
    _PORCUPINE_OK = False


class WakeWordError(RuntimeError):
    pass


def _oww_resolve_path(name_or_path: str) -> str:
    """Risolve nome breve ('alexa') → path .onnx, oppure ritorna path diretto."""
    import os
    if os.path.isfile(name_or_path):
        return name_or_path
    from openwakeword import get_pretrained_model_paths
    matches = [p for p in get_pretrained_model_paths() if name_or_path.lower() in p.lower()]
    if not matches:
        raise WakeWordError(
            f"Modello openWakeWord '{name_or_path}' non trovato.\n"
            f"Disponibili: {[p.split('/')[-1] for p in get_pretrained_model_paths()]}"
        )
    return matches[0]


def _is_porcupine_configured() -> bool:
    return bool(config.PORCUPINE_ACCESS_KEY and config.PORCUPINE_MODEL_PATH)


class WakeWordDetector:
    """
    Ascolta il microfono in un thread daemon e chiama on_detect()
    quando la wake word viene rilevata.

    Backend: Porcupine se configurato via .env, altrimenti openWakeWord.
    """

    _DEBOUNCE_S = 2.0

    def __init__(self):
        self._oww_model = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def start_background(self, on_detect: callable) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        if _is_porcupine_configured():
            backend = "porcupine"
        elif config.WAKE_WORD_CUSTOM_MODEL:
            backend = "custom-nico"
        else:
            backend = "openwakeword"
        self._thread = threading.Thread(
            target=self._listen,
            args=(on_detect,),
            name="wake-word",
            daemon=True,
        )
        self._thread.start()
        logger.info("WakeWordDetector avviato (backend: %s).", backend)

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("WakeWordDetector fermato.")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    def _listen(self, on_detect: callable) -> None:
        if not _SD_OK:
            logger.error("sounddevice non installato. WakeWordDetector non disponibile.")
            return
        if _is_porcupine_configured():
            self._listen_porcupine(on_detect)
        elif config.WAKE_WORD_CUSTOM_MODEL:
            self._listen_custom(on_detect)
        else:
            self._listen_oww(on_detect)

    # ------------------------------------------------------------------
    # backend: Porcupine
    # ------------------------------------------------------------------

    def _listen_porcupine(self, on_detect: callable) -> None:
        if not _PORCUPINE_OK:
            logger.error("pvporcupine non installato. pip install pvporcupine")
            return

        try:
            porcupine = pvporcupine.create(
                access_key=config.PORCUPINE_ACCESS_KEY,
                keyword_paths=[config.PORCUPINE_MODEL_PATH],
            )
        except Exception as exc:
            logger.error("Porcupine init fallito: %s", exc)
            return

        logger.info(
            "Porcupine pronto (frame_length=%d, sample_rate=%d).",
            porcupine.frame_length, porcupine.sample_rate,
        )

        try:
            with sd.InputStream(
                samplerate=porcupine.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=porcupine.frame_length,
            ) as stream:
                logger.info("In ascolto (Porcupine — 'Nico')...")
                while not self._stop_event.is_set():
                    pcm, _ = stream.read(porcupine.frame_length)
                    result = porcupine.process(pcm.flatten().tolist())
                    if result >= 0:
                        logger.info("Wake word 'Nico' rilevata!")
                        try:
                            on_detect()
                        except Exception as exc:
                            logger.error("Errore in on_detect: %s", exc)
                        self._stop_event.wait(self._DEBOUNCE_S)
        except sd.PortAudioError as exc:
            logger.error("Errore audio: %s", exc)
        finally:
            porcupine.delete()

    # ------------------------------------------------------------------
    # backend: modello custom embedding-based (Nico)
    # ------------------------------------------------------------------

    def _listen_custom(self, on_detect: callable) -> None:
        """
        Rilevazione custom 'Nico' via LogisticRegression su embedding OWW.

        Ogni 0.5s analizza l'ultima finestra da 1.5s:
          audio → AudioFeatures.embed_clips → mean(frames) → clf.predict_proba
        """
        try:
            import joblib
            from openwakeword.utils import AudioFeatures
        except ImportError as e:
            logger.error("Dipendenza mancante per modello custom: %s", e)
            return

        try:
            clf = joblib.load(config.WAKE_WORD_CUSTOM_MODEL)
        except Exception as exc:
            logger.error("Caricamento modello custom fallito: %s", exc)
            return

        af = AudioFeatures()
        logger.info("Modello custom 'Nico' caricato da %s.", config.WAKE_WORD_CUSTOM_MODEL)

        # finestra scorrevole: 1.5s buffer, slide ogni 0.5s
        WIN_SAMPLES   = int(config.AUDIO_SAMPLE_RATE * 1.5)   # 24000
        SLIDE_SAMPLES = int(config.AUDIO_SAMPLE_RATE * 0.5)   # 8000
        buffer = np.zeros(WIN_SAMPLES, dtype=np.int16)

        try:
            with sd.InputStream(
                samplerate=config.AUDIO_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=SLIDE_SAMPLES,
            ) as stream:
                logger.info("In ascolto (modello custom 'Nico')...")
                while not self._stop_event.is_set():
                    chunk, _ = stream.read(SLIDE_SAMPLES)
                    chunk = chunk.flatten()
                    # slide buffer
                    buffer[:WIN_SAMPLES - SLIDE_SAMPLES] = buffer[SLIDE_SAMPLES:]
                    buffer[WIN_SAMPLES - SLIDE_SAMPLES:] = chunk

                    emb = af.embed_clips(buffer[np.newaxis, :])   # (1, frames, 96)
                    feat = emb.mean(axis=1)                        # (1, 96)
                    score = clf.predict_proba(feat)[0, 1]          # P(Nico)

                    if score >= config.WAKE_WORD_THRESHOLD:
                        logger.info("'Nico' rilevato (score=%.3f)!", score)
                        try:
                            on_detect()
                        except Exception as exc:
                            logger.error("Errore in on_detect: %s", exc)
                        # svuota buffer + debounce
                        buffer[:] = 0
                        self._stop_event.wait(self._DEBOUNCE_S)
        except sd.PortAudioError as exc:
            logger.error("Errore audio: %s", exc)

    # ------------------------------------------------------------------
    # backend: openWakeWord (fallback)
    # ------------------------------------------------------------------

    def _listen_oww(self, on_detect: callable) -> None:
        try:
            self._ensure_oww_model()
        except WakeWordError as exc:
            logger.error("%s", exc)
            return

        try:
            with sd.InputStream(
                samplerate=config.AUDIO_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=config.AUDIO_CHUNK_SIZE,
            ) as stream:
                logger.info("In ascolto (openWakeWord — '%s')...", config.WAKE_WORD_MODEL)
                while not self._stop_event.is_set():
                    chunk, _ = stream.read(config.AUDIO_CHUNK_SIZE)
                    score = self._oww_score(chunk.flatten())
                    if score >= config.WAKE_WORD_THRESHOLD:
                        logger.info("Wake word rilevata (score=%.3f)!", score)
                        try:
                            on_detect()
                        except Exception as exc:
                            logger.error("Errore in on_detect: %s", exc)
                        self._stop_event.wait(self._DEBOUNCE_S)
        except sd.PortAudioError as exc:
            logger.error("Errore audio: %s", exc)

    def _ensure_oww_model(self) -> None:
        if self._oww_model is not None:
            return
        if not _OWW_OK:
            raise WakeWordError("openwakeword non installato. pip install openwakeword")
        path = _oww_resolve_path(config.WAKE_WORD_MODEL)
        logger.info("Caricamento modello openWakeWord '%s'...", path)
        self._oww_model = _OWWModel(wakeword_model_paths=[path])
        logger.info("Modello openWakeWord pronto.")

    def _oww_score(self, audio_chunk: np.ndarray) -> float:
        predictions: dict = self._oww_model.predict(audio_chunk)
        if predictions:
            return float(max(predictions.values()))
        return 0.0


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-7s %(message)s")

    print(f"sounddevice disponibile:   {_SD_OK}")
    print(f"openwakeword disponibile:  {_OWW_OK}")
    print(f"pvporcupine disponibile:   {_PORCUPINE_OK}")
    print(f"Porcupine configurato:     {_is_porcupine_configured()}")
    if _is_porcupine_configured():
        _backend = "porcupine"
    elif config.WAKE_WORD_CUSTOM_MODEL:
        _backend = "custom-nico"
    else:
        _backend = "openwakeword"
    print(f"Backend attivo:            {_backend}")

    # ------------------------------------------------------------------
    # 1. _is_porcupine_configured() con vari valori config
    # ------------------------------------------------------------------
    print("\n=== 1. _is_porcupine_configured() ===")
    orig_key  = config.PORCUPINE_ACCESS_KEY
    orig_path = config.PORCUPINE_MODEL_PATH

    cases = [
        ("key123", "/path/to/nico.ppn", True),
        ("",       "/path/to/nico.ppn", False),
        ("key123", "",                  False),
        ("",       "",                  False),
    ]
    for key, path, expected in cases:
        config.PORCUPINE_ACCESS_KEY  = key
        config.PORCUPINE_MODEL_PATH  = path
        got = _is_porcupine_configured()
        ok  = "OK" if got == expected else f"FAIL (got {got})"
        print(f"  key={key!r:<8} path={path!r:<25} → {got}  {ok}")

    config.PORCUPINE_ACCESS_KEY = orig_key
    config.PORCUPINE_MODEL_PATH = orig_path

    # ------------------------------------------------------------------
    # 2. openWakeWord score con mock model
    # ------------------------------------------------------------------
    print("\n=== 2. openWakeWord score ===")

    det = WakeWordDetector()
    cases = [
        ("alexa", {"alexa": 0.9},          0.9),
        ("alexa", {"alexa--custom": 0.85}, 0.85),
        ("alexa", {"hey_siri": 0.3},        0.3),
        ("alexa", {},                        0.0),
    ]
    for model_name, preds, expected in cases:
        config.WAKE_WORD_MODEL = model_name
        class _M:
            def predict(self, c, _p=preds): return _p
        det._oww_model = _M()
        got = det._oww_score(np.zeros(config.AUDIO_CHUNK_SIZE, dtype=np.int16))
        ok  = "OK" if abs(got - expected) < 0.001 else f"FAIL (got {got})"
        print(f"  preds={str(preds):<35} expected={expected}  {ok}")

    config.WAKE_WORD_MODEL = "alexa"

    # ------------------------------------------------------------------
    # 3. Porcupine dispatch con mock — rileva e chiama on_detect
    # ------------------------------------------------------------------
    print("\n=== 3. Porcupine dispatch (mock) ===")

    detected_porcupine: list[int] = []

    class _MockPorcupine:
        frame_length = 512
        sample_rate  = 16000
        _call        = 0

        def process(self, pcm):
            self._call += 1
            return 0 if self._call == 2 else -1   # rileva al 2° frame

        def delete(self): pass

    class _DetPorcupine(WakeWordDetector):
        def _listen_porcupine(self, on_detect):
            porcupine = _MockPorcupine()
            for _ in range(4):
                if self._stop_event.is_set():
                    break
                pcm = np.zeros(porcupine.frame_length, dtype=np.int16).tolist()
                result = porcupine.process(pcm)
                if result >= 0:
                    detected_porcupine.append(1)
                    try:
                        on_detect()
                    except Exception as exc:
                        logger.error("%s", exc)
                    self._stop_event.wait(0.05)   # debounce breve nel test
            porcupine.delete()

    config.PORCUPINE_ACCESS_KEY = "test_key"
    config.PORCUPINE_MODEL_PATH = "test.ppn"

    pd = _DetPorcupine()
    pd.start_background(on_detect=lambda: detected_porcupine.append(99))
    time.sleep(0.3)
    pd.stop(timeout=1.0)

    config.PORCUPINE_ACCESS_KEY = orig_key
    config.PORCUPINE_MODEL_PATH = orig_path

    assert len(detected_porcupine) >= 1, f"nessuna rilevazione porcupine: {detected_porcupine}"
    print(f"  Rilevazioni: {detected_porcupine}  OK")

    # ------------------------------------------------------------------
    # 4. stop() e lifecycle thread
    # ------------------------------------------------------------------
    print("\n=== 4. stop() e thread lifecycle ===")

    call_count: list[int] = []

    class _DetectorMock(WakeWordDetector):
        def _listen(self, on_detect):
            for _ in range(3):
                if self._stop_event.is_set():
                    break
                on_detect()
                self._stop_event.wait(0.05)

    mock_det = _DetectorMock()
    mock_det.start_background(lambda: call_count.append(1))
    assert mock_det.is_running
    time.sleep(0.4)
    mock_det.stop(timeout=2.0)
    assert not mock_det.is_running
    print(f"  Attivazioni: {len(call_count)} (attese 3)  {'OK' if len(call_count) == 3 else 'FAIL'}")

    # ------------------------------------------------------------------
    # 5. WakeWordError per modello inesistente
    # ------------------------------------------------------------------
    print("\n=== 5. WakeWordError modello sconosciuto ===")
    try:
        _oww_resolve_path("modello_inesistente_xyz_123")
        print("  FAIL — avrebbe dovuto sollevare WakeWordError")
    except WakeWordError:
        print("  WakeWordError sollevata correttamente  OK")

    print("\n=== Test wake_word completati ===")
