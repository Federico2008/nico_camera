import logging
import tempfile
import wave
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False

try:
    import whisper as _whisper_lib
    _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False

# Stringhe di allucinazione note di Whisper su silenzio/rumore italiano
_HALLUCINATIONS: set[str] = {
    "sottotitoli e revisione a cura di qtss",
    "sottotitoli a cura di",
    "grazie per l'attenzione",
    "iscriviti al canale",
    "like e iscriviti",
    "sottotitoli",
    "by qtss",
}


def _is_hallucination(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return True
    return any(h in low for h in _HALLUCINATIONS)


# Parametri VAD — tunabili da config.py / .env
_CHUNK_SAMPLES = 512  # ~32ms a 16kHz — granularità fissa, non tunabile


class STTError(RuntimeError):
    pass


class STT:
    """
    Speech-to-text locale via Whisper.
    Il modello viene caricato in modo lazy alla prima chiamata.
    """

    _model = None  # singleton a livello di classe — caricato una sola volta

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            if not _WHISPER_OK:
                raise STTError(
                    "openai-whisper non installato.\n"
                    "  pip install openai-whisper"
                )
            logger.info("Caricamento modello Whisper '%s'...", config.WHISPER_MODEL_SIZE)
            cls._model = _whisper_lib.load_model(
                config.WHISPER_MODEL_SIZE,
                device="cpu",
            )
            logger.info("Modello Whisper pronto.")
        return cls._model

    # ------------------------------------------------------------------

    def record_and_transcribe(self, max_seconds: float = 8.0) -> str | None:
        """Registra dal microfono fino al silenzio (max max_seconds), poi trascrive."""
        audio = self._record(max_seconds)
        if audio is None or len(audio) == 0:
            return None
        return self.transcribe_array(audio)

    def transcribe_file(self, path: str | Path) -> str:
        """Trascrive un file .wav esistente — utile per test e debug."""
        model = self._get_model()
        result = model.transcribe(
            str(path),
            language="it",
            fp16=False,  # RPi non ha GPU, fp16 causerebbe errori o lentezza
        )
        return result["text"].strip()

    def transcribe_array(self, audio: np.ndarray) -> str:
        """Trascrive un array numpy float32 a 16kHz."""
        model = self._get_model()
        result = model.transcribe(
            audio,
            language="it",
            fp16=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            initial_prompt=(
                "Assistente vocale Nico. Comandi: guardami, mi vedi, cosa vedi, "
                "descrivi, analizza, che ore sono, come va, da quanto sono alla scrivania, "
                "quanto ho lavorato, quante volte mi sono alzato, "
                "ricordami di, imposta sveglia alle, svegliami alle, "
                "segna nota, ricorda che, annota che, aggiungi nota, "
                "che idee avevo su, cosa devo comprare, lista della spesa, "
                "riassumi la mia settimana, fammi piano studio per domani."
            ),
        )
        text = result["text"].strip()
        return "" if _is_hallucination(text) else text

    # ------------------------------------------------------------------
    # registrazione con VAD
    # ------------------------------------------------------------------

    def _record(self, max_seconds: float) -> np.ndarray | None:
        if not _SD_OK:
            raise STTError("sounddevice non installato: pip install sounddevice")

        sr             = config.AUDIO_SAMPLE_RATE
        rms_threshold  = config.STT_SILENCE_RMS_THRESHOLD
        min_chunks     = int(config.STT_MIN_SPEECH_S * sr / _CHUNK_SAMPLES)
        max_chunks     = int(max_seconds * sr / _CHUNK_SAMPLES)
        silence_needed = int(config.STT_SILENCE_DURATION_S * sr / _CHUNK_SAMPLES)

        chunks: list[np.ndarray] = []
        silent_run = 0

        try:
            with sd.InputStream(
                samplerate=sr,
                channels=1,
                dtype="int16",
                blocksize=_CHUNK_SAMPLES,
            ) as stream:
                for i in range(max_chunks):
                    chunk, _ = stream.read(_CHUNK_SAMPLES)
                    flat = chunk.flatten()
                    chunks.append(flat)

                    rms = float(np.sqrt(np.mean(flat.astype(np.float32) ** 2)))
                    is_silent = rms < rms_threshold

                    if i >= min_chunks:
                        silent_run = silent_run + 1 if is_silent else 0
                        if silent_run >= silence_needed:
                            logger.debug("Silenzio rilevato dopo %d chunk.", i + 1)
                            break

        except sd.PortAudioError as exc:
            raise STTError(f"Errore microfono: {exc}") from exc

        if not chunks:
            return None

        # converte int16 → float32 normalizzato per Whisper
        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        return audio


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print(f"sounddevice disponibile: {_SD_OK}")
    print(f"whisper disponibile:     {_WHISPER_OK}")

    # ------------------------------------------------------------------
    # 1. test VAD logic (senza hardware)
    # ------------------------------------------------------------------
    print("\n=== 1. logica VAD ===")
    sr            = config.AUDIO_SAMPLE_RATE
    rms_threshold = config.STT_SILENCE_RMS_THRESHOLD
    min_c         = int(config.STT_MIN_SPEECH_S * sr / _CHUNK_SAMPLES)
    sil_c         = int(config.STT_SILENCE_DURATION_S * sr / _CHUNK_SAMPLES)
    print(f"  chunk={_CHUNK_SAMPLES} samples ({_CHUNK_SAMPLES/sr*1000:.0f}ms)")
    print(f"  min speech chunks:    {min_c}  ({config.STT_MIN_SPEECH_S}s)")
    print(f"  silence chunks needed:{sil_c}  ({config.STT_SILENCE_DURATION_S}s)")
    print(f"  silence RMS threshold:{rms_threshold}")

    # simula array di audio: 2s parlato + 1.5s silenzio
    speech   = (np.random.randn(int(2.0 * sr)) * 3000).astype(np.float32)
    silence  = (np.random.randn(int(2.0 * sr)) * 50).astype(np.float32)
    combined = np.concatenate([speech, silence])

    # verifica che il silenzio venga rilevato
    chunks = [combined[i:i+_CHUNK_SAMPLES] for i in range(0, len(combined), _CHUNK_SAMPLES)]
    silent_run = 0
    stop_at = None
    for i, c in enumerate(chunks):
        rms = float(np.sqrt(np.mean(c ** 2)))
        if i >= min_c:
            silent_run = silent_run + 1 if rms < rms_threshold else 0
            if silent_run >= sil_c:
                stop_at = i
                break
    print(f"  VAD stop a chunk {stop_at} (~{stop_at * _CHUNK_SAMPLES / sr:.1f}s)  {'OK' if stop_at is not None else 'FAIL'}")

    # ------------------------------------------------------------------
    # 2. float32 conversion
    # ------------------------------------------------------------------
    print("\n=== 2. conversione int16 → float32 ===")
    int16_arr = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    float_arr = int16_arr.astype(np.float32) / 32768.0
    assert float_arr.max() <= 1.0 and float_arr.min() >= -1.0
    print(f"  range: [{float_arr.min():.3f}, {float_arr.max():.3f}]  OK")

    # ------------------------------------------------------------------
    # 3. trascrizione file esistente (se whisper installato + test.wav)
    # ------------------------------------------------------------------
    test_wav = Path("/home/federico/Desktop/nico/test.wav")
    if _WHISPER_OK and test_wav.exists():
        print(f"\n=== 3. trascrizione {test_wav.name} ===")
        stt = STT()
        t0 = time.time()
        text = stt.transcribe_file(test_wav)
        elapsed = time.time() - t0
        print(f"  Risultato: '{text}'")
        print(f"  Tempo:     {elapsed:.1f}s")
    elif not _WHISPER_OK:
        print("\n=== 3. trascrizione — SKIP (whisper non installato) ===")
        print("  pip install openai-whisper")
    else:
        print(f"\n=== 3. trascrizione — SKIP ({test_wav} non trovato) ===")

    print("\nTest STT completati.")
