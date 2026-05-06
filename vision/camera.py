import base64
import io
import time

from PIL import Image

import config

try:
    from picamera2 import Picamera2
    _PICAMERA2_OK = True
except ImportError:
    _PICAMERA2_OK = False


class CameraError(RuntimeError):
    pass


class Camera:
    """
    Context manager per Raspberry Pi Camera Module 3 (picamera2 / libcamera).

    Uso tipico:
        with Camera() as cam:
            img = cam.capture_frame()          # PIL Image
            b64 = cam.capture_frame_base64()   # JPEG base64 per ollama
    """

    # Attesa post-start per stabilizzazione esposizione e autofocus.
    # Camera Module 3 con autofocus: 1.5s è il minimo sicuro.
    _WARMUP_S = 1.5

    def __init__(self):
        self._cam = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not _PICAMERA2_OK:
            raise CameraError(
                "picamera2 non trovato.\n"
                "  Installa con: sudo apt install python3-picamera2\n"
                "  oppure aggiungi al venv: pip install picamera2"
            )
        try:
            self._cam = Picamera2()
            cfg = self._cam.create_still_configuration(
                main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"}
            )
            self._cam.configure(cfg)
            self._cam.start()
            time.sleep(self._WARMUP_S)
        except Exception as exc:
            self._cam = None
            raise CameraError(f"Impossibile avviare la camera: {exc}") from exc

    def stop(self) -> None:
        if self._cam is not None:
            try:
                self._cam.stop()
                self._cam.close()
            except Exception:
                pass
            self._cam = None

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # capture
    # ------------------------------------------------------------------

    def _assert_running(self) -> None:
        if self._cam is None:
            raise CameraError("Camera non avviata. Usa Camera() come context manager.")

    def capture_frame(self) -> Image.Image:
        """Cattura un singolo frame come PIL Image (RGB)."""
        self._assert_running()
        array = self._cam.capture_array()
        return Image.fromarray(array)

    def capture_frame_base64(self, quality: int = 85) -> str:
        """Cattura un frame e lo restituisce come stringa JPEG base64.

        Formato diretto per l'API ollama: {"images": [<base64>]}
        """
        img = self.capture_frame()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()

    def capture_frames(self, n: int = 3, interval_s: float = 1.5) -> list[Image.Image]:
        """Cattura n frame a distanza interval_s secondi l'uno dall'altro.

        Usato dalla modalità tier2 "guardami": 3-5 frame in ~5 secondi.
        """
        self._assert_running()
        frames: list[Image.Image] = []
        for i in range(n):
            frames.append(self.capture_frame())
            if i < n - 1:
                time.sleep(interval_s)
        return frames

    def capture_frames_base64(self, n: int = 3, interval_s: float = 1.5,
                               quality: int = 85) -> list[str]:
        """Versione base64 di capture_frames — pronta per GPT-4o Vision."""
        frames = self.capture_frames(n=n, interval_s=interval_s)
        result = []
        for img in frames:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            result.append(base64.b64encode(buf.getvalue()).decode())
        return result


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _PICAMERA2_OK:
        print("ERRORE: picamera2 non trovato.")
        print("  sudo apt install python3-picamera2")
        raise SystemExit(1)

    print(f"Risoluzione configurata: {config.CAMERA_RESOLUTION}")
    print("Avvio camera...")

    with Camera() as cam:
        print("Camera attiva. Cattura frame singolo...")
        img = cam.capture_frame()
        out_path = "/tmp/nico_test.jpg"
        img.save(out_path, quality=85)
        w, h = img.size
        print(f"  Dimensione: {w}x{h} px")
        print(f"  Salvato in: {out_path}")

        print("Cattura base64...")
        b64 = cam.capture_frame_base64()
        print(f"  Base64: {len(b64)} caratteri (~{len(b64) * 3 // 4 // 1024} KB)")

        print("Cattura 3 frame (test tier2)...")
        t0 = time.time()
        frames = cam.capture_frames(n=3, interval_s=1.0)
        elapsed = time.time() - t0
        print(f"  {len(frames)} frame catturati in {elapsed:.1f}s")

    print("\nTest completato. Apri /tmp/nico_test.jpg per verificare l'immagine.")
