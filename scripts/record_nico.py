#!/usr/bin/env python3
"""
Registra campioni audio per addestrare la wake word "Nico".

Uso:
    python scripts/record_nico.py

Produce:
    assets/nico_samples/positive/  — 40 clip di "Nico"
    assets/nico_samples/negative/  — 40 clip di altra voce/rumore

Ogni clip dura 1.5s a 16kHz mono int16.
"""

import sys
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
DURATION_S  = 1.5
SAMPLES     = int(SAMPLE_RATE * DURATION_S)
N_CLIPS     = 40

POSITIVE_DIR = Path(__file__).parent.parent / "assets" / "nico_samples" / "positive"
NEGATIVE_DIR = Path(__file__).parent.parent / "assets" / "nico_samples" / "negative"

NEGATIVE_PROMPTS = [
    "ciao",        "grazie",      "computer",    "aprire",
    "musica",      "volume",      "luce",        "finestra",
    "temperatura", "meteo",       "orario",      "aiuto",
    "casa",        "porta",       "telefono",    "messaggio",
    "mattina",     "sera",        "oggi",        "domani",
    "mangiare",    "dormire",     "andare",      "venire",
    "grande",      "piccolo",     "nuovo",       "vecchio",
    "rosso",       "verde",       "acqua",       "fuoco",
    "mare",        "montagna",    "sole",        "luna",
    "uno",         "due",         "tre",         "quattro",
]

try:
    import sounddevice as sd
except ImportError:
    print("ERRORE: sounddevice non installato. pip install sounddevice")
    sys.exit(1)


def _save_wav(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)       # 16-bit
        f.setframerate(SAMPLE_RATE)
        f.writeframes(audio.tobytes())


def _record_one(label: str, countdown: bool = True) -> np.ndarray:
    if countdown:
        for i in (3, 2, 1):
            print(f"\r  {i}...", end="", flush=True)
            time.sleep(0.7)
        print("\r  REC  ", end="", flush=True)
    audio = sd.rec(SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    print("\r  OK   ")
    return audio.flatten()


def _record_set(kind: str, prompts: list[str], out_dir: Path, start_idx: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"FASE: {kind} ({len(prompts)} clip da {DURATION_S}s)")
    print("Premi INVIO per iniziare ogni registrazione.")
    print("="*50)

    for i, prompt in enumerate(prompts):
        idx = start_idx + i
        out_path = out_dir / f"{idx:03d}.wav"
        if out_path.exists():
            print(f"  [{idx+1}/{len(prompts)}] già presente — skip")
            continue

        print(f"\n  [{idx+1}/{len(prompts)}]  Dì: \033[1m{prompt}\033[0m")
        input("  Premi INVIO quando sei pronto...")
        audio = _record_one(prompt)

        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        if rms < 200:
            print("  Troppo silenzioso — riprova.")
            continue

        _save_wav(out_path, audio)
        print(f"  Salvato → {out_path.name}  (RMS={rms:.0f})")


def main():
    print("=== Registrazione campioni wake word 'Nico' ===")
    print(f"Microfono: {sd.query_devices(kind='input')['name']}")
    print(f"Sample rate: {SAMPLE_RATE} Hz | Durata clip: {DURATION_S}s")

    # --- Fase 1: positivi ("Nico") ---
    existing_pos = list(POSITIVE_DIR.glob("*.wav")) if POSITIVE_DIR.exists() else []
    start_pos = len(existing_pos)
    remaining_pos = N_CLIPS - start_pos
    if remaining_pos > 0:
        prompts_pos = ["NICO"] * remaining_pos
        _record_set("POSITIVI — dì 'Nico'", prompts_pos, POSITIVE_DIR, start_idx=start_pos)
    else:
        print(f"\nPositivi già completi ({N_CLIPS} clip).")

    # --- Fase 2: negativi (altri suoni) ---
    existing_neg = list(NEGATIVE_DIR.glob("*.wav")) if NEGATIVE_DIR.exists() else []
    start_neg = len(existing_neg)
    remaining_neg = N_CLIPS - start_neg
    if remaining_neg > 0:
        prompts_neg = NEGATIVE_PROMPTS[start_neg:start_neg + remaining_neg]
        _record_set("NEGATIVI — dì la parola mostrata", prompts_neg, NEGATIVE_DIR, start_idx=start_neg)
    else:
        print(f"Negativi già completi ({N_CLIPS} clip).")

    pos_count = len(list(POSITIVE_DIR.glob("*.wav")))
    neg_count = len(list(NEGATIVE_DIR.glob("*.wav")))
    print(f"\n=== Registrazione completata ===")
    print(f"  Positivi: {pos_count}  |  Negativi: {neg_count}")
    print(f"\nProssimo passo:")
    print(f"  python scripts/train_nico.py")


if __name__ == "__main__":
    main()
