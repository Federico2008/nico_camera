#!/usr/bin/env python3
"""
Addestra un modello custom per la wake word "Nico".

Funzionamento:
  1. Genera campioni sintetici con Piper (variazioni di "Nico")
  2. Carica campioni reali registrati con record_nico.py
  3. Estrae embedding 96-dim con openWakeWord AudioFeatures
  4. Allena LogisticRegression (sklearn)
  5. Salva il modello in assets/nico_model.joblib

Uso:
    python scripts/train_nico.py

Poi aggiungi a .env:
    WAKE_WORD_CUSTOM_MODEL=/home/federico/Desktop/nico_camera/assets/nico_model.joblib
    WAKE_WORD_MODEL=nico    (serve solo come nome per i log)
"""

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# path
# ---------------------------------------------------------------------------

BASE      = Path(__file__).parent.parent
POS_DIR   = BASE / "assets" / "nico_samples" / "positive"
NEG_DIR   = BASE / "assets" / "nico_samples" / "negative"
MODEL_OUT = BASE / "assets" / "nico_model.joblib"

PIPER_BIN   = BASE.parent / "nico" / "piper" / "piper"
PIPER_VOICE = BASE.parent / "nico" / "piper" / "voices" / "it_IT-paola-medium.onnx"

SAMPLE_RATE = 16000

# Variazioni testuali di "Nico" da passare a Piper per la sintesi
PIPER_POSITIVES = [
    "Nico",  "nico",  "NICO",
    "Nico!",  "Nico?",  "Nico.",
    "Ehi Nico",  "Ciao Nico",  "Ok Nico",
    "Nico ascoltami",  "Nico dimmi",
]

# Parole italiane per negativi sintetici (diverse da quelle registrate)
PIPER_NEGATIVES = [
    "sì", "no", "bene", "male", "forse",
    "adesso", "subito", "pronto", "vai", "stop",
    "accendi", "spegni", "aumenta", "diminuisci",
    "quanto", "dove", "quando", "perché", "come",
]


# ---------------------------------------------------------------------------
# audio helpers
# ---------------------------------------------------------------------------

def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "r") as f:
        data = f.readframes(f.getnframes())
    arr = np.frombuffer(data, dtype=np.int16)
    target = int(SAMPLE_RATE * 1.5)
    if len(arr) < target:
        arr = np.pad(arr, (0, target - len(arr)))
    else:
        arr = arr[:target]
    return arr


def _synth_piper(text: str, piper_bin: Path, voice: Path) -> np.ndarray | None:
    """Genera audio con Piper, ritorna array int16 da 1.5s."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [str(piper_bin), "--model", str(voice), "--output_file", tmp_path],
            input=text.encode(),
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return _load_wav(Path(tmp_path))
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# embedding extraction
# ---------------------------------------------------------------------------

def _extract_embeddings(clips: list[np.ndarray], af) -> np.ndarray:
    """clips: lista di array int16 da 1.5s → matrice (N, 96)."""
    if not clips:
        return np.zeros((0, 96))
    arr = np.stack(clips)                       # (N, 24000)
    emb = af.embed_clips(arr, batch_size=32)    # (N, frames, 96)
    return emb.mean(axis=1)                     # (N, 96) — media sui frame


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=== Training modello wake word 'Nico' ===\n")

    # --- 1. verifica dipendenze ---
    try:
        from openwakeword.utils import AudioFeatures
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        import joblib
    except ImportError as e:
        print(f"ERRORE dipendenza: {e}")
        print("pip install openwakeword scikit-learn joblib")
        sys.exit(1)

    # --- 2. carica campioni reali ---
    print("[1/5] Caricamento campioni reali...")
    pos_real = [_load_wav(p) for p in sorted(POS_DIR.glob("*.wav"))] if POS_DIR.exists() else []
    neg_real = [_load_wav(p) for p in sorted(NEG_DIR.glob("*.wav"))] if NEG_DIR.exists() else []
    print(f"  Positivi reali: {len(pos_real)}  |  Negativi reali: {len(neg_real)}")

    # --- 3. genera campioni sintetici con Piper ---
    print("[2/5] Generazione campioni sintetici con Piper...")
    pos_synth: list[np.ndarray] = []
    neg_synth: list[np.ndarray] = []

    if PIPER_BIN.exists():
        for text in PIPER_POSITIVES:
            clip = _synth_piper(text, PIPER_BIN, PIPER_VOICE)
            if clip is not None:
                pos_synth.append(clip)
        for text in PIPER_NEGATIVES:
            clip = _synth_piper(text, PIPER_BIN, PIPER_VOICE)
            if clip is not None:
                neg_synth.append(clip)
        print(f"  Positivi sintetici: {len(pos_synth)}  |  Negativi sintetici: {len(neg_synth)}")
    else:
        print(f"  Piper non trovato in {PIPER_BIN} — skip sintesi.")

    # aggiunge clip di silenzio come negativi extra
    for _ in range(10):
        neg_synth.append(np.zeros(int(SAMPLE_RATE * 1.5), dtype=np.int16))

    pos_all = pos_real + pos_synth
    neg_all = neg_real + neg_synth

    if len(pos_all) < 5:
        print(f"\nERRORE: troppo pochi campioni positivi ({len(pos_all)}).")
        print("Esegui prima: python scripts/record_nico.py")
        sys.exit(1)

    print(f"  Totale: {len(pos_all)} positivi, {len(neg_all)} negativi")

    # --- 4. estrai embedding ---
    print("[3/5] Estrazione embedding (AudioFeatures)...")
    af = AudioFeatures()
    X_pos = _extract_embeddings(pos_all, af)
    X_neg = _extract_embeddings(neg_all, af)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(pos_all) + [0] * len(neg_all))
    print(f"  Feature matrix: {X.shape}  |  Label distribution: pos={y.sum()} neg={(y==0).sum()}")

    # --- 5. allena classificatore ---
    print("[4/5] Allenamento LogisticRegression...")
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    scores = cross_val_score(clf, X, y, cv=min(5, len(pos_all)//2), scoring="f1")
    print(f"  F1 cross-validation: {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"  Accuracy training set: {train_acc:.3f}")

    # --- 6. salva ---
    print(f"[5/5] Salvataggio → {MODEL_OUT}")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(clf, MODEL_OUT)
    print(f"  Modello salvato ({MODEL_OUT.stat().st_size / 1024:.1f} KB)")

    print("\n=== Training completato ===")
    print("\nAggiungi a .env:")
    print(f"  WAKE_WORD_CUSTOM_MODEL={MODEL_OUT.absolute()}")
    print(f"  WAKE_WORD_MODEL=nico")
    print("\nTest rapido:")
    print("  PYTHONPATH=. python audio/wake_word.py")


if __name__ == "__main__":
    main()
