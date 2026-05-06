#!/bin/bash
set -e

echo "=== Nico — setup dipendenze ==="

# --- picamera2 (pacchetto di sistema su RPi OS, deve esistere prima del venv) ---
if ! python3 -c "import picamera2" 2>/dev/null; then
    echo "[0/4] Installo python3-picamera2 via apt..."
    sudo apt-get install -y python3-picamera2
else
    echo "[0/4] picamera2 già presente nel sistema."
fi

# --- venv con accesso ai pacchetti di sistema (necessario per picamera2) ---
if [ ! -d "venv" ]; then
    echo "      Creo virtual environment (--system-site-packages)..."
    python3 -m venv --system-site-packages venv
fi
source venv/bin/activate

# --- Python ---
echo "[1/4] Installazione pacchetti Python..."
pip install -r requirements.txt

# --- openWakeWord: scarica i modelli built-in ---
echo "[2/4] Download modelli openWakeWord..."
python3 -c "
import openwakeword
openwakeword.utils.download_models()
print('Modelli openWakeWord scaricati.')
"

# --- ollama ---
echo "[3/4] Installazione ollama..."
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "      ollama installato."
else
    echo "      ollama già presente: $(ollama --version)"
fi

# --- moondream2 ---
echo "[4/4] Download modello moondream (Vision Tier 1)..."
echo "      Dimensione: ~1.7GB, potrebbe volerci qualche minuto..."
ollama pull moondream
echo "      moondream pronto."

echo ""
echo "=== Setup completato ==="
echo "Prossimo passo: copia .env e verifica le path di Piper"
echo "Test DB:     PYTHONPATH=. python3 memory/db.py"
echo "Test camera: PYTHONPATH=. python3 vision/camera.py"
