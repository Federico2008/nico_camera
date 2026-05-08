import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# --- Database ---
DB_PATH = BASE_DIR / "nico.db"

# --- API ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
BRAIN_MODEL = "openai/gpt-4o"
VISION_MODEL = "openai/gpt-4o"

# --- Audio ---
WHISPER_MODEL_SIZE = "base"

# Wake word — Porcupine se configurato, altrimenti openWakeWord (fallback locale)
# Setup Porcupine "Nico":
#   1. console.picovoice.ai → crea account gratuito
#   2. "AccessKey" → copia chiave → aggiungi a .env: PORCUPINE_ACCESS_KEY=...
#   3. "Wake Word" → addestra "Nico" → piattaforma "Linux (ARM64)" → scarica .ppn
#   4. Salva in assets/nico_linux.ppn → aggiungi a .env: PORCUPINE_MODEL_PATH=.../assets/nico_linux.ppn
PORCUPINE_ACCESS_KEY  = os.getenv("PORCUPINE_ACCESS_KEY", "")
PORCUPINE_MODEL_PATH  = os.getenv("PORCUPINE_MODEL_PATH", "")
WAKE_WORD_CUSTOM_MODEL = os.getenv("WAKE_WORD_CUSTOM_MODEL", "")  # path .joblib → "Nico"
WAKE_WORD_MODEL       = "alexa"   # usato dal fallback openWakeWord se nessun custom
WAKE_WORD_THRESHOLD   = 0.70
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 1280         # 80ms a 16kHz — finestra richiesta da openWakeWord

# --- TTS (Piper) ---
PIPER_BIN = Path(os.getenv("PIPER_BIN", str(BASE_DIR.parent / "nico" / "piper" / "piper")))
PIPER_VOICE = Path(os.getenv("PIPER_VOICE", str(BASE_DIR.parent / "nico" / "piper" / "voices" / "it_IT-paola-medium.onnx")))
PIPER_OUTPUT_FILE = Path("/tmp/nico_speech.wav")

# --- Weather ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
WEATHER_CITY        = os.getenv("WEATHER_CITY", "Milano")

# --- Morning briefing ---
BRIEFING_HOUR = int(os.getenv("BRIEFING_HOUR", "8"))

# --- Vision ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MOONDREAM_MODEL = "moondream"
CAMERA_RESOLUTION = (640, 480)

# Loop passivo: intervallo adattivo basato sul tempo di inferenza reale
# Ogni ciclo aspetta max(PASSIVE_LOOP_MIN_INTERVAL_S, inferenza_precedente + PASSIVE_LOOP_BUFFER_S)
PASSIVE_LOOP_MIN_INTERVAL_S = 60   # moondream su CPU ~2min → ogni minuto è realistico
PASSIVE_LOOP_BUFFER_S = 5

# --- STT (Whisper VAD) ---
STT_SILENCE_RMS_THRESHOLD = int(os.getenv("STT_SILENCE_RMS_THRESHOLD", "300"))  # int16 scale
STT_SILENCE_DURATION_S    = float(os.getenv("STT_SILENCE_DURATION_S", "2.0"))   # secondi silenzio per stop
STT_MIN_SPEECH_S          = float(os.getenv("STT_MIN_SPEECH_S", "0.5"))         # secondi minimi prima del VAD

# Domande fisse inviate a moondream ad ogni frame passivo
PASSIVE_QUESTIONS_FIXED = [
    "Is there a person in the room?",
    "Is the person sitting at a desk?",
    "Is the person using a computer?",
]

# --- Pattern learning ---
QUESTIONS_EVOLVE_AFTER = 50     # osservazioni prima di generare domande candidate
QUESTIONS_VALIDATE_ON = 20      # osservazioni per validare una domanda candidata
QUESTIONS_SUSPEND_STATIC = 200  # osservazioni prima di riproporre domande con risposta fissa

# --- Privacy ---
PRIVACY_LED_PIN = int(os.getenv("PRIVACY_LED_PIN", "17"))   # GPIO BCM per LED attività
PRIVACY_BTN_PIN = int(os.getenv("PRIVACY_BTN_PIN", "27"))   # GPIO BCM per kill switch

# --- Aggregator ---
AGGREGATOR_GAP_S        = 300   # gap > 5min tra eventi in_room=True → nuova sessione
AGGREGATOR_MIN_DURATION_S = 120 # sessioni < 2min ignorate (rumore)
AGGREGATOR_RUN_EVERY_S  = 3600  # job ogni ora

# --- Router vocale ---
CLASS_A_KEYWORDS = [
    "che ore", "orario", "timer", "meteo", "temperatura", "calcola", "quanto fa",
    "quanti", "data", "giorno", "settimana", "mese", "anno",
]
CLASS_C_KEYWORDS = [
    "guardami", "guarda", "cosa vedi", "mi vedi", "vedi", "cosa c'è", "descrivi", "analizza",
    "cosa sto", "come sono", "cosa ho", "che faccio", "come mi vedi", "cosa noti",
]
