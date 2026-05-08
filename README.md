# N.I.C.O — Neural Intelligence Control Operations

Assistente AI personale vocale che gira su **Raspberry Pi 5**.  
Ascolta la wake word "Nico", risponde a voce tramite GPT-4o, monitora la stanza con la camera e ricorda promemoria.

---

## Funzionalità

- **Wake word** — rileva "Nico" offline (Porcupine o openWakeWord)
- **Speech-to-text** — trascrizione locale con Whisper
- **Risposta vocale** — GPT-4o via OpenRouter + sintesi con Piper TTS
- **Visione** — cattura frame con la camera e li analizza con GPT-4o Vision ("guardami", "cosa vedi?")
- **Monitoraggio passivo** — moondream (Ollama) analizza la stanza ogni ~60s e traccia routine
- **Promemoria e sveglie** — imposta con la voce o dal web, Nico parla all'orario stabilito
- **Note vocali** — "segna nota: ...", "ricorda che ..."
- **Dashboard web** — interfaccia su `http://raspberrypi:5000` con cronometro, timer, chat testuale, token usati

---

## Requisiti

| Componente | Versione |
|---|---|
| Raspberry Pi | 5 (testato), 4 dovrebbe funzionare |
| Raspberry Pi OS | Bookworm 64-bit |
| Python | 3.11+ |
| RAM | 4 GB minimo, 8 GB consigliato |

> **Windows / macOS: non supportato.**  
> Il progetto dipende da `picamera2` (solo Linux/RPi), `signal.pause()` (solo Unix) e GPIO.  
> Puoi far girare la dashboard web su qualsiasi OS ma non il sistema vocale completo.

---

## Installazione

### 1. Clona il repo

```bash
git clone https://github.com/TUO_USERNAME/nico-camera.git
cd nico-camera
```

### 2. Configura le variabili d'ambiente

```bash
cp .env.example .env
nano .env   # inserisci le tue chiavi API
```

Chiavi necessarie:
- `OPENROUTER_API_KEY` — da [openrouter.ai/keys](https://openrouter.ai/keys) (gratuito con crediti iniziali)
- `PORCUPINE_ACCESS_KEY` — da [console.picovoice.ai](https://console.picovoice.ai) *(opzionale, fallback gratuito disponibile)*

### 3. Installa le dipendenze

```bash
chmod +x install.sh
./install.sh
```

Lo script installa automaticamente:
- `picamera2` via apt
- pacchetti Python nel venv
- modelli openWakeWord
- ollama + moondream

### 4. Configura Piper TTS

Scarica il binario e la voce italiana:

```bash
# nella cartella padre del progetto
mkdir -p ../nico/piper/voices
# scarica piper da https://github.com/rhasspy/piper/releases
# scarica la voce da https://huggingface.co/rhasspy/piper-voices/tree/main/it/it_IT/paola/medium
```

### 5. Avvia

```bash
source venv/bin/activate
python main.py
```

La dashboard web è disponibile su `http://localhost:5000`.

---

## Struttura del progetto

```
nico-camera/
├── main.py                  # entry point, gestione segnali, loop principale
├── config.py                # tutte le configurazioni da .env
├── audio/
│   ├── stt.py               # Speech-to-text (Whisper)
│   ├── tts.py               # Text-to-speech (Piper)
│   └── wake_word.py         # rilevamento wake word (Porcupine / openWakeWord)
├── brain/
│   ├── gpt.py               # chiamate OpenRouter (chat, stream, vision) + token tracking
│   ├── router.py            # classificazione richieste (A/B/C) + intent detection
│   ├── context_builder.py   # assembla il contesto da DB per GPT
│   └── response_cache.py    # cache risposte frequenti
├── memory/
│   ├── db.py                # SQLite: eventi, sessioni, note, promemoria, preferenze
│   ├── aggregator.py        # raggruppa eventi in sessioni
│   ├── learner.py           # evoluzione adattiva delle domande passive
│   └── logger.py            # log eventi dal passive loop
├── monitoring/
│   ├── passive_loop.py      # loop di monitoraggio con moondream
│   └── reminder_scheduler.py # scheduler promemoria (parla all'orario stabilito)
├── vision/
│   └── camera.py            # cattura frame con picamera2
├── privacy/
│   └── controller.py        # LED attività + kill switch hardware (GPIO)
├── web/
│   ├── dashboard.py         # Flask API + avvio server
│   ├── templates/index.html # interfaccia web
│   └── static/app.js        # JavaScript dashboard
├── assets/                  # modelli wake word (.ppn, .joblib)
├── .env.example             # template variabili d'ambiente
├── requirements.txt
└── install.sh               # script setup automatico
```

---

## Dashboard web

Apri `http://raspberrypi:5000` da qualsiasi dispositivo sulla stessa rete.

| Pannello | Funzione |
|---|---|
| Stato | attività rilevata, presenza in stanza, scrivania |
| Sistema | eventi totali, uptime, cache, **token sessione/totali** |
| Promemoria | lista, segna fatto, elimina |
| Note | ricerca note vocali |
| Sessioni | storico sessioni di lavoro |
| Cronometro | start/stop/reset |
| Timer | conto alla rovescia con beep audio |
| Aggiungi promemoria | form data+ora, tipo promemoria o sveglia |
| Chat testuale | scrivi comandi come se parlassi a Nico |

---

## Comandi vocali principali

```
"Nico"                                  → attiva l'ascolto
"Ricordami di X alle 10"                → imposta promemoria
"Svegliami alle 7"                      → sveglia (parla 4 volte)
"Segna nota: ..."                       → salva nota
"Guardami" / "Cosa vedi?"              → analisi visiva live
"Da quanto sono alla scrivania?"        → dati dal monitoraggio passivo
"Fammi un piano studio per domani"      → piano personalizzato
"Riassumi la mia settimana"             → riepilogo sessioni
```

---

## Licenza

MIT
