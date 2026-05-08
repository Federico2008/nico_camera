import logging
import threading
from datetime import datetime

from flask import Flask, jsonify, render_template, request

import memory.db as db
from brain.response_cache import cache_stats

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

_start_time = datetime.now()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    from brain.gpt import get_token_stats
    events = db.get_recent_events(1)
    latest = events[0] if events else None
    uptime_s = int((datetime.now() - _start_time).total_seconds())
    return jsonify({
        "activity":        latest["activity_label"] if latest else None,
        "in_room":         bool(latest["in_room"])  if latest and latest["in_room"]  is not None else None,
        "at_desk":         bool(latest["at_desk"])  if latest and latest["at_desk"]  is not None else None,
        "last_update":     latest["timestamp"][:16].replace("T", " ") if latest else None,
        "avg_inference_ms": db.avg_inference_time_ms(),
        "total_events":    db.count_events(),
        "uptime_s":        uptime_s,
        "cache":           cache_stats(),
        "tokens":          get_token_stats(),
    })


@app.route("/api/reminders")
def api_reminders():
    return jsonify(db.get_upcoming_reminders(limit=15))


@app.route("/api/reminders/fired")
def api_reminders_fired():
    from monitoring.reminder_scheduler import get_recently_fired
    return jsonify(get_recently_fired(since_s=120))


@app.route("/api/reminders", methods=["POST"])
def api_reminder_add():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    trigger_str = (data.get("trigger_time") or "").strip()
    category = (data.get("category") or "promemoria").strip()
    if not text or not trigger_str:
        return jsonify({"error": "text e trigger_time richiesti"}), 400
    try:
        trigger = datetime.fromisoformat(trigger_str)
    except ValueError:
        try:
            trigger = datetime.fromisoformat(trigger_str + ":00")
        except ValueError:
            return jsonify({"error": "formato trigger_time non valido"}), 400
    rid = db.add_reminder(text=text, trigger_time=trigger, category=category)
    return jsonify({"ok": True, "id": rid})


@app.route("/api/reminders/<int:rid>/done", methods=["POST"])
def api_reminder_done(rid: int):
    db.mark_reminder_done(rid)
    return jsonify({"ok": True})


@app.route("/api/reminders/<int:rid>", methods=["DELETE"])
def api_reminder_delete(rid: int):
    db.delete_reminder(rid)
    return jsonify({"ok": True})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    from brain.response_cache import cache_response, get_cached, try_direct_answer
    from brain.router import IntentType, RequestClass, classify, detect_intent
    from brain.context_builder import build as build_context
    from brain.gpt import chat

    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "testo vuoto"}), 400

    try:
        direct = try_direct_answer(text)
        if direct:
            return jsonify({"response": direct})

        cached = get_cached(text)
        if cached:
            return jsonify({"response": cached})

        intent = detect_intent(text)
        if intent.intent != IntentType.NONE:
            if intent.response is not None:
                return jsonify({"response": intent.response})
            if intent.gpt_prompt is not None:
                reply = chat(intent.gpt_prompt, intent.gpt_context or "")
                return jsonify({"response": reply})

        result = classify(text)
        if result.cls == RequestClass.C:
            from vision.camera import Camera, CameraError
            from brain.gpt import chat_with_vision
            try:
                with Camera() as cam:
                    frames_b64 = cam.capture_frames_base64(n=3, interval_s=1.5)
            except CameraError as exc:
                logger.error("Errore camera (chat): %s", exc)
                return jsonify({"response": "Non riesco ad accedere alla camera in questo momento."})
            context = build_context(result.cls, user_text=text)
            reply = chat_with_vision(text, frames_b64, context)
            return jsonify({"response": reply})

        context = build_context(result.cls, user_text=text)
        reply = chat(text, context)
        cache_response(text, reply)
        return jsonify({"response": reply})
    except Exception as exc:
        logger.error("Errore chat testuale: %s", exc)
        return jsonify({"error": "Errore interno del server"}), 500


@app.route("/api/notes")
def api_notes():
    return jsonify(db.get_notes(
        category=request.args.get("category"),
        keyword=request.args.get("q"),
        limit=25,
    ))


@app.route("/api/sessions")
def api_sessions():
    return jsonify(db.get_recent_sessions(10))


def start(host: str = "0.0.0.0", port: int = 5000) -> None:
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
        name="web-dashboard",
    )
    t.start()
    logger.info("Dashboard avviata su http://%s:%d", host, port)
