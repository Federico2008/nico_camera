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
    })


@app.route("/api/reminders")
def api_reminders():
    return jsonify(db.get_upcoming_reminders(limit=15))


@app.route("/api/reminders/<int:rid>/done", methods=["POST"])
def api_reminder_done(rid: int):
    db.mark_reminder_done(rid)
    return jsonify({"ok": True})


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
    import os
    os.environ["WERKZEUG_RUN_MAIN"] = "true"
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
        name="web-dashboard",
    )
    t.start()
    logger.info("Dashboard avviata su http://%s:%d", host, port)
