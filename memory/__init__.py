from .db import (
    init_db,
    log_event,
    get_recent_events,
    count_events,
    start_session,
    end_session,
    get_recent_sessions,
    upsert_pattern,
    get_patterns,
    set_preference,
    get_preference,
    get_all_preferences,
)
from .logger import write_observation
from .aggregator import run_once as aggregate, start_background as start_aggregator
from .learner import get_active_questions, maybe_evolve, pool_status
