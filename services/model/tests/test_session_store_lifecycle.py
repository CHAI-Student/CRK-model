import logging

from model_service.session import SessionData, SessionStore


def _session(session_id: str, *, status: str, stage: str) -> SessionData:
    return SessionData(
        session_id=session_id,
        zone=5,
        status=status,
        processing_stage=stage,
    )


def test_session_store_normal_lifecycle_update_does_not_warn(caplog):
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    caplog.set_level(logging.WARNING, logger="model_service.session.session_store")

    store.save("session-1", _session("session-1", status="processing", stage="queued"))
    store.save("session-1", _session("session-1", status="complete", stage="complete"))

    assert "Session overwritten" not in caplog.text


def test_session_store_completed_session_replacement_still_warns(caplog):
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    caplog.set_level(logging.WARNING, logger="model_service.session.session_store")

    store.save("session-1", _session("session-1", status="complete", stage="complete"))
    caplog.clear()

    store.save("session-1", _session("session-1", status="complete", stage="complete"))

    assert "Session overwritten: session-1" in caplog.text
