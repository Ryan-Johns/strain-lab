"""'Lock the app' for the online version: one person uses it (e.g. for batch processing) while others wait.

The lock lives in the shared catalog, so everyone sees it, it survives restarts, and it runs out by itself
after the time chosen, so a forgotten lock can't shut people out for long. It is a courtesy between
collaborators, not security: anyone signed in can unlock it from the lock screen if they need to.
"""
import time
import uuid

import streamlit as st

DURATIONS = {"30 minutes": 30, "1 hour": 60, "2 hours": 120, "4 hours": 240, "8 hours": 480}


def _remaining(lock) -> str:
    m = max(0, int(round((float(lock["until"]) - time.time()) / 60)))
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def _who(lock) -> str:
    return lock.get("by") or "someone"


def _my_key() -> str:
    return st.session_state.setdefault("lock_key", uuid.uuid4().hex)


def enabled() -> bool:
    from .ui import hosted
    from .cloud import setting
    return hosted() or bool(setting("STRAIN_LAB_PASSWORD"))


def gate(store):
    """Stop here with a 'locked' screen if someone else holds the lock."""
    if not enabled():
        return
    lock = store.current_lock()
    if not lock or lock.get("key") == _my_key():
        return
    me = (st.session_state.get("user_name") or "").strip()
    st.title("🔒 The app is in use")
    st.info(f"**{_who(lock)}** locked the app" + (f" for: *{lock['note']}*" if lock.get("note") else "") +
            f". It unlocks automatically in **{_remaining(lock)}**, or sooner when they unlock it.")
    if st.button("Check again", type="primary", key="lock_check"):
        st.rerun()
    if me and me.lower() == _who(lock).lower():
        st.caption("Is this your lock (for example, you refreshed the page)?")
        if st.button("Continue — this is my lock", key="lock_adopt"):
            st.session_state["lock_key"] = lock["key"]
            st.rerun()
    with st.expander("Unlock anyway"):
        st.write(f"Only do this if {_who(lock)} has finished or agrees. If they are in the middle of a batch run, "
                 "it carries on, but you would both be using the app's memory at once.")
        sure = st.checkbox(f"{_who(lock)} has finished or agrees", key="lock_force_ok")
        if st.button("Unlock", disabled=not sure, key="lock_force"):
            store.release_lock()
            st.rerun()
    st.stop()


def sidebar(store):
    """Lock / unlock controls in the sidebar."""
    if not enabled():
        return
    lock = store.current_lock()
    if lock and lock.get("key") == _my_key():
        st.sidebar.warning(f"🔒 You have the app locked ({_remaining(lock)} left). Others see a 'please wait' screen.")
        c1, c2 = st.sidebar.columns(2)
        if c1.button("Unlock", key="lock_release", type="primary"):
            store.release_lock(_my_key())
            st.rerun()
        if c2.button("+1 hour", key="lock_extend"):
            left = max(0.0, (float(lock["until"]) - time.time()) / 60)
            store.take_lock(lock.get("by", ""), _my_key(), left + 60, lock.get("note", ""))
            st.rerun()
        return
    with st.sidebar.expander("🔒 Lock the app"):
        st.caption("While locked, only you can use the app; anyone else sees a 'please wait' screen. "
                   "Handy for batch processing. It unlocks by itself after the time you choose.")
        dur = st.selectbox("For up to", list(DURATIONS), index=1, key="lock_duration")
        note = st.text_input("Reason (optional)", placeholder="batch filtering L3", key="lock_note")
        if st.button("Lock now", key="lock_take"):
            got = store.take_lock((st.session_state.get("user_name") or "").strip(), _my_key(),
                                  DURATIONS[dur], note.strip())
            if got.get("key") != _my_key():
                st.error(f"{_who(got)} locked it just before you.")
            else:
                st.rerun()
