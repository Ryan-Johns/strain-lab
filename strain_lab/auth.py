"""Simple shared-password sign-in for the hosted app (on when STRAIN_LAB_PASSWORD is set)."""
import hmac
import time

import streamlit as st


def require_login():
    """Show the sign-in form until the right password is entered; ask everyone for their name."""
    from .cloud import setting
    password = setting("STRAIN_LAB_PASSWORD")
    if not password:
        st.sidebar.text_input("Your name", key="user_name", help="Shown as 'added by' on files and runs you save.")
        return
    if st.session_state.get("signed_in"):
        st.sidebar.caption(f"Signed in as **{st.session_state.get('user_name') or 'guest'}**")
        if st.sidebar.button("Sign out", key="sign_out"):
            st.session_state.clear()
            st.rerun()
        return
    st.title("Strain Gauge Filtering Lab")
    with st.form("sign_in"):
        name = st.text_input("Your name", help="Shown as 'added by' on files and runs you save.")
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in", type="primary")
    if ok:
        if hmac.compare_digest(pw.encode(), password.encode()):
            st.session_state["signed_in"] = True
            st.session_state["user_name"] = name.strip()
            st.rerun()
        time.sleep(1.0)                  # slows down password guessing
        st.error("Wrong password.")
    st.stop()
