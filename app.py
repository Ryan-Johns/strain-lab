"""Strain Gauge Filtering Lab — run with:  python -m streamlit run app.py"""
import streamlit as st

from strain_lab import applock
from strain_lab.auth import require_login
from strain_lab.config import APP_TITLE
from strain_lab.pages import PAGES
from strain_lab.ui import get_store, init_state, preset_sidebar, sync_status, PLOT_DETAIL

st.set_page_config(page_title=APP_TITLE, layout="wide")
require_login()
init_state()
if applock.enabled():
    applock.gate(get_store())          # someone else has locked the app -> 'please wait' screen

st.sidebar.title(APP_TITLE)
page = st.sidebar.radio("Go to page", list(PAGES), key="nav_page")
st.sidebar.markdown("---")
sync_status()
if applock.enabled():
    applock.sidebar(get_store())
st.sidebar.write(f"Session library: {len(st.session_state.library)} trace(s)")
if st.session_state.library and st.sidebar.button("Clear library", key="clear_library"):
    st.session_state.library = []
    st.rerun()
st.sidebar.caption("The session library holds traces you filtered from uploads; it is cleared when you close the app. "
                   "Data-store files, runs and presets are permanent and shared.")
st.sidebar.selectbox("Plot detail", list(PLOT_DETAIL), index=1, key="plot_detail",
                     help="How many points per line are sent to the browser. Peaks and valleys are always kept; "
                          "fewer points make every click faster.")
preset_sidebar()

PAGES[page]()
