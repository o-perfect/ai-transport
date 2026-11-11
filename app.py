"""Shipping Route Planner + Corridor Risk Agent.

Run with:  streamlit run app.py
Pages share st.session_state; the keys they exchange are in shared_state.py.
"""
import streamlit as st

from app_chat import chat_page
from core.chokepoints import ChokepointRegistry
from shared_state import SCENARIO, scenario_names

st.set_page_config(page_title="Shipping Route Planner", page_icon="🚢", layout="wide")

# Chokepoints closed on the map are the chat's scenario; refreshed every run so
# the chat page sees toggles even when opened first.
st.session_state[SCENARIO] = scenario_names(ChokepointRegistry().closed())

st.navigation([
    st.Page("app_planner.py", title="Route planner", icon="🗺️", default=True),
    st.Page(chat_page, title="Corridor risk agent", icon="💬", url_path="agent"),
]).run()
