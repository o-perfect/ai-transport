"""Corridor Risk Agent with a live map: Streamlit UI.

Run with:  streamlit run app.py
Input is natural language only: you ask the agent in the chat (right), and the
routes and chokepoint scenario from its latest route comparison are drawn on
the map (left), with route cards below it. They meet in st.session_state (keys
in shared_state.py). Presentation lives in ui/ (styles.css, components.py, map.py).
"""
import streamlit as st
from streamlit_folium import st_folium

from app_chat import chat_panel
from shared_state import AGENT_BUSY, AGENT_ROUTES
from ui import components as ui
from ui.map import route_map

MAP_HEIGHT = 640
CHAT_MESSAGES_HEIGHT = MAP_HEIGHT - 152  # chat card (title + messages + input) ends level with the map

st.set_page_config(page_title="Corridor Risk Agent", page_icon="🛰️", layout="wide")
ui.inject_css()
ui.header(busy=st.session_state.get(AGENT_BUSY, False))

left, right = st.columns([3, 2], gap="medium")
with right:
    run_turn = chat_panel(height=CHAT_MESSAGES_HEIGHT)
with left:
    routes = st.session_state.get(AGENT_ROUTES, {})
    features, scenario = routes.get("features", []), routes.get("scenario", {})
    with st.container(key="map_card"):
        st_folium(route_map(routes), height=MAP_HEIGHT, use_container_width=True, returned_objects=[])
    if features:
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        with c1:
            ui.results_title(features)
        if c2.button("Clear map", icon=":material/layers_clear:", width="stretch",
                     disabled=st.session_state.get(AGENT_BUSY, False)):
            del st.session_state[AGENT_ROUTES]
            st.rerun()
        ui.kpi_strip(features, scenario)
        ui.route_cards(features, scenario)
        with st.expander("Table view"):
            st.dataframe(ui.route_table(features), hide_index=True, width="stretch")
if run_turn:  # after the map is drawn, so it stays visible while the agent works
    run_turn()
