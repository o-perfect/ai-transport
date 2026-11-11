"""Corridor Risk Agent with a live map: Streamlit UI.

Run with:  streamlit run app.py
Input is natural language only: you ask the agent in the chat (right), and the
routes and chokepoint scenario from its latest route comparison are drawn on
the map (left). They meet in st.session_state (keys in shared_state.py).
"""
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from agent.network import CHOKEPOINTS
from app_chat import chat_panel
from shared_state import AGENT_BUSY, AGENT_ROUTES

st.set_page_config(page_title="Corridor Risk Agent", page_icon="🚢", layout="wide")

ROUTE_COLORS = ["#1a73e8", "#9334e6", "#e52592", "#12b5cb"]  # baseline, alt 1, alt 2, alt 3
CLOSED_RED = "#d93025"
WAR_RISK_ORANGE = "#e8710a"
OPEN_GREY = "#7a7f87"


def money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    return f"${x:,.0f}"


def chokepoint_status(name, scenario):
    if name in scenario.get("blocked", []):
        return "Closed", CLOSED_RED
    if name in scenario.get("vessel_cannot_use", []):
        return "Too small for this vessel", CLOSED_RED
    if name in scenario.get("war_risk", []):
        return "Open, war risk", WAR_RISK_ORANGE
    return "Open", OPEN_GREY


def route_map(routes) -> folium.Map:
    features = routes.get("features", [])
    scenario = routes.get("scenario", {})
    m = folium.Map(location=[20, 30], zoom_start=2, tiles=None,
                   world_copy_jump=True, control_scale=True)
    folium.TileLayer(  # free, no API key; light grey keeps the routes readable
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Light grey", max_zoom=16).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.LayerControl(position="topright").add_to(m)

    view = []
    # alternatives first so the baseline is drawn on top
    for i, feat in reversed(list(enumerate(features))):
        props = feat["properties"]
        latlon = [(lat, lon) for lon, lat in feat["geometry"]["coordinates"]]
        view += latlon
        folium.PolyLine(latlon, color=ROUTE_COLORS[i % len(ROUTE_COLORS)],
                        weight=5 if props["role"] == "baseline" else 4, opacity=0.85,
                        dash_array="8 8" if props["role"] == "baseline" else None,
                        tooltip=f"{props['label']} · {props['distance_nm']:,} nm · "
                                f"{props['sea_days']} days · {money(props['total_cost_usd'])}").add_to(m)

    for name, (_, (lon, lat)) in CHOKEPOINTS.items():
        status, color = chokepoint_status(name, scenario)
        for shift in (-360, 0, 360):  # draw on every world copy
            folium.CircleMarker([lat, lon + shift], radius=7, color=color, fill=True,
                                fill_opacity=0.85, weight=2,
                                tooltip=f"<b>{name}</b><br>{status}").add_to(m)

    if features:
        base = features[0]
        start, end = base["geometry"]["coordinates"][0], base["geometry"]["coordinates"][-1]
        folium.Marker([start[1], start[0]], tooltip=f"From: {base['properties']['origin']}",
                      icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
        folium.Marker([end[1], end[0]], tooltip=f"To: {base['properties']['destination']}",
                      icon=folium.Icon(color="red", icon="flag", prefix="fa")).add_to(m)
        m.fit_bounds([[min(c[0] for c in view), min(c[1] for c in view)],
                      [max(c[0] for c in view), max(c[1] for c in view)]], padding=(30, 30))
    return m


def route_table(features) -> pd.DataFrame:
    base = features[0]["properties"]
    return pd.DataFrame([{
        "Route": p["label"],
        "Distance (nm)": f"{p['distance_nm']:,}",
        "Sea days": p["sea_days"],
        "Total cost": money(p["total_cost_usd"]),
        "Extra vs baseline": "—" if p["role"] == "baseline" else
            f"{p['distance_nm'] - base['distance_nm']:+,} nm · "
            f"{p['sea_days'] - base['sea_days']:+.1f} d · "
            f"{'+' if p['total_cost_usd'] >= base['total_cost_usd'] else '-'}"
            f"{money(abs(p['total_cost_usd'] - base['total_cost_usd']))}",
    } for p in (f["properties"] for f in features)])


# ---------------------------------------------------------------- page
st.title("🚢 Corridor Risk Agent")
st.caption("Describe a disruption in the chat, e.g. a closed canal or a high-risk strait. "
           "The agent's route options and scenario appear on the map.")

left, right = st.columns([3, 2], gap="medium")
with right:
    run_turn = chat_panel(height=640)
with left:
    routes = st.session_state.get(AGENT_ROUTES, {})
    features = routes.get("features", [])
    st_folium(route_map(routes), height=560, use_container_width=True, returned_objects=[])
    st.caption("Routes: 🔵 dashed = baseline (normal route), 🟣 🩷 🩵 = alternatives.   "
               "Chokepoints: ⚪ open, 🟠 war risk, 🔴 closed.")
    if features:
        first = features[0]["properties"]
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.markdown(f"**{first['origin']} → {first['destination']}**: route options from the agent")
        if c2.button("Clear map", width="stretch", disabled=st.session_state.get(AGENT_BUSY, False)):
            del st.session_state[AGENT_ROUTES]
            st.rerun()
        st.dataframe(route_table(features), hide_index=True, width="stretch")
    else:
        st.info("No routes yet. Ask the agent to compare routes, e.g. "
                "*\"Suez is closed, what are the options from Shanghai to Rotterdam?\"*")
if run_turn:  # after the map is drawn, so it stays visible while the agent works
    run_turn()
