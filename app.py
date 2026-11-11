"""Shipping Route Planner with the agent chat beside the map: Streamlit UI.

Run with:  streamlit run app.py
All logic lives in core/ and agent/; this file only handles the page and the map.
The map and the chat (app_chat.py) exchange data through st.session_state
(keys in shared_state.py).
"""
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from app_chat import chat_panel
from core.chokepoints import ChokepointRegistry
from core.costs import estimate_voyage
from core.ports import load_ports
from core.routing import compute_route
from core.ships import load_ships
from shared_state import AGENT_BUSY, AGENT_ROUTES, SCENARIO, scenario_names

st.set_page_config(page_title="Shipping Route Planner", page_icon="🚢", layout="wide")

ROUTE_BLUE = "#1a73e8"
NORMAL_GREY = "#7a7f87"
AGENT_COLORS = ["#9334e6", "#e52592", "#12b5cb", "#b06000"]  # baseline, alt 1, alt 2, alt 3


def money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    return f"${x:,.0f}"


def risk_color(cp) -> str:
    if not cp.is_open:
        return "#d93025"  # red = closed
    if cp.risk_score >= 0.6:
        return "#e8710a"  # orange = high risk
    if cp.risk_score >= 0.3:
        return "#f9ab00"  # amber = medium
    return "#1e8e3e"      # green = low


# ---------------------------------------------------------------- data
ports = load_ports()
ships = load_ships()
registry = ChokepointRegistry()  # re-read every run so external changes (AI agent) show up

labels = ports["label"].tolist()
by_label = ports.set_index("label")


def default_index(locode: str) -> int:
    match = ports.index[ports["locode"] == locode]
    return int(match[0]) if len(match) else 0


# ---------------------------------------------------------------- sidebar
busy = st.session_state.get(AGENT_BUSY, False)  # frozen while the agent works

with st.sidebar:
    if busy:
        st.info("🔒 Controls are locked while the agent is working.")
    st.header("Route")
    origin_label = st.selectbox("From", labels, index=default_index("NLRTM"),
                                placeholder="Search a port…", disabled=busy)
    dest_label = st.selectbox("To", labels, index=default_index("CNSHA"),
                              placeholder="Search a port…", disabled=busy)

    st.header("Ship")
    ship = st.selectbox("Ship type", ships, format_func=lambda s: s.name, disabled=busy)
    speed = st.slider("Speed (knots)", 8.0, 25.0, float(ship.service_speed_kn), 0.5,
                      key=f"speed_{ship.id}", disabled=busy)
    bunker = st.number_input("Fuel price (USD / tonne)", 200, 1500, 600, 10, disabled=busy)
    with st.expander("Ship attributes"):
        st.dataframe(
            pd.DataFrame(ship.to_features().items(), columns=["Attribute", "Value"]),
            hide_index=True, width="stretch",
        )

    st.header("Chokepoints")
    st.caption("Switch off to close a chokepoint. The route will avoid it.")
    for cp in registry.all():
        is_open = st.toggle(cp.name, value=cp.is_open, key=f"cp_{cp.id}_{cp.status}", disabled=busy,
                            help=f"Risk score {cp.risk_score:.2f} · "
                                 f"war-risk premium {cp.war_risk_premium_pct}% of hull value"
                                 + (f" · {cp.note}" if cp.note else ""))
        if is_open != cp.is_open:
            registry.set_status(cp.id, "open" if is_open else "closed")
            st.rerun()

    with st.expander("Add a custom chokepoint"):
        with st.form("add_cp", clear_on_submit=True):
            name = st.text_input("Name", placeholder="e.g. Danish Straits", disabled=busy)
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", -90.0, 90.0, 0.0, format="%.3f", disabled=busy)
            lon = c2.number_input("Longitude", -180.0, 180.0, 0.0, format="%.3f", disabled=busy)
            radius = st.number_input("Radius (km)", 10, 1000, 100, disabled=busy)
            risk = st.slider("Risk score", 0.0, 1.0, 0.3, 0.05, disabled=busy)
            premium = st.number_input("War-risk premium (% of hull)", 0.0, 5.0, 0.0, 0.05,
                                      disabled=busy)
            if st.form_submit_button("Add chokepoint", disabled=busy) and name.strip():
                registry.add(name.strip(), lat, lon, radius, risk_score=risk,
                             war_risk_premium_pct=premium)
                st.rerun()

# ---------------------------------------------------------------- routing
st.title("Shipping Route Planner")
st.caption("Plan a sea route between any two ports, close chokepoints, and compare the detour's cost and risk. "
           "Ask the agent on the right; its routes are drawn on the map.")

# Chokepoints closed on the map are the chat's scenario.
st.session_state[SCENARIO] = scenario_names(registry.closed())


def planner():
    """Metrics, map and cost details for the selected route."""
    origin = by_label.loc[origin_label].to_dict()
    dest = by_label.loc[dest_label].to_dict()

    if origin_label == dest_label:
        st.info("Pick two different ports to plan a route.")
        return

    chokepoints = registry.all()
    planned = compute_route(origin, dest, chokepoints, ship, respect_closures=True)
    normal = compute_route(origin, dest, chokepoints, ship, respect_closures=False)
    has_detour = planned.found and normal.found and abs(planned.distance_nm - normal.distance_nm) > 1

    est_p = estimate_voyage(planned, ship, speed, bunker) if planned.found else None
    est_n = estimate_voyage(normal, ship, speed, bunker) if normal.found else None

    # ---------------------------------------------------------------- metrics
    if est_p:
        m1, m2, m3, m4 = st.columns(4)
        d_nm = est_p.distance_nm - est_n.distance_nm if (est_n and has_detour) else None
        d_days = est_p.days - est_n.days if (est_n and has_detour) else None
        d_cost = est_p.total_cost - est_n.total_cost if (est_n and has_detour) else None
        m1.metric("Distance", f"{est_p.distance_nm:,.0f} nm",
                  f"{d_nm:+,.0f} nm vs normal" if d_nm else None, delta_color="inverse")
        m2.metric("Time at sea", f"{est_p.days:.1f} days",
                  f"{d_days:+.1f} days" if d_days else None, delta_color="inverse")
        m3.metric("Estimated cost", money(est_p.total_cost),
                  f"{'+' if d_cost >= 0 else '-'}{money(abs(d_cost))}" if d_cost else None,
                  delta_color="inverse")
        m4.metric("Route risk", f"{est_p.risk_label} ({est_p.risk_score:.2f})")
    else:
        st.error(planned.message)

    # ---------------------------------------------------------------- map
    main_route = planned if planned.found else normal
    m = folium.Map(location=[20, 0], zoom_start=2, tiles=None,
                   world_copy_jump=True, control_scale=True)
    folium.TileLayer(  # free, no API key; light grey keeps the route readable
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Light grey", max_zoom=16).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.LayerControl(position="topright").add_to(m)

    if has_detour or (not planned.found and normal.found):
        folium.PolyLine(normal.coords, color=NORMAL_GREY, weight=3, opacity=0.8,
                        dash_array="8 8", tooltip="Normal route (all chokepoints open)").add_to(m)
    if planned.found:
        folium.PolyLine(planned.coords, color=ROUTE_BLUE, weight=5, opacity=0.9,
                        tooltip=f"Planned route: {planned.distance_nm:,.0f} nm").add_to(m)

    agent_routes = st.session_state.get(AGENT_ROUTES, {}).get("features", [])
    agent_coords = []
    for i, feat in enumerate(agent_routes):
        props = feat["properties"]
        latlon = [(lat, lon) for lon, lat in feat["geometry"]["coordinates"]]
        agent_coords += latlon
        folium.PolyLine(latlon, color=AGENT_COLORS[i % len(AGENT_COLORS)], weight=4, opacity=0.85,
                        dash_array="4 6" if props["role"] == "baseline" else None,
                        tooltip=f"Agent · {props['origin']} → {props['destination']} · "
                                f"{props['label']} · {props['distance_nm']:,} nm").add_to(m)

    for cp in chokepoints:
        status = "CLOSED" if not cp.is_open else "Open"
        tip = (f"<b>{cp.name}</b><br>Status: {status}<br>Risk: {cp.risk_score:.2f}"
               + (f"<br>{cp.note}" if cp.note else ""))
        for shift in (-360, 0, 360):  # draw on every world copy
            folium.CircleMarker([cp.lat, cp.lon + shift], radius=8, color=risk_color(cp),
                                fill=True, fill_opacity=0.85, weight=2, tooltip=tip).add_to(m)
            if not cp.is_open:
                folium.Circle([cp.lat, cp.lon + shift], radius=cp.radius_km * 1000,
                              color="#d93025", weight=1, fill=True, fill_opacity=0.1).add_to(m)

    view = agent_coords + (list(main_route.coords) if main_route.found else [])
    if view:
        m.fit_bounds([[min(c[0] for c in view), min(c[1] for c in view)],
                      [max(c[0] for c in view), max(c[1] for c in view)]], padding=(30, 30))
    if main_route.found:
        start, end = main_route.coords[0], main_route.coords[-1]
    else:
        start, end = (origin["lat"], origin["lon"]), (dest["lat"], dest["lon"])
    folium.Marker(start, tooltip=f"From: {origin_label}",
                  icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
    folium.Marker(end, tooltip=f"To: {dest_label}",
                  icon=folium.Icon(color="red", icon="flag", prefix="fa")).add_to(m)

    if agent_routes:
        first = agent_routes[0]["properties"]
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.info(f"Also showing {len(agent_routes)} route(s) from the agent chat: "
                f"{first['origin']} → {first['destination']}. Hover a line for details.")
        if c2.button("Clear agent routes", width="stretch", disabled=busy):
            del st.session_state[AGENT_ROUTES]
            st.rerun()

    st_folium(m, height=560, use_container_width=True, returned_objects=[])
    st.caption("🔵 Planned route   ⚪ dashed = normal route   "
               + ("🟣 agent routes (dotted = agent baseline)   " if agent_routes else "")
               + "● chokepoints: green = low risk, amber = medium, orange = high, red = closed")

    # ---------------------------------------------------------------- details
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Cost breakdown")
        rows = [("Distance (nm)", "distance_nm", lambda v: f"{v:,.0f}"),
                ("Days at sea", "days", lambda v: f"{v:.1f}"),
                ("Fuel burned (t)", "fuel_t", lambda v: f"{v:,.0f}"),
                ("Fuel cost", "fuel_cost", money),
                ("Ship time cost", "time_cost", money),
                ("Canal tolls", "tolls", money),
                ("War-risk insurance", "war_risk_insurance", money),
                ("Total", "total_cost", money),
                ("Risk score", "risk_score", lambda v: f"{v:.2f}")]
        table = {"Item": [r[0] for r in rows]}
        if est_p:
            table["Planned route"] = [fmt(getattr(est_p, attr)) for _, attr, fmt in rows]
        if est_n and (has_detour or not est_p):
            table["Normal route"] = [fmt(getattr(est_n, attr)) for _, attr, fmt in rows]
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.caption("Simple model: fuel scales with speed³; tolls scale with ship size; "
                   "war-risk insurance = premium % × hull value per risky chokepoint.")

    with right:
        st.subheader("Chokepoints on this route")
        if planned.found and planned.traversed:
            for cp in planned.traversed:
                st.markdown(f"- **{cp.name}**: risk {cp.risk_score:.2f}"
                            + (f", toll {money(cp.toll_usd * ship.toll_factor)}" if cp.toll_usd else "")
                            + (f", war-risk {cp.war_risk_premium_pct}%" if cp.war_risk_premium_pct else ""))
        elif planned.found:
            st.write("None: open water all the way.")
        if planned.blocked_by:
            st.markdown("**Avoided:**")
            for cp in planned.blocked_by:
                reason = "closed" if not cp.is_open else "ship too large"
                st.markdown(f"- {cp.name} ({reason})")


left, right = st.columns([3, 2], gap="medium")
with right:
    run_turn = chat_panel(height=640)
with left:
    planner()
if run_turn:  # after the map is drawn, so it stays visible while the agent works
    run_turn()
