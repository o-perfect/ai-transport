"""The folium map: dark basemap, glowing route lines, status-coloured chokepoints,
a floating legend and an empty state. Draws only what the agent returned
(st.session_state[AGENT_ROUTES]); styles come from the map part of ui/styles.css."""
import html

import folium
from folium.plugins import AntPath

from agent.network import CHOKEPOINTS

from . import components as ui

ESRI_DARK = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
             "World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}")
ESRI_LIGHT = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
              "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}")

PIN_SVG = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22D3EE" stroke-width="1.7" '
           'stroke-linecap="round" stroke-linejoin="round"><path d="M3 17c3-1 4.5-6 9-6s6 4 9 3"/>'
           '<circle cx="12" cy="11" r="1.6" fill="#22D3EE" stroke="none"/></svg>')


def _latlon(feature):
    return [(lat, lon) for lon, lat in feature["geometry"]["coordinates"]]


def _route_tooltip(i, p):
    return folium.Tooltip(
        f"<b>{ui.route_tag(i, p)}</b>: {html.escape(ui.route_name(p))}<br>"
        f'<span class="mono">{p["distance_nm"]:,} nm / {p["sea_days"]:.1f} d / {ui.money(p["total_cost_usd"])}</span>',
        sticky=True)


def _draw_routes(m, features, rec):
    # baseline at the bottom, then the alternatives, the recommended one on top
    order = sorted(range(len(features)), key=lambda i: (i == rec, features[i]["properties"]["role"] != "baseline"))
    for i in order:
        p, pts, color = features[i]["properties"], _latlon(features[i]), ui.route_color(i, features[i]["properties"])
        if p["role"] == "baseline":
            folium.PolyLine(pts, color=color, weight=2, opacity=0.85, dash_array="6 7",
                            tooltip=_route_tooltip(i, p)).add_to(m)
            if i == rec:  # undisrupted: the normal route is the best one, give it the flow
                AntPath(pts, color=color, pulse_color="#E6EDF7", weight=2, opacity=0.6,
                        delay=1800, dash_array=[8, 26]).add_to(m)
            continue
        folium.PolyLine(pts, color=color, weight=10, opacity=0.14,  # soft glow under the line
                        tooltip=_route_tooltip(i, p)).add_to(m)
        if i == rec:
            AntPath(pts, color=color, pulse_color="#E6FBFF", weight=3, opacity=0.95,
                    delay=1800, dash_array=[8, 26], tooltip=_route_tooltip(i, p)).add_to(m)
        else:
            folium.PolyLine(pts, color=color, weight=2.5, opacity=0.95, tooltip=_route_tooltip(i, p)).add_to(m)


def _draw_chokepoints(m, features, scenario):
    for name, (_, (lon, lat)) in CHOKEPOINTS.items():
        label, kind, color = ui.chokepoint_status(name, scenario)
        on = [ui.route_tag(i, f["properties"]) for i, f in enumerate(features)
              if name in (f["properties"].get("chokepoints") or [])]
        halo = '<span class="halo"></span>' if kind in ("closed", "war") else ""
        icon_html = f'<div class="cp {kind}" style="--c:{color}">{halo}<span class="core"></span></div>'
        tip = f'<b>{html.escape(name)}</b><br><span style="color:{color}">{label}</span>'
        pop = (f'<div class="cp-pop"><h4>{html.escape(name)}</h4>'
               f'<div class="status" style="color:{color}">{label}</div>'
               f'<div class="cp-row">Compared routes through here: <b>{", ".join(on) or "none"}</b></div>'
               f'<div class="cp-row mono">{lat:.2f}°, {lon:.2f}°</div></div>')
        for shift in (-360, 0, 360):  # draw on every world copy
            folium.Marker([lat, lon + shift],
                          icon=folium.DivIcon(html=icon_html, icon_size=(22, 22), icon_anchor=(11, 11),
                                              class_name="cp-icon"),
                          tooltip=folium.Tooltip(tip, sticky=False),
                          popup=folium.Popup(pop, max_width=260)).add_to(m)


def _draw_ports(m, features):
    base = features[0]
    (lon0, lat0), (lon1, lat1) = base["geometry"]["coordinates"][0], base["geometry"]["coordinates"][-1]
    for (lat, lon), cls, name in (((lat0, lon0), "origin", base["properties"]["origin"]),
                                  ((lat1, lon1), "destination", base["properties"]["destination"])):
        folium.Marker([lat, lon],
                      icon=folium.DivIcon(html=f'<div class="pt {cls}"></div>', icon_size=(14, 14),
                                          icon_anchor=(7, 7), class_name="pt-icon"),
                      tooltip=folium.Tooltip(html.escape(name), permanent=True, direction="right",
                                             offset=(10, 0), class_name="port-label")).add_to(m)


def _legend(features, rec):
    rows = ['<div class="lg-title">Routes</div>',
            '<div class="lg-row"><span class="lg-line lg-dash"></span>Baseline (normal route)</div>']
    alts = [(i, f["properties"]) for i, f in enumerate(features) if f["properties"]["role"] != "baseline"]
    for i, p in alts:
        color = ui.route_color(i, p)
        star = " (recommended)" if i == rec else ""
        rows.append(f'<div class="lg-row"><span class="lg-line glow" style="background:{color};color:{color}">'
                    f'</span>{ui.route_tag(i, p)}{star}</div>')
    if not alts:
        rows.append(f'<div class="lg-row"><span class="lg-line glow" style="background:{ui.ALT_COLORS[0]};'
                    f'color:{ui.ALT_COLORS[0]}"></span>Alternatives</div>')
    rows += ['<div class="lg-sep"></div>', '<div class="lg-title">Chokepoints</div>']
    for label, color in (("Closed", ui.RED), ("War risk (open)", ui.AMBER), ("Not assessed", ui.NEUTRAL)):
        rows.append(f'<div class="lg-row"><span class="lg-dot" style="background:{color};'
                    f'box-shadow:0 0 8px {color}"></span>{label}</div>')
    return '<div class="map-legend">' + "".join(rows) + "</div>"


EMPTY_STATE = f"""<div class="map-empty"><div class="ring">{PIN_SVG}</div>
<h3>Describe a disruption to see route options</h3>
<p>Try <i>"Suez is closed, options from Shanghai to Rotterdam?"</i> in the chat.
The agent's routes, costs and chokepoint statuses appear here.</p></div>"""


def route_map(routes: dict) -> folium.Map:
    features = routes.get("features", [])
    scenario = routes.get("scenario", {})
    rec = ui.recommended_index(features, scenario) if features else None

    m = folium.Map(location=[22, 40], zoom_start=2, min_zoom=2, tiles=None,
                   world_copy_jump=True, control_scale=True)
    # Esri Dark Gray: free and keyless (CARTO's Dark Matter now needs an API key)
    folium.TileLayer(ESRI_DARK, attr="Tiles &copy; Esri", name="Dark gray (Esri)", max_zoom=16).add_to(m)
    folium.TileLayer(ESRI_LIGHT, attr="Tiles &copy; Esri", name="Light gray (Esri)", max_zoom=16,
                     show=False).add_to(m)
    folium.LayerControl(position="topright").add_to(m)
    m.get_root().header.add_child(folium.Element(f"<style>{ui.map_css()}</style>"))

    if features:
        _draw_routes(m, features, rec)
    _draw_chokepoints(m, features, scenario)
    if features:
        _draw_ports(m, features)
        view = [pt for f in features for pt in _latlon(f)]
        m.fit_bounds([[min(c[0] for c in view), min(c[1] for c in view)],
                      [max(c[0] for c in view), max(c[1] for c in view)]], padding=(40, 40))
    else:
        m.get_root().html.add_child(folium.Element(EMPTY_STATE))
    m.get_root().html.add_child(folium.Element(_legend(features, rec)))
    return m
