"""Presentation helpers for the Streamlit page: CSS, header, route result cards.

Everything here only formats data the agent already returned (the GeoJSON in
st.session_state[AGENT_ROUTES], see shared_state.py); nothing changes what the
agent computes. Styles are in ui/styles.css.
"""
import base64
import html
import pathlib

import pandas as pd
import streamlit as st

CSS_FILE = pathlib.Path(__file__).with_name("styles.css")
MAP_MARKER = "/* @@map@@"

# Palette (keep in sync with .streamlit/config.toml and ui/styles.css)
ACCENT = "#22D3EE"
BASELINE_GREY = "#8B96A8"
ALT_COLORS = ["#22D3EE", "#A78BFA", "#F472B6"]  # alt 1, alt 2, alt 3: line on the map = card stripe
RED, AMBER, GREEN, NEUTRAL = "#EF4444", "#F59E0B", "#22C55E", "#64748B"

# Example prompt chips: (icon, title, subtitle). The prompt sent is app_chat.EXAMPLES[i], unchanged.
EXAMPLE_CHIPS = [
    (":material/sailing:", "Suez Canal closed", "Shanghai → Rotterdam, options and cost"),
    (":material/shield:", "Red Sea war risk", "Insure the transit or go via the Cape?"),
    (":material/oil_barrel:", "Hormuz closure", "Which trade links are hit, how badly"),
    (":material/monitoring:", "Panama traffic", "Live transits vs. last year"),
]

def svg_img(svg: str, size: int) -> str:
    """st.html strips inline <svg>, so icons go in as data-URI images."""
    if "xmlns=" not in svg:  # required when an SVG is loaded as an image
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    data = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}" alt="">'


LOGO_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#22D3EE" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<circle cx="12" cy="12" r="9" opacity=".35"/><circle cx="12" cy="12" r="4.5" opacity=".55"/>'
    '<path d="M3.5 15.5c2.6-.6 4-3.6 7-4.4 3.4-.9 5.2 1.7 9.9.4"/>'
    '<circle cx="10.5" cy="11.1" r="1.5" fill="#22D3EE" stroke="none"/></svg>'
)
AGENT_ICON_SVG = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#22D3EE" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 18l4-6"/><circle cx="12" cy="18" r="1.2" fill="#22D3EE"/></svg>'
)


# ---------------------------------------------------------------- CSS
def _css_parts() -> tuple[str, str]:  # read each run, so edits to styles.css show on reload
    page, _, rest = CSS_FILE.read_text(encoding="utf-8").partition(MAP_MARKER)
    return page, rest.split("*/", 1)[-1]


def inject_css():
    """Page styles; call once per run, before anything is drawn."""
    st.html(f"<style>{_css_parts()[0]}</style>")


def map_css() -> str:
    """Styles for inside the folium iframe (legend, markers, tooltips)."""
    return _css_parts()[1]


# ---------------------------------------------------------------- formatting
def money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    return f"${x:,.0f}"


def signed_money(x: float) -> str:
    return ("+" if x > 0 else "−" if x < 0 else "±") + money(abs(x))


def _delta(value: float, text: str) -> str:
    """Amber up-arrow when worse than baseline, green down-arrow when better."""
    if round(value, 1) == 0:
        return f'<span class="flat">± {text}</span>'
    cls, arrow = ("up", "↑") if value > 0 else ("down", "↓")
    return f'<span class="{cls}">{arrow} {text}</span>'


# ---------------------------------------------------------------- scenario semantics
def chokepoint_status(name: str, scenario: dict) -> tuple[str, str, str]:
    """(label, kind, colour); kind is closed | restricted | war | neutral."""
    if name in scenario.get("blocked", []):
        return "Closed", "closed", RED
    if name in scenario.get("vessel_cannot_use", []):
        return "Too large for this vessel", "restricted", RED
    if name in scenario.get("war_risk", []):
        return "War risk (open)", "war", AMBER
    return "Not assessed in this scenario", "neutral", NEUTRAL


def route_color(i: int, props: dict) -> str:
    return BASELINE_GREY if props["role"] == "baseline" else ALT_COLORS[(i - 1) % len(ALT_COLORS)]


def route_tag(i: int, props: dict) -> str:
    return "Baseline" if props["role"] == "baseline" else f"Alt {i}"


def _short(name: str) -> str:
    return name.replace("Strait of ", "").replace(" Strait", "")


def route_name(props: dict) -> str:
    chokes = props.get("chokepoints") or []
    return "Open sea" if not chokes else "via " + " → ".join(_short(c) for c in chokes)  # in passing order


def route_risk(props: dict, scenario: dict) -> tuple[str, str]:
    """(badge text, css class) from the chokepoints the route passes."""
    chokes = set(props.get("chokepoints") or [])
    closed = chokes & set(scenario.get("blocked", [])) | chokes & set(scenario.get("vessel_cannot_use", []))
    if closed:
        return "Passes closure", "blocked"
    if chokes & set(scenario.get("war_risk", [])):
        return "War risk", "war"
    return "No flagged risk", "ok"


def recommended_index(features: list, scenario: dict) -> int | None:
    """UI heuristic, not the agent's verdict: the cheapest route that doesn't
    pass a closed chokepoint (the baseline wins ties)."""
    feasible = [i for i, f in enumerate(features) if route_risk(f["properties"], scenario)[1] != "blocked"]
    if not feasible:
        return None
    return min(feasible, key=lambda i: (features[i]["properties"]["total_cost_usd"], i))


# ---------------------------------------------------------------- header
def header(busy: bool):
    pill = ('<span class="cra-pill busy"><span class="dot"></span>Analyzing…</span>' if busy else
            '<span class="cra-pill"><span class="dot"></span>Agent ready</span>')
    st.html(f"""
<div class="cra-header">
  <div class="cra-brand">
    <div class="cra-logo">{svg_img(LOGO_SVG, 22)}</div>
    <div>
      <div class="cra-title">Corridor Risk Agent<span class="cra-title-sub">Maritime Intelligence</span></div>
      <div class="cra-tagline">Type or speak a disruption in plain language. The agent checks live traffic and
        news, reroutes, and prices every option.</div>
    </div>
  </div>
  <div class="cra-header-right">
    <span class="cra-source">Data: IMF PortWatch, searoute, Google News</span>
    {pill}
  </div>
</div>""")


def chat_title():
    st.html(f"""
<div class="cra-chat-title">
  <div class="ico">{svg_img(AGENT_ICON_SVG, 15)}</div>
  <div><b>Risk analyst</b><small>Answers built on live data and tools</small></div>
</div>""")


def example_label(i: int) -> str:
    _, title, sub = EXAMPLE_CHIPS[i]
    return f"**{title}**  \n{sub}"


def example_icon(i: int) -> str:
    return EXAMPLE_CHIPS[i][0]


# ---------------------------------------------------------------- results panel
def results_title(features: list):
    p = features[0]["properties"]
    st.html(f"""
<div class="cra-section">
  <span class="cra-eyebrow">Route options</span>
  <span class="cra-section-title">{html.escape(p['origin'])}<span class="arrow">→</span>{html.escape(p['destination'])}</span>
</div>""")


def kpi_strip(features: list, scenario: dict):
    base = features[0]["properties"]
    rec = recommended_index(features, scenario)
    flagged = [*scenario.get("blocked", []), *scenario.get("war_risk", [])]
    flagged_sub = ", ".join(
        f"{_short(c)} {'closed' if c in scenario.get('blocked', []) else 'war risk'}" for c in flagged) or "none"

    if rec is None:
        rec_value, days_value, days_sub, cost_value, cost_sub = "None feasible", "—", "", "—", ""
    else:
        p = features[rec]["properties"]
        rec_value = f"{route_tag(rec, p)}: {route_name(p)}"
        d_days, d_nm = p["sea_days"] - base["sea_days"], p["distance_nm"] - base["distance_nm"]
        d_cost = p["total_cost_usd"] - base["total_cost_usd"]
        days_value, days_sub = f"{d_days:+.1f} d", f"{d_nm:+,} nm vs. baseline"
        cost_value, cost_sub = signed_money(d_cost), f"baseline {money(base['total_cost_usd'])}"

    tiles = [
        ("Recommended", rec_value, "lowest cost avoiding closures", "text"),
        ("Added transit", days_value, days_sub, ""),
        ("Cost impact", cost_value, cost_sub, ""),
        ("Flagged chokepoints", str(len(flagged)), flagged_sub, ""),
    ]
    st.html('<div class="cra-kpis">' + "".join(
        f'<div class="cra-kpi"><div class="cra-kpi-label">{label}</div>'
        f'<div class="cra-kpi-value {cls}" title="{html.escape(value)}">{html.escape(value)}</div>'
        f'<div class="cra-kpi-sub">{html.escape(sub)}</div></div>'
        for label, value, sub, cls in tiles) + "</div>")


def route_cards(features: list, scenario: dict):
    base = features[0]["properties"]
    rec = recommended_index(features, scenario)
    cards = []
    for i, f in enumerate(features):
        p = f["properties"]
        color, baseline = route_color(i, p), p["role"] == "baseline"
        risk_text, risk_cls = route_risk(p, scenario)
        badges = f'<span class="cra-badge {risk_cls}">{risk_text}</span>'
        if i == rec:
            badges = '<span class="cra-badge reco">Recommended</span>' + badges
        if baseline:
            deltas = '<span class="ref">Reference route (normal conditions)</span>'
        else:
            deltas = "".join([
                _delta(p["distance_nm"] - base["distance_nm"], f"{p['distance_nm'] - base['distance_nm']:+,} nm"),
                _delta(p["sea_days"] - base["sea_days"], f"{p['sea_days'] - base['sea_days']:+.1f} d"),
                _delta(p["total_cost_usd"] - base["total_cost_usd"],
                       signed_money(p["total_cost_usd"] - base["total_cost_usd"])),
            ])
        classes = "cra-card" + (" baseline" if baseline else "") + (" recommended" if i == rec else "")
        cards.append(f"""
<div class="{classes}" style="--route:{color}">
  <div class="cra-card-top">
    <span class="cra-route-tag"><span class="cra-swatch{' dashed' if baseline else ''}"></span>{route_tag(i, p)}</span>
    <span class="cra-badges">{badges}</span>
  </div>
  <div class="cra-card-name">{html.escape(route_name(p))}</div>
  <div class="cra-cost">
    <div class="cra-metric-label">Est. voyage cost</div><div class="cra-metric-value">{money(p['total_cost_usd'])}</div>
  </div>
  <div class="cra-metrics">
    <div><div class="cra-metric-label">Distance</div><div class="cra-metric-value">{p['distance_nm']:,} nm</div></div>
    <div><div class="cra-metric-label">Transit</div><div class="cra-metric-value">{p['sea_days']:.1f} d</div></div>
  </div>
  <div class="cra-deltas">{deltas}</div>
</div>""")
    n = len(cards)
    st.html(f'<div class="cra-cards n{min(n, 4)}">' + "".join(cards) + "</div>"
            '<div class="cra-note">"Recommended" marks the lowest total cost among routes that avoid closed '
            "chokepoints. It is a display aid; the agent's own assessment is in the chat.</div>")


def route_table(features) -> pd.DataFrame:
    base = features[0]["properties"]
    return pd.DataFrame([{
        "Route": p["label"],
        "Distance (nm)": f"{p['distance_nm']:,}",
        "Sea days": p["sea_days"],
        "Total cost": money(p["total_cost_usd"]),
        "Extra vs baseline": "—" if p["role"] == "baseline" else
            f"{p['distance_nm'] - base['distance_nm']:+,} nm, "
            f"{p['sea_days'] - base['sea_days']:+.1f} d, "
            f"{signed_money(p['total_cost_usd'] - base['total_cost_usd'])}",
    } for p in (f["properties"] for f in features)])
