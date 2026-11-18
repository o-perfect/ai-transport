"""Tools the agent can call. Each returns a JSON-serialisable dict.

Keys starting with "_" (e.g. "_series") are kept in the trace for the UI's
charts but stripped before the result is sent back to the LLM.
"""

import datetime as dt
import hashlib
import json
import pathlib
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime

import requests

from core import costs

from . import network as net

PORTWATCH = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
GOOGLE_NEWS = "https://news.google.com/rss/search"
MARITIME_SITE = "gcaptain.com"  # maritime trade press, searched through Google News
CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / ".cache"

# One keep-alive session for all tools: over a long-distance VPN a new TLS
# handshake costs 1-3 s, so reusing connections matters more than anything else.
SESSION = requests.Session()
SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=16))

PORTWATCH_CHOKEPOINTS = {
    "Suez Canal": "chokepoint1", "Panama Canal": "chokepoint2", "Bosporus Strait": "chokepoint3",
    "Bab el-Mandeb Strait": "chokepoint4", "Malacca Strait": "chokepoint5",
    "Strait of Hormuz": "chokepoint6", "Cape of Good Hope": "chokepoint7",
    "Gibraltar Strait": "chokepoint8", "Dover Strait": "chokepoint9", "Oresund Strait": "chokepoint10",
    "Taiwan Strait": "chokepoint11", "Korea Strait": "chokepoint12", "Tsugaru Strait": "chokepoint13",
    "Luzon Strait": "chokepoint14", "Lombok Strait": "chokepoint15", "Ombai Strait": "chokepoint16",
    "Bohai Strait": "chokepoint17", "Torres Strait": "chokepoint18", "Sunda Strait": "chokepoint19",
    "Makassar Strait": "chokepoint20", "Magellan Strait": "chokepoint21",
    "Yucatan Channel": "chokepoint22", "Windward Passage": "chokepoint23",
    "Mona Passage": "chokepoint24", "Balabac Strait": "chokepoint25", "Bering Strait": "chokepoint26",
    "Mindoro Strait": "chokepoint27", "Kerch Strait": "chokepoint28",
}

# Ship types and all cost figures live in core/costs.py (data/ships.json,
# data/cost_assumptions.json); the tools below never compute a cost themselves.
SHIPS = costs.ship_types()
DEFAULT_SHIP = "ulcv"
PASSAGES = {name: passage for name, (passage, _) in net.CHOKEPOINTS.items()}


def _portwatch_query(layer, params):
    CACHE_DIR.mkdir(exist_ok=True)
    key = CACHE_DIR / (f"{layer}_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
                       .replace(" ", "").replace("'", "").replace("/", "")[:180] + ".json")
    if key.exists() and dt.datetime.now().timestamp() - key.stat().st_mtime < 6 * 3600:
        return json.loads(key.read_text())
    r = SESSION.get(f"{PORTWATCH}/{layer}/FeatureServer/0/query",
                    params={"f": "json", "returnGeometry": "false", "outFields": "*", **params},
                    timeout=30)
    r.raise_for_status()
    rows = [f["attributes"] for f in r.json().get("features", [])]
    key.write_text(json.dumps(rows))
    return rows


def _daily(portid, start, end):
    return _portwatch_query("Daily_Chokepoints_Data", {
        "where": f"portid='{portid}' AND date >= DATE '{start}' AND date <= DATE '{end}'",
        "orderByFields": "date ASC", "resultRecordCount": 2000,
    })


def _avg(rows, field):
    return round(sum(r[field] for r in rows) / len(rows), 1) if rows else None


def get_chokepoint_traffic(chokepoint, days=30):
    """Recent daily transits through a chokepoint (IMF PortWatch, AIS-based),
    compared with the same window one year earlier and in 2023 (pre Red Sea crisis)."""
    name = chokepoint if chokepoint in PORTWATCH_CHOKEPOINTS else net.resolve(chokepoint)
    if name not in PORTWATCH_CHOKEPOINTS:
        return {"error": f"Unknown chokepoint '{chokepoint}'. Known: {list(PORTWATCH_CHOKEPOINTS)}"}
    pid = PORTWATCH_CHOKEPOINTS[name]
    days = max(7, min(int(days), 365))

    latest = _portwatch_query("Daily_Chokepoints_Data", {
        "where": f"portid='{pid}'", "orderByFields": "date DESC", "resultRecordCount": 1})
    if not latest:
        return {"error": "No data returned from PortWatch"}
    end = dt.date.fromisoformat(latest[0]["date"])
    start = end - dt.timedelta(days=days - 1)

    def same_window(year):
        try:
            s, e = start.replace(year=year - (end.year - start.year)), end.replace(year=year)
        except ValueError:  # 29 Feb
            s, e = start - dt.timedelta(days=365), end - dt.timedelta(days=365)
        return s, e

    windows = [(start, end), same_window(end.year - 1), same_window(2023)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        recent, last_year, pre_crisis = pool.map(lambda w: _daily(pid, *w), windows)

    def pct(a, b):
        return None if not a or not b else round(100 * (a - b) / b, 1)

    fields = ["n_total", "n_container", "n_tanker", "n_dry_bulk", "capacity"]
    summary = {f: {"recent_avg": _avg(recent, f), "year_ago_avg": _avg(last_year, f),
                   "2023_avg": _avg(pre_crisis, f),
                   "change_vs_year_ago_pct": pct(_avg(recent, f), _avg(last_year, f)),
                   "change_vs_2023_pct": pct(_avg(recent, f), _avg(pre_crisis, f))}
               for f in fields}
    return {
        "chokepoint": name,
        "window": f"{start} to {end} ({days} days)",
        "units": "n_* = vessel transits per day; capacity = deadweight tonnes per day",
        "daily_averages": summary,
        "source": "IMF PortWatch Daily_Chokepoints_Data",
        "_series": {"date": [r["date"] for r in recent], "n_total": [r["n_total"] for r in recent],
                    "year_ago_n_total": [r["n_total"] for r in last_year]},
    }


def _route_cost(r, ship, speed_knots, fuel_price, war_risk, war_risk_pct, ets_share):
    c = costs.voyage_cost(r["distance_nm"], r["chokepoints"], ship, speed_knots=speed_knots,
                          fuel_price_usd_per_t=fuel_price, flagged_war_risk=tuple(war_risk),
                          war_risk_premium_pct=war_risk_pct, ets_share=ets_share)
    return {
        "distance_nm": round(r["distance_nm"]), "sea_days": round(c.sea_days, 1),
        "canal_waiting_days": c.canal_waiting_days, "chokepoints": r["chokepoints"],
        "war_risk_areas_transited": c.war_risk_chokepoints,
        "cost_usd": c.cost_usd(),
        "cost_per_unit": {"usd": round(c.per_unit, 1), "unit": c.per_unit_label},
        "fuel_t": round(c.fuel_t), "co2_t": round(c.co2_t),
        "path": r["path"],
    }


def _feature(r, costs, role, label):
    return {"type": "Feature", "geometry": r["geometry"],
            "properties": {"role": role, "label": label, "origin": r["path"][0],
                           "destination": r["path"][-1], "distance_nm": costs["distance_nm"],
                           "sea_days": costs["sea_days"], "total_cost_usd": costs["cost_usd"]["total"],
                           "chokepoints": r["chokepoints"]}}


def compare_routes(origin, destination, blocked_chokepoints=(), vessel_type=DEFAULT_SHIP,
                   speed_knots=None, fuel_price_usd_per_t=None, war_risk_chokepoints=(),
                   war_risk_premium_pct=None):
    """Baseline route vs. best alternatives when chokepoints are blocked, with
    distance, sea time and voyage cost from core/costs.py (fuel, charter, tolls,
    war-risk cover, EU ETS) and CO2."""
    o, d = net.resolve_port(origin), net.resolve_port(destination)
    bad = [x for x, r in ((origin, o), (destination, d)) if r is None]
    if bad:
        return {"error": f"Unknown port(s) {bad}. Closest matches: "
                         f"{ {x: net.suggest_ports(x) for x in bad} }. Use a port name, "
                         "'Name, Country' or a UN/LOCODE."}
    if vessel_type not in SHIPS:
        return {"error": f"Unknown vessel_type. Options: {list(SHIPS)}"}
    ship = SHIPS[vessel_type]
    blocked = [net.resolve(c) for c in blocked_chokepoints if net.resolve(c) in net.CHOKEPOINTS]
    unknown = [c for c in blocked_chokepoints if net.resolve(c) not in net.CHOKEPOINTS]
    war_risk = [net.resolve(c) for c in war_risk_chokepoints if net.resolve(c)]
    restricted = costs.too_big_for(ship, PASSAGES)
    ets_share = costs.eu_ets_share(net.port_country(o), net.port_country(d))

    def costed(r):
        return _route_cost(r, ship, speed_knots, fuel_price_usd_per_t, war_risk,
                           war_risk_premium_pct, ets_share)

    base = net.route(o, d, restricted)
    if base is None:
        return {"error": "No route between these ports in the network model"}
    baseline = costed(base)
    alt_routes = net.alternatives(o, d, tuple(blocked) + tuple(restricted), k=3)
    alts = [costed(r) for r in alt_routes]
    for a in alts:
        a["extra_vs_baseline"] = {
            "distance_nm": a["distance_nm"] - baseline["distance_nm"],
            "sea_days": round(a["sea_days"] - baseline["sea_days"], 1),
            "cost_usd": a["cost_usd"]["total"] - baseline["cost_usd"]["total"],
            "co2_t": a["co2_t"] - baseline["co2_t"],
        }
    charged_war_risk = sorted({c for r in (baseline, *alts) for c in r["war_risk_areas_transited"]})
    return {
        "origin": o, "destination": d, "blocked": blocked, "unrecognised_blocked": unknown,
        "vessel": f"{vessel_type}: {ship.name}", "vessel_cannot_use": restricted,
        "eu_ets_share_of_co2": ets_share,
        "assumptions": costs.assumptions_used(ship, speed_knots, fuel_price_usd_per_t,
                                              war_risk_premium_pct),
        "baseline_route": baseline,
        "alternatives": alts if alts else "No feasible route with these blockages",
        "_geometry": {"type": "FeatureCollection", "features": [
            _feature(base, baseline, "baseline", "Baseline: " + (" / ".join(base["chokepoints"]) or "open sea")),
            *(_feature(r, a, "alternative", f"Alt {i}: " + (" / ".join(r["chokepoints"]) or "open sea"))
              for i, (r, a) in enumerate(zip(alt_routes, alts), 1))],
            "scenario": {"blocked": blocked, "war_risk": sorted({*war_risk, *charged_war_risk}),
                         "vessel_cannot_use": restricted}},
    }


AFFECTED_SHOWN = 15  # the full list is long and slows the LLM down; the UI gets all of it


def find_affected_routes(chokepoint, vessel_type=DEFAULT_SHIP, speed_knots=None):
    """Which port-to-port links in the network depend on a chokepoint, and how
    much distance/time each loses if it is closed. Sorted by extra sea days."""
    c = net.resolve(chokepoint)
    if c not in net.CHOKEPOINTS:
        return {"error": f"'{chokepoint}' is not in the routing network. Options: {list(net.CHOKEPOINTS)}"}
    ship = SHIPS.get(vessel_type, SHIPS[DEFAULT_SHIP])
    speed_knots = speed_knots or ship.service_speed_kn
    no_go = tuple(costs.too_big_for(ship, PASSAGES))
    rows = []
    ports = list(net.PORTS)
    for i, a in enumerate(ports):
        for b in ports[i + 1:]:
            r = net.route(a, b, no_go)
            if r is None or c not in r["chokepoints"]:
                continue
            nm = r["distance_nm"]
            r2 = net.route(a, b, no_go + (c,))
            row = {"link": f"{a} - {b}", "baseline_nm": round(nm)}
            if r2 is None:
                row.update(detour="none (link cut)")
            else:
                nm2 = r2["distance_nm"]
                row.update(detour_nm=round(nm2), extra_nm=round(nm2 - nm),
                           extra_days=round((nm2 - nm) / (speed_knots * 24), 1),
                           extra_pct=round(100 * (nm2 - nm) / nm, 1),
                           detour_via=r2["chokepoints"])
            rows.append(row)
    rows.sort(key=lambda r: -r.get("extra_days", 1e9))
    return {"chokepoint": c, "speed_knots": speed_knots, "links_affected": len(rows),
            "links_total": len(ports) * (len(ports) - 1) // 2,
            "links_cut": sum("detour" in r for r in rows),
            "affected": rows[:AFFECTED_SHOWN],
            "note": f"Worst {AFFECTED_SHOWN} links shown" if len(rows) > AFFECTED_SHOWN else "All links shown",
            "_all_affected": rows}


def _news_search(q):
    """Google News RSS search, newest first. Cached for 30 min; an older copy is
    used if the request fails (e.g. 429 rate limit)."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = CACHE_DIR / f"news_{hashlib.md5(q.encode()).hexdigest()}.xml"
    if not key.exists() or dt.datetime.now().timestamp() - key.stat().st_mtime > 1800:
        try:
            r = SESSION.get(GOOGLE_NEWS, params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            r.raise_for_status()
            key.write_bytes(r.content)
        except requests.RequestException:
            if not key.exists():
                raise  # nothing cached to fall back on
    items = []
    for it in ET.fromstring(key.read_bytes()).iter("item"):
        source = it.findtext("source", "")
        title = it.findtext("title", "").removesuffix(f" - {source}")
        if not title.endswith("Archives"):  # site listing pages, not articles
            items.append({"date": f"{parsedate_to_datetime(it.findtext('pubDate')):%Y-%m-%d}",
                          "source": source, "title": title})
    return sorted(items, key=lambda i: i["date"], reverse=True)


def get_news(query, days=7):
    """Recent headlines for a query from maritime trade press (gCaptain) and general news."""
    days = max(1, min(int(days), 30))
    with ThreadPoolExecutor(max_workers=2) as pool:
        maritime, general = pool.map(_news_search, [f"{query} site:{MARITIME_SITE} when:{days}d",
                                                    f"{query} when:{days}d"])
    general = [i for i in general if i["source"] != "gCaptain"]
    return {"query": query, "days": days, "maritime": maritime[:6], "general": general[:8],
            "_items": maritime + general}


# --- OpenAI-style tool schemas ----------------------------------------------

_port_param = {"type": "string",
               "description": "Any world port: a name (e.g. 'Sydney'), 'Name, Country' to disambiguate, "
                              "or a UN/LOCODE (e.g. 'AUSYD'). Main hubs: " + ", ".join(net.PORTS)}
_choke_enum = list(net.CHOKEPOINTS)
_vessel_param = {"type": "string", "enum": list(SHIPS), "default": DEFAULT_SHIP,
                 "description": "Ship type: " + "; ".join(f"{s.id} = {s.name}" for s in SHIPS.values())
                                + ". Pick the one that fits the cargo (containers, dry bulk, crude, LNG)."}
_speed_param = {"type": "number", "description": "Speed in knots. Omit to use the ship's service speed."}

SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_chokepoint_traffic",
        "description": "Live AIS-based daily vessel transits through a maritime chokepoint from IMF PortWatch. "
                       "Returns recent daily averages by vessel type and % change vs. the same window a year "
                       "ago and in 2023 (before the Red Sea crisis). Use to assess current disruption level.",
        "parameters": {"type": "object", "properties": {
            "chokepoint": {"type": "string", "enum": list(PORTWATCH_CHOKEPOINTS)},
            "days": {"type": "integer", "description": "Window length in days (7-365)", "default": 30},
        }, "required": ["chokepoint"]}}},
    {"type": "function", "function": {
        "name": "compare_routes",
        "description": "Compute the normal (baseline) sea route between two ports and up to 3 alternative routes "
                       "when chokepoints are blocked. Returns distance, sea days, canal waiting days, voyage cost "
                       "breakdown (fuel, charter time, canal tolls, war-risk insurance, EU ETS carbon cost), cost "
                       "per TEU or per tonne, CO2, the extra vs. baseline, and the cost assumptions with sources. "
                       "Chokepoints with a current war-risk premium (e.g. Bab el-Mandeb, Hormuz) are always "
                       "charged; use war_risk_chokepoints to also price other dangerous-but-open chokepoints.",
        "parameters": {"type": "object", "properties": {
            "origin": _port_param,
            "destination": _port_param,
            "blocked_chokepoints": {"type": "array", "items": {"type": "string", "enum": _choke_enum},
                                    "description": "Chokepoints that are closed"},
            "vessel_type": _vessel_param,
            "speed_knots": _speed_param,
            "fuel_price_usd_per_t": {"type": "number",
                                     "description": "Omit to use the current assumption (VLSFO)"},
            "war_risk_chokepoints": {"type": "array", "items": {"type": "string", "enum": _choke_enum},
                                     "description": "Extra open but high-risk chokepoints that trigger a war-risk premium"},
            "war_risk_premium_pct": {"type": "number",
                                     "description": "Overrides the war-risk premium (% of hull value per transit) "
                                                    "for every charged chokepoint. Omit to use the defaults."},
        }, "required": ["origin", "destination"]}}},
    {"type": "function", "function": {
        "name": "find_affected_routes",
        "description": "Network impact analysis over the model's main hub ports: the port-to-port links whose "
                       "normal route uses the given chokepoint, with the detour distance and extra days if it "
                       "closes (worst links first).",
        "parameters": {"type": "object", "properties": {
            "chokepoint": {"type": "string", "enum": _choke_enum},
            "vessel_type": _vessel_param,
            "speed_knots": _speed_param,
        }, "required": ["chokepoint"]}}},
    {"type": "function", "function": {
        "name": "get_news",
        "description": "Recent news headlines (date, outlet, title) from maritime trade press (gCaptain) and "
                       "general news (Google News). Use to learn what is happening now: closures, attacks, "
                       "war-risk insurance, fuel prices, port strikes or congestion.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search terms, e.g. 'Red Sea shipping attacks', "
                                                      "'Suez Canal', 'bunker fuel prices'"},
            "days": {"type": "integer", "description": "How many days back (1-30)", "default": 7},
        }, "required": ["query"]}}},
]

FUNCTIONS = {
    "get_chokepoint_traffic": get_chokepoint_traffic,
    "compare_routes": compare_routes,
    "find_affected_routes": find_affected_routes,
    "get_news": get_news,
}
