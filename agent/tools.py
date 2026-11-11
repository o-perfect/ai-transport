"""Tools the agent can call. Each returns a JSON-serialisable dict.

Keys starting with "_" (e.g. "_series") are kept in the trace for the UI's
charts but stripped before the result is sent back to the LLM.
"""

import datetime as dt
import json
import pathlib

import requests

from . import network as net

PORTWATCH = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / ".cache"

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

# Indicative vessel profiles. All figures are planning assumptions, not quotes.
# fuel_t_day is consumption at ref_knots; it scales with (speed/ref)^3.
VESSELS = {
    "container_large": dict(desc="~14,000 TEU container ship", fuel_t_day=80, ref_knots=16,
                            charter_usd_day=60_000, hull_value_usd=150e6,
                            tolls={"Suez Canal": 500_000, "Panama Canal": 450_000}, no_go=[]),
    "container_medium": dict(desc="~8,000 TEU container ship", fuel_t_day=60, ref_knots=16,
                             charter_usd_day=40_000, hull_value_usd=90e6,
                             tolls={"Suez Canal": 330_000, "Panama Canal": 300_000}, no_go=[]),
    "tanker_suezmax": dict(desc="Suezmax crude tanker (~1M bbl)", fuel_t_day=50, ref_knots=13,
                           charter_usd_day=45_000, hull_value_usd=80e6,
                           tolls={"Suez Canal": 350_000}, no_go=["Panama Canal"]),
    "bulk_capesize": dict(desc="Capesize dry bulk carrier (~180,000 dwt)", fuel_t_day=45, ref_knots=13,
                          charter_usd_day=25_000, hull_value_usd=60e6,
                          tolls={"Suez Canal": 300_000}, no_go=["Panama Canal"]),
}
CO2_PER_T_FUEL = 3.151  # t CO2 per t VLSFO (IMO default factor)


def _portwatch_query(layer, params):
    CACHE_DIR.mkdir(exist_ok=True)
    key = CACHE_DIR / (f"{layer}_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
                       .replace(" ", "").replace("'", "").replace("/", "")[:180] + ".json")
    if key.exists() and dt.datetime.now().timestamp() - key.stat().st_mtime < 6 * 3600:
        return json.loads(key.read_text())
    r = requests.get(f"{PORTWATCH}/{layer}/FeatureServer/0/query",
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
    recent = _daily(pid, start, end)

    def same_window(year):
        try:
            s, e = start.replace(year=year - (end.year - start.year)), end.replace(year=year)
        except ValueError:  # 29 Feb
            s, e = start - dt.timedelta(days=365), end - dt.timedelta(days=365)
        return _daily(pid, s, e)

    last_year = same_window(end.year - 1)
    pre_crisis = same_window(2023)

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


def _route_cost(r, vessel, speed_knots, fuel_price, war_risk, war_risk_pct):
    nm, chokes = r["distance_nm"], r["chokepoints"]
    days = nm / (speed_knots * 24)
    fuel_t = days * vessel["fuel_t_day"] * (speed_knots / vessel["ref_knots"]) ** 3
    tolls = sum(vessel["tolls"].get(c, 0) for c in chokes)
    risky = [c for c in chokes if c in war_risk]
    insurance = vessel["hull_value_usd"] * war_risk_pct / 100 if risky else 0
    fuel_cost = fuel_t * fuel_price
    charter = days * vessel["charter_usd_day"]
    return {
        "distance_nm": round(nm), "sea_days": round(days, 1), "chokepoints": chokes,
        "war_risk_areas_transited": risky,
        "cost_usd": {"fuel": round(fuel_cost), "charter_time": round(charter), "canal_tolls": tolls,
                     "war_risk_insurance": round(insurance),
                     "total": round(fuel_cost + charter + tolls + insurance)},
        "fuel_t": round(fuel_t), "co2_t": round(fuel_t * CO2_PER_T_FUEL),
        "path": r["path"],
    }


def _feature(r, role, label):
    return {"type": "Feature", "geometry": r["geometry"],
            "properties": {"role": role, "label": label, "origin": r["path"][0],
                           "destination": r["path"][-1], "distance_nm": round(r["distance_nm"]),
                           "chokepoints": r["chokepoints"]}}


def compare_routes(origin, destination, blocked_chokepoints=(), vessel_type="container_large",
                   speed_knots=14.0, fuel_price_usd_per_t=600.0, war_risk_chokepoints=(),
                   war_risk_premium_pct=0.7):
    """Baseline route vs. best alternatives when chokepoints are blocked, with
    distance, sea time, voyage cost (fuel, charter, tolls, war-risk cover) and CO2."""
    o, d = net.resolve_port(origin), net.resolve_port(destination)
    bad = [x for x, r in ((origin, o), (destination, d)) if r is None]
    if bad:
        return {"error": f"Unknown port(s) {bad}. Closest matches: "
                         f"{ {x: net.suggest_ports(x) for x in bad} }. Use a port name, "
                         "'Name, Country' or a UN/LOCODE."}
    if vessel_type not in VESSELS:
        return {"error": f"Unknown vessel_type. Options: {list(VESSELS)}"}
    vessel = VESSELS[vessel_type]
    blocked = [net.resolve(c) for c in blocked_chokepoints if net.resolve(c) in net.CHOKEPOINTS]
    unknown = [c for c in blocked_chokepoints if net.resolve(c) not in net.CHOKEPOINTS]
    war_risk = [net.resolve(c) for c in war_risk_chokepoints if net.resolve(c)]
    restricted = vessel["no_go"]

    def costed(r):
        return _route_cost(r, vessel, speed_knots, fuel_price_usd_per_t, war_risk,
                           war_risk_premium_pct)

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
    return {
        "origin": o, "destination": d, "blocked": blocked, "unrecognised_blocked": unknown,
        "vessel": f"{vessel_type}: {vessel['desc']}", "vessel_cannot_use": restricted,
        "assumptions": {"speed_knots": speed_knots, "fuel_price_usd_per_t": fuel_price_usd_per_t,
                        "charter_usd_day": vessel["charter_usd_day"],
                        "war_risk_premium_pct_of_hull": war_risk_premium_pct if war_risk else None,
                        "note": "Sea time only; excludes port time, canal queues and cargo inventory cost."},
        "baseline_route": baseline,
        "alternatives": alts if alts else "No feasible route with these blockages",
        "_geometry": {"type": "FeatureCollection", "features": [
            _feature(base, "baseline", "Baseline: " + (" / ".join(base["chokepoints"]) or "open sea")),
            *(_feature(r, "alternative", f"Alt {i}: " + (" / ".join(r["chokepoints"]) or "open sea"))
              for i, r in enumerate(alt_routes, 1))]},
    }


AFFECTED_SHOWN = 15  # the full list is long and slows the LLM down; the UI gets all of it


def find_affected_routes(chokepoint, vessel_type="container_large", speed_knots=14.0):
    """Which port-to-port links in the network depend on a chokepoint, and how
    much distance/time each loses if it is closed. Sorted by extra sea days."""
    c = net.resolve(chokepoint)
    if c not in net.CHOKEPOINTS:
        return {"error": f"'{chokepoint}' is not in the routing network. Options: {list(net.CHOKEPOINTS)}"}
    no_go = tuple(VESSELS.get(vessel_type, VESSELS["container_large"])["no_go"])
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


# --- OpenAI-style tool schemas ----------------------------------------------

_port_param = {"type": "string",
               "description": "Any world port: a name (e.g. 'Sydney'), 'Name, Country' to disambiguate, "
                              "or a UN/LOCODE (e.g. 'AUSYD'). Main hubs: " + ", ".join(net.PORTS)}
_choke_enum = list(net.CHOKEPOINTS)

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
                       "when chokepoints are blocked. Returns distance, sea days, voyage cost breakdown "
                       "(fuel, charter time, canal tolls, war-risk insurance), CO2, and the extra vs. baseline. "
                       "Use war_risk_chokepoints to price passing a dangerous-but-open chokepoint instead of "
                       "blocking it.",
        "parameters": {"type": "object", "properties": {
            "origin": _port_param,
            "destination": _port_param,
            "blocked_chokepoints": {"type": "array", "items": {"type": "string", "enum": _choke_enum},
                                    "description": "Chokepoints that are closed"},
            "vessel_type": {"type": "string", "enum": list(VESSELS), "default": "container_large"},
            "speed_knots": {"type": "number", "default": 14},
            "fuel_price_usd_per_t": {"type": "number", "default": 600},
            "war_risk_chokepoints": {"type": "array", "items": {"type": "string", "enum": _choke_enum},
                                     "description": "Open but high-risk chokepoints that trigger a war-risk premium"},
            "war_risk_premium_pct": {"type": "number", "default": 0.7,
                                     "description": "War-risk premium, % of hull value per voyage"},
        }, "required": ["origin", "destination"]}}},
    {"type": "function", "function": {
        "name": "find_affected_routes",
        "description": "Network impact analysis over the model's main hub ports: the port-to-port links whose "
                       "normal route uses the given chokepoint, with the detour distance and extra days if it "
                       "closes (worst links first).",
        "parameters": {"type": "object", "properties": {
            "chokepoint": {"type": "string", "enum": _choke_enum},
            "vessel_type": {"type": "string", "enum": list(VESSELS), "default": "container_large"},
            "speed_knots": {"type": "number", "default": 14},
        }, "required": ["chokepoint"]}}},
]

FUNCTIONS = {
    "get_chokepoint_traffic": get_chokepoint_traffic,
    "compare_routes": compare_routes,
    "find_affected_routes": find_affected_routes,
}
