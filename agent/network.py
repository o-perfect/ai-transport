"""Sea routing for the agent, on the searoute maritime network.

Ports are a table of main hubs plus the ~3,900 world ports bundled with
searoute (core/ports.py); chokepoints are the IMF PortWatch
chokepoints that have a matching searoute passage, so closing one means
passing that passage to searoute as a restriction. Alternatives come from
re-routing with each passage the route uses blocked in turn.

All coordinates are [lon, lat], as searoute and GeoJSON expect.
"""

import copy
import difflib
import math
import warnings
from functools import lru_cache

import searoute as sr
from searoute.searoute import setup_M

from core.ports import load_ports

# Main hub ports, name -> [lon, lat]. find_affected_routes checks every pair of
# these; compare_routes also takes any port from the world list (resolve_port).
PORTS = {
    "Shanghai": [121.8, 31.2],
    "Shenzhen": [114.1, 22.5],
    "Hong Kong": [114.17, 22.29],
    "Busan": [129.0, 35.1],
    "Tokyo": [139.8, 35.4],
    "Singapore": [103.8, 1.26],
    "Colombo": [79.85, 6.95],
    "Mumbai": [72.8, 18.9],
    "Jebel Ali": [55.1, 25.0],
    "Ras Tanura": [50.17, 26.65],
    "Jeddah": [39.1, 21.5],
    "Piraeus": [23.6, 37.9],
    "Constanta": [28.65, 44.17],
    "Genoa": [8.9, 44.4],
    "Rotterdam": [4.1, 51.95],
    "Hamburg": [8.7, 53.9],
    "Durban": [31.0, -29.9],
    "Lagos": [3.4, 6.4],
    "Santos": [-46.3, -24.0],
    "New York": [-73.9, 40.5],
    "Houston": [-94.7, 29.3],
    "Los Angeles": [-118.3, 33.7],
}

# PortWatch chokepoint name -> (searoute passage, [lon, lat]). PortWatch
# chokepoints without a searoute passage (Taiwan, Dover, Lombok, ...) can't be
# blocked on this network and are left out. searoute's "chili" passage is the
# Cape Horn / Drake Passage route, not the Magellan Strait, so it isn't mapped.
CHOKEPOINTS = {
    "Suez Canal": ("suez", [32.35, 30.6]),
    "Panama Canal": ("panama", [-79.7, 9.1]),
    "Bosporus Strait": ("bosporus", [29.05, 41.1]),
    "Bab el-Mandeb Strait": ("babalmandab", [43.3, 12.6]),
    "Malacca Strait": ("malacca", [100.6, 3.2]),
    "Strait of Hormuz": ("ormuz", [56.3, 26.5]),
    "Cape of Good Hope": ("south_africa", [20.0, -34.8]),
    "Gibraltar Strait": ("gibraltar", [-5.6, 35.95]),
    "Sunda Strait": ("sunda", [105.8, -6.0]),
    "Bering Strait": ("bering", [-169.0, 65.8]),
}
PASSAGE_TO_CHOKEPOINT = {p: name for name, (p, _) in CHOKEPOINTS.items()}

# Always avoided: the Northwest Passage is ice-bound for normal shipping.
ALWAYS_RESTRICTED = ("northwest",)

ALIASES = {
    "la": "Los Angeles", "long beach": "Los Angeles", "ny": "New York", "nyc": "New York",
    "dubai": "Jebel Ali", "yantian": "Shenzhen", "hk": "Hong Kong", "istanbul": "Bosporus Strait",
    "suez": "Suez Canal", "panama": "Panama Canal", "hormuz": "Strait of Hormuz",
    "bab el-mandeb": "Bab el-Mandeb Strait", "bab al-mandab": "Bab el-Mandeb Strait",
    "red sea": "Bab el-Mandeb Strait", "malacca": "Malacca Strait", "bosphorus": "Bosporus Strait",
    "gibraltar": "Gibraltar Strait", "good hope": "Cape of Good Hope", "cape": "Cape of Good Hope",
}


def resolve(name):
    """Case-insensitive lookup of a port or chokepoint name, with aliases."""
    names = [*PORTS, *CHOKEPOINTS]
    if name in names:
        return name
    low = name.strip().lower()
    if low in ALIASES:
        return ALIASES[low]
    if len(low) < 4:
        return None
    for n in names:
        if n.lower() == low:
            return n
    for n in names:
        if low in n.lower() or n.lower() in low:
            return n
    return None


@lru_cache(maxsize=1)
def _world_ports():
    """label "Name, Country (LOCODE)" -> [lon, lat], plus lookup tables."""
    df = load_ports()
    coords = {r.label: [r.lon, r.lat] for r in df.itertuples()}
    by_locode = {r.locode.lower(): r.label for r in df.itertuples() if r.locode}
    by_name = {}
    for r in df.itertuples():  # sorted by name, country: first one wins
        by_name.setdefault(r.name.lower(), r.label)
    return coords, by_locode, by_name


def resolve_port(name):
    """A main hub name, or a world port label "Name, Country (LOCODE)".
    Accepts a port name, "Name, Country", a label, or a UN/LOCODE."""
    r = resolve(name)
    if r in PORTS:
        return r
    coords, by_locode, by_name = _world_ports()
    low = name.strip().lower()
    if low.replace(" ", "") in by_locode:
        return by_locode[low.replace(" ", "")]
    if low in by_name:
        return by_name[low]
    for label in coords:  # "Sydney, Australia" or a full label
        if label.lower().startswith(low):
            return label
    return None


def suggest_ports(name, n=5):
    """Close matches for an unknown port name, as world port labels."""
    _, _, by_name = _world_ports()
    names = list(PORTS) + list(by_name)
    hits = difflib.get_close_matches(name.strip().lower(), [x.lower() for x in names], n=n)
    return [resolve_port(h) for h in hits]


def port_coords(port):
    return PORTS.get(port) or _world_ports()[0][port]


HUB_COUNTRIES = {
    "Shanghai": "CN", "Shenzhen": "CN", "Hong Kong": "HK", "Busan": "KR", "Tokyo": "JP",
    "Singapore": "SG", "Colombo": "LK", "Mumbai": "IN", "Jebel Ali": "AE", "Ras Tanura": "SA",
    "Jeddah": "SA", "Piraeus": "GR", "Constanta": "RO", "Genoa": "IT", "Rotterdam": "NL",
    "Hamburg": "DE", "Durban": "ZA", "Lagos": "NG", "Santos": "BR", "New York": "US",
    "Houston": "US", "Los Angeles": "US",
}


def port_country(port):
    """ISO alpha-2 country code of a resolved port: the hub table, or the first two
    letters of the UN/LOCODE in a world port label "Name, Country (LOCODE)"."""
    if port in HUB_COUNTRIES:
        return HUB_COUNTRIES[port]
    if port.endswith(")") and "(" in port:
        return port[port.rindex("(") + 1:port.rindex("(") + 3].upper()
    return ""


@lru_cache(maxsize=1)
def _network():
    """searoute's network with two fixes so blocked passages really are blocked:
    - Edge data is stored per direction and searoute tags passages on one
      direction only, so its two-way search could still cross a "blocked"
      passage the other way. Tags are copied to both directions.
    - Its "south_africa" passage is a few edges the graph can go around at 40S,
      so every edge crossing 20E south of Cape Agulhas (34.8S) is tagged too:
      any route around Africa has to cross there.
    """
    M = copy.deepcopy(setup_M())
    tags = {}
    for u, v in M.edges():
        for a, b in ((u, v), (v, u)):
            passage = (M.get_edge_data(a, b) or {}).get("passage")
            if passage:
                tags[frozenset((u, v))] = passage
        (x1, y1), (x2, y2) = u, v
        if min(x1, x2) <= 20 <= max(x1, x2) and abs(x1 - x2) < 180 and max(y1, y2) < -34:
            tags[frozenset((u, v))] = "south_africa"
    for pair, passage in tags.items():
        u, v = tuple(pair)
        for a, b in ((u, v), (v, u)):
            data = M.get_edge_data(a, b)
            if data is not None:
                data["passage"] = passage
    return M


@lru_cache(maxsize=4096)
def _searoute(origin, destination, restrictions):
    """(length_nm, coords, passages) or None if every path is restricted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feat = sr.searoute(port_coords(origin), port_coords(destination), units="naut",
                           restrictions=list(restrictions), append_orig_dest=True,
                           return_passages=True, M=_network())
    length = feat.properties.get("length") or 0
    if not length or math.isinf(length) or any("No path" in str(w.message) for w in caught):
        return None
    coords = tuple(tuple(c) for c in feat.geometry["coordinates"])
    return length, coords, tuple(feat.properties.get("traversed_passages") or ())


def _order_along(names, coords):
    """Sort chokepoints by where the route passes closest to them."""
    def nearest_index(name):
        cx, cy = CHOKEPOINTS[name][1]
        return min(range(len(coords)), key=lambda i: (((coords[i][0] - cx + 180) % 360) - 180) ** 2
                   + (coords[i][1] - cy) ** 2)
    return sorted(names, key=nearest_index)


def route(origin, destination, blocked=()):
    """Shortest sea route avoiding the blocked chokepoints, or None.

    Returns {"distance_nm", "chokepoints" (in passing order), "path"
    (origin, chokepoints, destination), "geometry" (GeoJSON LineString)}.
    """
    restrictions = tuple(sorted({*ALWAYS_RESTRICTED, *(CHOKEPOINTS[c][0] for c in blocked)}))
    found = _searoute(origin, destination, restrictions)
    if found is None:
        return None
    length, coords, passages = found
    chokes = _order_along([PASSAGE_TO_CHOKEPOINT[p] for p in passages if p in PASSAGE_TO_CHOKEPOINT],
                          coords)
    return {
        "distance_nm": length,
        "chokepoints": chokes,
        "path": [origin, *chokes, destination],
        "geometry": {"type": "LineString", "coordinates": [list(c) for c in coords]},
    }


def alternatives(origin, destination, blocked=(), k=3):
    """Up to k distinct routes, shortest first: the best route avoiding
    `blocked`, then re-routes with each chokepoint an already-found route
    passes through additionally blocked."""
    found = {}
    frontier = [tuple(blocked)]
    seen = set()
    while frontier and len(found) < k * 3:
        b = frontier.pop(0)
        if frozenset(b) in seen:
            continue
        seen.add(frozenset(b))
        r = route(origin, destination, b)
        if r is None:
            continue
        found.setdefault(tuple(r["chokepoints"]), r)
        frontier += [b + (c,) for c in r["chokepoints"]]
    return sorted(found.values(), key=lambda r: r["distance_nm"])[:k]
