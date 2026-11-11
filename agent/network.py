"""Coarse global sea-lane graph for route comparison.

Nodes are ports, chokepoints and open-sea waypoints. Edge lengths are
great-circle distances between nodes (nautical miles), with waypoints placed
so straight segments stay at sea. Accurate to roughly +/-5% against published
port-to-port distances, which is enough for comparing alternatives. Swap this
out for the frontend's route planner once the interfaces are agreed.
"""

import heapq
import math

# name -> (lat, lon, kind)
NODES = {
    # ports
    "Shanghai": (31.2, 121.8, "port"),
    "Shenzhen": (22.5, 114.1, "port"),
    "Busan": (35.1, 129.0, "port"),
    "Tokyo": (35.4, 139.8, "port"),
    "Singapore": (1.26, 103.8, "port"),
    "Mumbai": (18.9, 72.8, "port"),
    "Jebel Ali": (25.0, 55.1, "port"),
    "Jeddah": (21.5, 39.1, "port"),
    "Piraeus": (37.9, 23.6, "port"),
    "Genoa": (44.4, 8.9, "port"),
    "Rotterdam": (51.95, 4.1, "port"),
    "Hamburg": (53.9, 8.7, "port"),
    "Durban": (-29.9, 31.0, "port"),
    "Lagos": (6.4, 3.4, "port"),
    "Santos": (-24.0, -46.3, "port"),
    "New York": (40.5, -73.9, "port"),
    "Houston": (29.3, -94.7, "port"),
    "Los Angeles": (33.7, -118.3, "port"),
    # chokepoints (names match IMF PortWatch)
    "Taiwan Strait": (24.5, 119.5, "chokepoint"),
    "Malacca Strait": (2.8, 101.0, "chokepoint"),
    "Sunda Strait": (-6.0, 105.8, "chokepoint"),
    "Strait of Hormuz": (26.5, 56.3, "chokepoint"),
    "Bab el-Mandeb Strait": (12.6, 43.3, "chokepoint"),
    "Suez Canal": (30.5, 32.3, "chokepoint"),
    "Gibraltar Strait": (35.95, -5.6, "chokepoint"),
    "Dover Strait": (51.0, 1.5, "chokepoint"),
    "Cape of Good Hope": (-34.8, 18.5, "chokepoint"),
    "Panama Canal": (9.0, -79.6, "chokepoint"),
    # open-sea waypoints
    "South China Sea": (8.0, 110.0, "waypoint"),
    "Andaman Sea exit": (6.0, 94.5, "waypoint"),
    "Off Sri Lanka": (5.5, 80.5, "waypoint"),
    "Gulf of Aden": (12.5, 50.0, "waypoint"),
    "Arabian Sea": (17.0, 60.0, "waypoint"),
    "Red Sea": (20.5, 38.5, "waypoint"),
    "Eastern Med": (33.5, 28.0, "waypoint"),
    "Central Med": (37.0, 11.5, "waypoint"),
    "Cape Finisterre": (43.5, -9.8, "waypoint"),
    "Ushant": (48.5, -5.5, "waypoint"),
    "South of Madagascar": (-27.5, 46.0, "waypoint"),
    "Off Dakar": (14.5, -18.5, "waypoint"),
    "Gulf of Guinea": (1.0, 2.0, "waypoint"),
    "South Atlantic": (-15.0, -10.0, "waypoint"),
    "Windward Passage": (20.0, -73.5, "waypoint"),
    "Florida Strait": (24.0, -81.0, "waypoint"),
    "North Atlantic": (40.0, -40.0, "waypoint"),
    "Azores": (38.0, -25.0, "waypoint"),
    "Off Brazil": (-8.0, -34.0, "waypoint"),
}

EDGES = [
    # East Asia
    ("Shanghai", "Taiwan Strait"), ("Shanghai", "Busan"), ("Busan", "Tokyo"),
    ("Taiwan Strait", "Shenzhen"), ("Shenzhen", "South China Sea"),
    ("South China Sea", "Singapore"), ("Singapore", "Malacca Strait"),
    ("Singapore", "Sunda Strait"),
    # Indian Ocean
    ("Malacca Strait", "Andaman Sea exit"), ("Andaman Sea exit", "Off Sri Lanka"),
    ("Off Sri Lanka", "Mumbai"), ("Off Sri Lanka", "Arabian Sea"),
    ("Off Sri Lanka", "Gulf of Aden"), ("Mumbai", "Arabian Sea"),
    ("Arabian Sea", "Strait of Hormuz"), ("Strait of Hormuz", "Jebel Ali"),
    ("Arabian Sea", "Gulf of Aden"), ("Gulf of Aden", "Bab el-Mandeb Strait"),
    ("Off Sri Lanka", "South of Madagascar"), ("Sunda Strait", "South of Madagascar"),
    ("Arabian Sea", "South of Madagascar"),
    ("South of Madagascar", "Durban"), ("Durban", "Cape of Good Hope"),
    ("South of Madagascar", "Cape of Good Hope"),
    # Red Sea / Med
    ("Bab el-Mandeb Strait", "Red Sea"), ("Red Sea", "Jeddah"),
    ("Red Sea", "Suez Canal"), ("Suez Canal", "Eastern Med"),
    ("Eastern Med", "Piraeus"), ("Eastern Med", "Central Med"),
    ("Piraeus", "Central Med"), ("Central Med", "Genoa"),
    ("Central Med", "Gibraltar Strait"),
    # Atlantic / Europe
    ("Gibraltar Strait", "Cape Finisterre"), ("Cape Finisterre", "Ushant"),
    ("Ushant", "Dover Strait"), ("Dover Strait", "Rotterdam"),
    ("Rotterdam", "Hamburg"), ("Gibraltar Strait", "Off Dakar"),
    ("Gibraltar Strait", "Azores"), ("Cape Finisterre", "Azores"),
    ("Off Dakar", "Cape Finisterre"), ("Off Dakar", "Gulf of Guinea"),
    ("Gulf of Guinea", "Lagos"), ("Gulf of Guinea", "South Atlantic"),
    ("Cape of Good Hope", "South Atlantic"), ("South Atlantic", "Off Dakar"),
    ("South Atlantic", "Off Brazil"), ("Off Brazil", "Santos"),
    ("Off Brazil", "Off Dakar"), ("Off Brazil", "Windward Passage"),
    ("Azores", "North Atlantic"), ("North Atlantic", "New York"),
    ("Ushant", "North Atlantic"),
    # Americas
    ("New York", "Florida Strait"), ("New York", "Windward Passage"),
    ("Florida Strait", "Houston"), ("Florida Strait", "Panama Canal"),
    ("Windward Passage", "Panama Canal"), ("Panama Canal", "Los Angeles"),
    # Trans-Pacific (open ocean, great circle is fine)
    ("Shanghai", "Los Angeles"), ("Busan", "Los Angeles"), ("Tokyo", "Los Angeles"),
]

# Extra distance a canal transit adds beyond the straight segment (nm).
CANAL_LENGTH_NM = {"Suez Canal": 105, "Panama Canal": 44}


def haversine_nm(a, b):
    lat1, lon1, _ = NODES[a]
    lat2, lon2, _ = NODES[b]
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 3440.065 * math.asin(math.sqrt(h))


def _build_graph():
    g = {n: {} for n in NODES}
    for a, b in EDGES:
        d = haversine_nm(a, b) + CANAL_LENGTH_NM.get(a, 0) / 2 + CANAL_LENGTH_NM.get(b, 0) / 2
        g[a][b] = d
        g[b][a] = d
    return g


GRAPH = _build_graph()
PORTS = [n for n, v in NODES.items() if v[2] == "port"]
CHOKEPOINTS = [n for n, v in NODES.items() if v[2] == "chokepoint"]


ALIASES = {
    "la": "Los Angeles", "long beach": "Los Angeles", "ny": "New York", "nyc": "New York",
    "dubai": "Jebel Ali", "hong kong": "Shenzhen", "yantian": "Shenzhen",
    "suez": "Suez Canal", "panama": "Panama Canal", "hormuz": "Strait of Hormuz",
    "bab el-mandeb": "Bab el-Mandeb Strait", "bab al-mandab": "Bab el-Mandeb Strait",
    "red sea": "Bab el-Mandeb Strait", "malacca": "Malacca Strait",
    "gibraltar": "Gibraltar Strait", "good hope": "Cape of Good Hope", "cape": "Cape of Good Hope",
}


def resolve(name):
    """Case-insensitive lookup of a port or chokepoint name, with aliases."""
    if name in NODES:
        return name
    low = name.strip().lower()
    if low in ALIASES:
        return ALIASES[low]
    if len(low) < 4:
        return None
    for n in NODES:
        if n.lower() == low:
            return n
    for n in NODES:
        if NODES[n][2] != "waypoint" and (low in n.lower() or n.lower() in low):
            return n
    return None


def alternatives(origin, destination, blocked=(), k=3):
    """Up to k distinct routes: the shortest, then the shortest when each
    chokepoint on an already-found route is additionally avoided."""
    found = {}
    frontier = [tuple(blocked)]
    seen = set()
    while frontier and len(found) < k * 3:
        b = frontier.pop(0)
        if frozenset(b) in seen:
            continue
        seen.add(frozenset(b))
        nm, path = shortest_path(origin, destination, b)
        if path is None:
            continue
        found.setdefault(tuple(path), nm)
        for n in path:
            if NODES[n][2] == "chokepoint" and n not in (origin, destination):
                frontier.append(b + (n,))
    return sorted(((nm, list(p)) for p, nm in found.items()))[:k]


def shortest_path(origin, destination, blocked=()):
    """Dijkstra over GRAPH, skipping blocked nodes. Returns (nm, path) or (None, None)."""
    blocked = set(blocked) - {origin, destination}
    dist = {origin: 0.0}
    prev = {}
    heap = [(0.0, origin)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == destination:
            path = [u]
            while u in prev:
                u = prev[u]
                path.append(u)
            return d, path[::-1]
        if d > dist.get(u, math.inf):
            continue
        for v, w in GRAPH[u].items():
            if v in blocked:
                continue
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return None, None
