"""Sea routing between any two points, avoiding closed chokepoints.

Built on the `searoute` maritime network graph. Chokepoints that map to a
built-in searoute passage are blocked via its `restrictions`; custom
chokepoints are blocked by tagging network edges as a restricted passage:
the edges crossing their `gate` line if they have one, otherwise every edge
inside their radius.
"""
import copy
import math
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import searoute as sr
from searoute.searoute import setup_M

from .chokepoints import Chokepoint
from .ships import ShipType

EARTH_RADIUS_KM = 6371.0
NM_PER_KM = 1 / 1.852


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass
class RouteResult:
    found: bool
    coords: list = field(default_factory=list)   # [(lat, lon), ...] unwrapped for drawing
    distance_nm: float = 0.0
    traversed: list = field(default_factory=list)  # Chokepoint objects passed through
    blocked_by: list = field(default_factory=list)  # Chokepoints avoided (closed / too small)
    message: str = ""


def _custom_tag(cp: Chokepoint) -> str:
    return f"custom::{cp.id}"


def _gate(cp: Chokepoint):
    """The chokepoint's gate as a hashable tuple of (lat, lon) points, or None."""
    return tuple((float(lat), float(lon)) for lat, lon in cp.gate) if cp.gate else None


def _orient(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_cross(p1, p2, q1, q2) -> bool:
    """Whether segments p1-p2 and q1-q2 intersect (touching counts), treating
    (lat, lon) as plane coordinates. Fine for gates a few hundred km long."""
    if (max(p1[0], p2[0]) < min(q1[0], q2[0]) or max(q1[0], q2[0]) < min(p1[0], p2[0])
            or max(p1[1], p2[1]) < min(q1[1], q2[1]) or max(q1[1], q2[1]) < min(p1[1], p2[1])):
        return False
    return (_orient(q1, q2, p1) * _orient(q1, q2, p2) <= 0
            and _orient(p1, p2, q1) * _orient(p1, p2, q2) <= 0)


def _crosses_gate(gate, a, b) -> bool:
    """Whether segment a-b ((lat, lon) points) crosses the gate polyline."""
    return any(_segments_cross(a, b, g1, g2) for g1, g2 in zip(gate, gate[1:]))


@lru_cache(maxsize=16)
def _network_with_custom_blocks(blocks: tuple):
    """Copy of the searoute network with edges blocked by custom chokepoints
    tagged. `blocks` = ((tag, gate, lat, lon, radius_km), ...): with a gate,
    edges crossing it are tagged; without, edges with an end inside the
    radius. Cached per combination.

    searoute keeps separate edge data for each direction, and edges() yields
    only one, so the tag is written to both: otherwise the route can still
    pass the "blocked" edge the other way.
    """
    M = copy.deepcopy(setup_M())
    hits = {}
    for u, v in M.edges():
        (x1, y1), (x2, y2) = u, v
        for tag, gate, lat, lon, radius in blocks:
            if gate:  # edges spanning the antimeridian are never near a gate
                hit = abs(x1 - x2) < 180 and _crosses_gate(gate, (y1, x1), (y2, x2))
            else:
                hit = any(haversine_km(lat, lon, y, x) <= radius for (x, y) in (u, v))
            if hit:
                hits[(u, v)] = tag
                break
    for (u, v), tag in hits.items():
        for a, b in ((u, v), (v, u)):
            data = M.get_edge_data(a, b)
            if data is not None:
                data["passage"] = tag
    return M


def _unwrap(lonlat_coords):
    """Shift longitudes so the line never jumps across the map at +/-180."""
    out, prev = [], None
    for lon, lat in lonlat_coords:
        if prev is not None:
            while lon - prev > 180:
                lon -= 360
            while lon - prev < -180:
                lon += 360
        out.append((lat, lon))
        prev = lon
    return out


def _route_passes(cp: Chokepoint, passages: list, coords) -> bool:
    if cp.searoute_passage:
        return cp.searoute_passage in passages
    gate = _gate(cp)
    if gate:
        for a, b in zip(coords, coords[1:]):
            # coords are unwrapped; shift each segment back near [-180, 180)
            shift = -360 * math.floor((a[1] + 180) / 360)
            if _crosses_gate(gate, (a[0], a[1] + shift), (b[0], b[1] + shift)):
                return True
        return False
    return any(haversine_km(cp.lat, cp.lon, lat, ((lon + 180) % 360) - 180) <= cp.radius_km
               for lat, lon in coords)


def compute_route(origin: dict, destination: dict, chokepoints: list[Chokepoint],
                  ship: ShipType | None = None, respect_closures: bool = True) -> RouteResult:
    """origin/destination: dicts with 'lat' and 'lon'.
    respect_closures=False gives the 'normal' route (ignores closed status,
    but still respects physical ship-size limits)."""
    blocked = []
    for cp in chokepoints:
        closed = respect_closures and not cp.is_open
        too_big = ship is not None and not ship.fits(cp.max_draft_m, cp.max_beam_m, cp.max_length_m)
        if closed or too_big:
            blocked.append(cp)

    restrictions = ["northwest"]  # Northwest Passage is ice-bound; always avoid
    custom = []
    for cp in blocked:
        if cp.searoute_passage:
            restrictions.append(cp.searoute_passage)
        else:
            restrictions.append(_custom_tag(cp))
            custom.append((_custom_tag(cp), _gate(cp), cp.lat, cp.lon, cp.radius_km))
    M = _network_with_custom_blocks(tuple(sorted(custom))) if custom else None

    o = [origin["lon"], origin["lat"]]
    d = [destination["lon"], destination["lat"]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            feat = sr.searoute(o, d, units="naut", restrictions=restrictions, M=M,
                               append_orig_dest=True, return_passages=True)
        except Exception as e:  # noqa: BLE001
            return RouteResult(False, blocked_by=blocked, message=f"Routing error: {e}")

    length = feat.properties.get("length") or 0
    if length <= 0 or any("No path" in str(w.message) for w in caught):
        names = ", ".join(c.name for c in blocked) or "none"
        return RouteResult(False, blocked_by=blocked,
                           message=f"No sea route found. Blocked chokepoints: {names}. "
                                   "The destination may be unreachable (e.g. a Persian Gulf "
                                   "port with Hormuz closed).")

    coords = _unwrap(feat.geometry["coordinates"])
    passages = feat.properties.get("traversed_passages") or []
    traversed = [cp for cp in chokepoints
                 if cp not in blocked and _route_passes(cp, passages, coords)]
    return RouteResult(True, coords, float(length), traversed, blocked)
