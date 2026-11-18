"""Voyage cost model. The agent's compare_routes tool calls voyage_cost() for
every route, so the LLM never calculates a cost itself: it only picks the
ports, the ship type and the scenario.

Inputs: a ship type (data/ships.json) and the price assumptions
(data/cost_assumptions.json, each one with a source).

    total = fuel + time (charter) + canal tolls + war-risk insurance + EU ETS

    fuel        sea days x fuel/day at service speed x (speed / service speed)^3 x fuel price
    time        (sea days + canal waiting days) x charter rate per day
    canal tolls base toll of each canal x the ship's toll_factor
    war risk    hull value x premium % for each risky chokepoint passed
    EU ETS      CO2 x share covered (100% EU-EU, 50% EU-non-EU) x carbon price

Port fees and cargo handling are left out: they are the same on every route
between two ports, so they don't change which route is cheaper.
"""
import json
from dataclasses import asdict, dataclass
from functools import lru_cache

from . import DATA_DIR
from .ships import ShipType, load_ships

ASSUMPTIONS_FILE = DATA_DIR / "cost_assumptions.json"
CHOKEPOINTS_FILE = DATA_DIR / "chokepoints.json"

# ISO 3166 alpha-2 codes of EU member states (plus Norway and Iceland, which
# apply the EU ETS for shipping too).
EU_ETS_COUNTRIES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
    "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "NO", "IS",
}


@lru_cache(maxsize=1)
def load_assumptions() -> dict:
    with open(ASSUMPTIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def ship_types() -> dict[str, ShipType]:
    return {s.id: s for s in load_ships()}


@lru_cache(maxsize=1)
def _passage_limits() -> dict[str, dict]:
    """searoute passage -> size limits, from data/chokepoints.json."""
    with open(CHOKEPOINTS_FILE, encoding="utf-8") as f:
        return {c["searoute_passage"]: {"max_draft": c.get("max_draft_m"), "max_beam": c.get("max_beam_m"),
                                        "max_length": c.get("max_length_m")}
                for c in json.load(f) if c.get("searoute_passage")}


def too_big_for(ship: ShipType, passages: dict[str, str]) -> list[str]:
    """Chokepoints (name -> searoute passage) the ship doesn't fit through,
    e.g. a laden VLCC is too deep for Suez, a 400 m ship too long for Panama."""
    limits = _passage_limits()
    return [name for name, p in passages.items() if p in limits and not ship.fits(**limits[p])]


def eu_ets_share(origin_country: str, destination_country: str) -> float:
    """Share of the voyage's CO2 that needs EU ETS allowances."""
    a = load_assumptions()["eu_ets"]
    n = (origin_country in EU_ETS_COUNTRIES) + (destination_country in EU_ETS_COUNTRIES)
    return {2: a["share_both_ports_eu"], 1: a["share_one_port_eu"], 0: 0.0}[n]


@dataclass
class VoyageCost:
    speed_knots: float
    sea_days: float
    canal_waiting_days: float
    fuel_t: float
    co2_t: float
    war_risk_chokepoints: list[str]
    eu_ets_share: float
    fuel: float
    charter_time: float
    canal_tolls: float
    war_risk_insurance: float
    eu_ets: float
    total: float
    per_unit: float          # total per TEU of capacity (container ships) or per tonne deadweight
    per_unit_label: str

    def cost_usd(self) -> dict:
        return {k: round(getattr(self, k)) for k in
                ("fuel", "charter_time", "canal_tolls", "war_risk_insurance", "eu_ets", "total")}

    def as_dict(self) -> dict:
        return asdict(self)


def voyage_cost(distance_nm: float, chokepoints: list[str], ship: ShipType,
                speed_knots: float | None = None, fuel_price_usd_per_t: float | None = None,
                flagged_war_risk: tuple[str, ...] = (), war_risk_premium_pct: float | None = None,
                ets_share: float = 0.0) -> VoyageCost:
    """Cost of one voyage along a route that passes `chokepoints` (names as in
    agent/network.py CHOKEPOINTS).

    speed_knots: defaults to the ship's service speed.
    fuel_price_usd_per_t: defaults to the assumption file.
    flagged_war_risk: chokepoints the scenario treats as dangerous but open. They
        and the chokepoints with a current premium in the assumption file are charged.
    war_risk_premium_pct: overrides the premium for every charged chokepoint.
    ets_share: from eu_ets_share() for the two ports.
    """
    a = load_assumptions()
    speed = speed_knots or ship.service_speed_kn
    fuel_price = fuel_price_usd_per_t or a["fuel_price_usd_per_t"]

    # time and fuel
    sea_days = distance_nm / (speed * 24)
    canals = [a["canals"][c] for c in chokepoints if c in a["canals"]]
    waiting_days = sum(c["waiting_days"] for c in canals)
    fuel_t = ship.fuel_per_day_at(speed) * sea_days
    co2_t = fuel_t * a["co2_t_per_t_fuel"]

    # money
    fuel = fuel_t * fuel_price
    charter_time = (sea_days + waiting_days) * ship.daily_cost_usd
    canal_tolls = sum(c["base_toll_usd"] for c in canals) * ship.toll_factor

    premiums = a["war_risk_premium_pct"]
    risky = [c for c in chokepoints if c in premiums or c in flagged_war_risk]
    war_risk = sum(ship.hull_value_usd * (war_risk_premium_pct if war_risk_premium_pct is not None
                                          else premiums.get(c, a["war_risk_premium_pct_default"])) / 100
                   for c in risky)

    eu_ets = co2_t * ets_share * a["eu_ets"]["usd_per_t_co2"]
    total = fuel + charter_time + canal_tolls + war_risk + eu_ets

    if ship.teu:
        per_unit, label = total / ship.teu, "USD per TEU of capacity"
    else:
        per_unit, label = total / ship.dwt_t, "USD per tonne deadweight"

    return VoyageCost(speed_knots=speed, sea_days=sea_days, canal_waiting_days=waiting_days,
                      fuel_t=fuel_t, co2_t=co2_t, war_risk_chokepoints=risky, eu_ets_share=ets_share,
                      fuel=fuel, charter_time=charter_time, canal_tolls=canal_tolls,
                      war_risk_insurance=war_risk, eu_ets=eu_ets, total=total,
                      per_unit=per_unit, per_unit_label=label)


def assumptions_used(ship: ShipType, speed_knots=None, fuel_price_usd_per_t=None,
                     war_risk_premium_pct=None) -> dict:
    """The parameters behind a cost figure, for the agent to cite."""
    a = load_assumptions()
    return {
        "ship": {"type": ship.id, "name": ship.name, "service_speed_kn": ship.service_speed_kn,
                 "fuel_t_per_day_at_service_speed": ship.fuel_t_per_day,
                 "charter_usd_day": ship.daily_cost_usd, "hull_value_usd": ship.hull_value_usd,
                 "toll_factor": ship.toll_factor},
        "speed_knots": speed_knots or ship.service_speed_kn,
        "fuel_price_usd_per_t": fuel_price_usd_per_t or a["fuel_price_usd_per_t"],
        "canals": a["canals"],
        "war_risk_premium_pct": ({"all charged chokepoints": war_risk_premium_pct}
                                 if war_risk_premium_pct is not None else
                                 {**a["war_risk_premium_pct"], "flagged by scenario": a["war_risk_premium_pct_default"]}),
        "eu_ets_usd_per_t_co2": a["eu_ets"]["usd_per_t_co2"],
        "sources": a["sources"],
        "note": "Sea and canal time only; excludes port fees and time, which are equal on every route "
                "between the same two ports.",
    }
