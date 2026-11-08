"""Voyage cost and risk estimates. Deliberately simple and transparent;
replace pieces with ML models / live APIs later."""
from dataclasses import dataclass

from .routing import RouteResult
from .ships import ShipType


@dataclass
class VoyageEstimate:
    distance_nm: float
    days: float
    fuel_t: float
    fuel_cost: float
    time_cost: float
    tolls: float
    war_risk_insurance: float
    total_cost: float
    risk_score: float  # 0..1 combined probability-style score

    @property
    def risk_label(self) -> str:
        if self.risk_score >= 0.6:
            return "High"
        if self.risk_score >= 0.3:
            return "Medium"
        return "Low"


def estimate_voyage(route: RouteResult, ship: ShipType, speed_kn: float,
                    bunker_usd_per_t: float) -> VoyageEstimate:
    days = route.distance_nm / (speed_kn * 24)
    fuel_t = ship.fuel_per_day_at(speed_kn) * days
    fuel_cost = fuel_t * bunker_usd_per_t
    time_cost = ship.daily_cost_usd * days
    tolls = sum(cp.toll_usd for cp in route.traversed) * ship.toll_factor
    war_risk = sum(cp.war_risk_premium_pct / 100 for cp in route.traversed) * ship.hull_value_usd

    # Combined risk: chance that at least one chokepoint "goes wrong"
    safe = 1.0
    for cp in route.traversed:
        safe *= 1 - cp.risk_score
    risk = 1 - safe

    total = fuel_cost + time_cost + tolls + war_risk
    return VoyageEstimate(route.distance_nm, days, fuel_t, fuel_cost, time_cost,
                          tolls, war_risk, total, risk)
