"""Ship types and their attributes. `to_features()` gives a numeric vector
that a future ML model can consume."""
import json
from dataclasses import asdict, dataclass

from . import DATA_DIR

SHIPS_FILE = DATA_DIR / "ships.json"


@dataclass
class ShipType:
    id: str
    name: str
    category: str            # container / bulk / tanker / gas
    dwt_t: float             # deadweight tonnage
    length_m: float
    beam_m: float            # width
    draft_m: float           # depth below waterline (fully laden)
    teu: int                 # container capacity (0 if not a container ship)
    service_speed_kn: float  # normal cruising speed
    fuel_t_per_day: float    # fuel burn at service speed
    daily_cost_usd: float    # charter / operating cost per day
    hull_value_usd: float    # used for war-risk insurance premiums
    toll_factor: float       # multiplier on canal tolls (bigger ship = higher toll)

    def fuel_per_day_at(self, speed_kn: float) -> float:
        """Fuel burn scales roughly with the cube of speed."""
        return self.fuel_t_per_day * (speed_kn / self.service_speed_kn) ** 3

    def fits(self, max_draft=None, max_beam=None, max_length=None) -> bool:
        return not (
            (max_draft is not None and self.draft_m > max_draft)
            or (max_beam is not None and self.beam_m > max_beam)
            or (max_length is not None and self.length_m > max_length)
        )

    def to_features(self) -> dict:
        """Numeric features for ML models."""
        return {k: v for k, v in asdict(self).items() if isinstance(v, (int, float))}


def load_ships() -> list[ShipType]:
    with open(SHIPS_FILE, encoding="utf-8") as f:
        return [ShipType(**s) for s in json.load(f)]
