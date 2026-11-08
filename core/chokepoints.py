"""Chokepoint registry: the single source of truth for which straits/canals
are open or closed and how risky they are.

State lives in data/chokepoints.json, so anything (the Streamlit UI, an AI
agent, a script, a cron job) can flip a switch and the map picks it up.

Agent / script usage:
    from core.chokepoints import ChokepointRegistry
    reg = ChokepointRegistry()
    reg.set_status("babalmandab", "closed", note="Attacks on shipping")
    reg.set_risk("hormuz", risk_score=0.9, war_risk_premium_pct=1.0)

Command line:
    python -m core.chokepoints list
    python -m core.chokepoints close babalmandab "Attacks on shipping"
    python -m core.chokepoints open babalmandab
"""
import json
import re
import sys
from dataclasses import asdict, dataclass

from . import DATA_DIR

CHOKEPOINTS_FILE = DATA_DIR / "chokepoints.json"
STATUSES = ("open", "closed")


@dataclass
class Chokepoint:
    id: str
    name: str
    lat: float
    lon: float
    radius_km: float = 80
    searoute_passage: str | None = None   # built-in searoute passage name, if any
    status: str = "open"
    risk_score: float = 0.0               # 0 (safe) .. 1 (extreme)
    war_risk_premium_pct: float = 0.0     # % of hull value charged per transit
    toll_usd: float = 0.0                 # base toll for a reference ship
    max_draft_m: float | None = None      # physical limits (e.g. Panama locks)
    max_beam_m: float | None = None
    max_length_m: float | None = None
    note: str = ""

    @property
    def is_open(self) -> bool:
        return self.status == "open"


class ChokepointRegistry:
    def __init__(self, path=CHOKEPOINTS_FILE):
        self.path = path
        self.reload()

    # ---- persistence -------------------------------------------------
    def reload(self):
        with open(self.path, encoding="utf-8") as f:
            self._items = {c["id"]: Chokepoint(**c) for c in json.load(f)}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self._items.values()], f, indent=2)

    # ---- queries ------------------------------------------------------
    def all(self) -> list[Chokepoint]:
        return list(self._items.values())

    def get(self, cp_id: str) -> Chokepoint:
        return self._items[cp_id]

    def closed(self) -> list[Chokepoint]:
        return [c for c in self._items.values() if not c.is_open]

    # ---- mutations (the "switches" an AI agent can flip) -----------
    def set_status(self, cp_id: str, status: str, note: str | None = None):
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        cp = self._items[cp_id]
        cp.status = status
        if note is not None:
            cp.note = note
        self.save()

    def set_risk(self, cp_id: str, risk_score: float | None = None,
                 war_risk_premium_pct: float | None = None):
        cp = self._items[cp_id]
        if risk_score is not None:
            cp.risk_score = max(0.0, min(1.0, float(risk_score)))
        if war_risk_premium_pct is not None:
            cp.war_risk_premium_pct = max(0.0, float(war_risk_premium_pct))
        self.save()

    def add(self, name: str, lat: float, lon: float, radius_km: float = 80,
            risk_score: float = 0.0, **extra) -> Chokepoint:
        cp_id = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "custom"
        base, n = cp_id, 2
        while cp_id in self._items:
            cp_id, n = f"{base}_{n}", n + 1
        cp = Chokepoint(id=cp_id, name=name, lat=lat, lon=lon,
                        radius_km=radius_km, risk_score=risk_score, **extra)
        self._items[cp_id] = cp
        self.save()
        return cp

    def remove(self, cp_id: str):
        self._items.pop(cp_id)
        self.save()


def _cli(argv):
    reg = ChokepointRegistry()
    if not argv or argv[0] == "list":
        for c in reg.all():
            print(f"{c.id:14} {c.status:7} risk={c.risk_score:.2f}  {c.name}")
    elif argv[0] in ("open", "close") and len(argv) >= 2:
        reg.set_status(argv[1], "open" if argv[0] == "open" else "closed",
                       note=argv[2] if len(argv) > 2 else None)
        print(f"{argv[1]} -> {reg.get(argv[1]).status}")
    else:
        print(__doc__)


if __name__ == "__main__":
    _cli(sys.argv[1:])
