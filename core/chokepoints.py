"""Chokepoint registry: the single source of truth for which straits/canals
are open or closed and how risky they are.

Two files:
- data/chokepoints.json: reference data (positions, limits, tolls, default
  status). Tracked in git and never written at runtime; edit it by hand.
- data/chokepoints.state.json: what has been changed at runtime (flipped
  switches, new risk values, custom chokepoints added or removed), stored
  as a diff against the reference data. Git-ignored, so trying out a
  scenario never shows up as a change to commit. Delete it, or run
  `python -m core.chokepoints reset`, to go back to the reference data.

Anything (an AI agent, a script, a cron job) can flip a switch through the
registry, and every new registry picks it up.

Agent / script usage:
    from core.chokepoints import ChokepointRegistry
    reg = ChokepointRegistry()
    reg.set_status("babalmandab", "closed", note="Attacks on shipping")
    reg.set_risk("hormuz", risk_score=0.9, war_risk_premium_pct=1.0)
    reg.reset()  # back to the reference data

Command line:
    python -m core.chokepoints list
    python -m core.chokepoints close babalmandab "Attacks on shipping"
    python -m core.chokepoints open babalmandab
    python -m core.chokepoints reset
"""
import json
import pathlib
import re
import sys
from dataclasses import asdict, dataclass, fields

from . import DATA_DIR

CHOKEPOINTS_FILE = DATA_DIR / "chokepoints.json"
STATE_FILE = DATA_DIR / "chokepoints.state.json"
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
    def __init__(self, path=CHOKEPOINTS_FILE, state_path=STATE_FILE):
        self.path = pathlib.Path(path)              # reference data, read only
        self.state_path = pathlib.Path(state_path)  # runtime changes, written by save()
        self.reload()

    # ---- persistence -------------------------------------------------
    def _load_reference(self) -> dict:
        with open(self.path, encoding="utf-8") as f:
            return {c["id"]: c for c in json.load(f)}

    def reload(self):
        items = self._load_reference()
        try:
            with open(self.state_path, encoding="utf-8") as f:
                state = json.load(f)
        except FileNotFoundError:
            state = {}
        known = {f.name for f in fields(Chokepoint)}
        for cp_id, changes in state.get("changed", {}).items():
            if cp_id in items:  # skip changes to chokepoints since dropped from the reference
                items[cp_id] = {**items[cp_id], **{k: v for k, v in changes.items() if k in known}}
        for cp_id in state.get("removed", []):
            items.pop(cp_id, None)
        for c in state.get("added", []):
            items[c["id"]] = {k: v for k, v in c.items() if k in known}
        self._items = {cp_id: Chokepoint(**c) for cp_id, c in items.items()}

    def save(self):
        """Write the difference from the reference data to the state file."""
        reference = self._load_reference()
        changed, added = {}, []
        for cp_id, cp in self._items.items():
            current = asdict(cp)
            if cp_id not in reference:
                added.append(current)
                continue
            ref = asdict(Chokepoint(**reference[cp_id]))  # fills defaults for missing keys
            diff = {k: v for k, v in current.items() if v != ref[k]}
            if diff:
                changed[cp_id] = diff
        removed = [cp_id for cp_id in reference if cp_id not in self._items]
        if not (changed or added or removed):
            self.state_path.unlink(missing_ok=True)
            return
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({"changed": changed, "added": added, "removed": removed}, f, indent=2)

    def reset(self):
        """Drop all runtime changes and go back to the reference data."""
        self.state_path.unlink(missing_ok=True)
        self.reload()

    def changed_ids(self) -> set[str]:
        """Ids that differ from the reference data (changed or added)."""
        reference = self._load_reference()
        return {cp_id for cp_id, cp in self._items.items()
                if cp_id not in reference or asdict(cp) != asdict(Chokepoint(**reference[cp_id]))}

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
        changed = reg.changed_ids()
        for c in reg.all():
            mark = "*" if c.id in changed else " "
            print(f"{mark} {c.id:14} {c.status:7} risk={c.risk_score:.2f}  {c.name}")
        if changed:
            print(f"\n* changed at runtime (saved in {reg.state_path.name}; "
                  "`python -m core.chokepoints reset` to undo)")
    elif argv[0] in ("open", "close") and len(argv) >= 2:
        reg.set_status(argv[1], "open" if argv[0] == "open" else "closed",
                       note=argv[2] if len(argv) > 2 else None)
        print(f"{argv[1]} -> {reg.get(argv[1]).status}")
    elif argv[0] == "reset":
        reg.reset()
        print("All chokepoints back to the reference data.")
    else:
        print(__doc__)


if __name__ == "__main__":
    _cli(sys.argv[1:])
