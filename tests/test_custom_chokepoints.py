"""Custom (geometry-based) chokepoints: closing one must block transit through
it without cutting off the ports along its coasts.

Run:  python -m unittest discover -s tests -t .
"""
import dataclasses
import unittest

from core.chokepoints import ChokepointRegistry
from core.ports import get_port
from core.routing import _crosses_gate, _gate, compute_route


def scenario(closed_ids=()):
    """All registry chokepoints, open except `closed_ids` (no file writes)."""
    return [dataclasses.replace(cp, status="closed" if cp.id in closed_ids else "open")
            for cp in ChokepointRegistry().all()]


def route(origin, destination, closed_ids=()):
    return compute_route(get_port(origin), get_port(destination), scenario(closed_ids))


def crosses(cp, result):
    gate = _gate(cp)
    return any(_crosses_gate(gate, a, b) for a, b in zip(result.coords, result.coords[1:]))


class TaiwanStraitGate(unittest.TestCase):
    taiwan = ChokepointRegistry().get("taiwan")

    def test_ports_on_the_strait_stay_reachable_when_closed(self):
        # These used to sit inside the 150 km blocking circle (or snap to network
        # nodes inside it), so closing the strait made them unreachable.
        for port in ("TWTXG", "CNXMN", "CNQZJ", "TWMAL"):  # Taichung, Xiamen, Quanzhou, Mailiao
            with self.subTest(port=port):
                r = route(port, "KRPUS", closed_ids=("taiwan",))
                self.assertTrue(r.found, r.message)
                self.assertFalse(crosses(self.taiwan, r))

    def test_closing_forces_transit_east_of_taiwan(self):
        normal = route("CNSHA", "HKHKG")
        self.assertIn("taiwan", [cp.id for cp in normal.traversed])
        detour = route("CNSHA", "HKHKG", closed_ids=("taiwan",))
        self.assertTrue(detour.found, detour.message)
        self.assertNotIn("taiwan", [cp.id for cp in detour.traversed])
        self.assertFalse(crosses(self.taiwan, detour))
        self.assertGreater(detour.distance_nm, normal.distance_nm)
        self.assertGreater(max(lon for _, lon in detour.coords), 121.5)  # Pacific side

    def test_routes_inside_the_strait_are_not_counted_as_transits(self):
        r = route("TWKHH", "CNXMN")  # Kaohsiung - Xiamen, south of the gate
        self.assertTrue(r.found, r.message)
        self.assertNotIn("taiwan", [cp.id for cp in r.traversed])


class RadiusFallback(unittest.TestCase):
    def test_chokepoint_without_gate_still_blocks_its_radius(self):
        cps = scenario()
        cps.append(dataclasses.replace(cps[0], id="test_disk", name="Test disk", lat=24.3,
                                       lon=119.6, radius_km=150, searoute_passage=None,
                                       status="closed", gate=None, max_draft_m=None,
                                       max_beam_m=None, max_length_m=None))
        r = compute_route(get_port("CNSHA"), get_port("HKHKG"), cps)
        self.assertTrue(r.found, r.message)
        self.assertIn("test_disk", [cp.id for cp in r.blocked_by])
        self.assertGreater(max(lon for _, lon in r.coords), 121.5)


class SegmentsCross(unittest.TestCase):
    def test_basic_cases(self):
        gate = ((0.0, 0.0), (0.0, 10.0))
        self.assertTrue(_crosses_gate(gate, (-1.0, 5.0), (1.0, 5.0)))
        self.assertFalse(_crosses_gate(gate, (1.0, 5.0), (2.0, 5.0)))       # same side
        self.assertFalse(_crosses_gate(gate, (-1.0, 11.0), (1.0, 11.0)))    # past the end
        self.assertFalse(_crosses_gate(gate, (-1.0, 0.0), (-1.0, 10.0)))    # parallel


if __name__ == "__main__":
    unittest.main()
