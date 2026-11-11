"""Runtime chokepoint changes go to a git-ignored state file layered on the
tracked reference data, which is never written.

Run:  python -m unittest discover -s tests -t .
"""
import json
import pathlib
import shutil
import tempfile
import unittest

from core.chokepoints import CHOKEPOINTS_FILE, ChokepointRegistry


class ChokepointState(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.ref = self.dir / "chokepoints.json"
        self.state = self.dir / "chokepoints.state.json"
        shutil.copy(CHOKEPOINTS_FILE, self.ref)
        self.ref_bytes = self.ref.read_bytes()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def registry(self):
        return ChokepointRegistry(self.ref, self.state)

    def assert_reference_untouched(self):
        self.assertEqual(self.ref.read_bytes(), self.ref_bytes)

    def test_changes_persist_without_touching_reference(self):
        reg = self.registry()
        reg.set_status("suez", "closed", note="Test closure")
        reg.set_risk("hormuz", risk_score=0.9, war_risk_premium_pct=1.0)
        self.assert_reference_untouched()

        again = self.registry()
        self.assertEqual(again.get("suez").status, "closed")
        self.assertEqual(again.get("suez").note, "Test closure")
        self.assertEqual(again.get("hormuz").risk_score, 0.9)
        self.assertEqual(again.changed_ids(), {"suez", "hormuz"})

        state = json.loads(self.state.read_text())
        self.assertEqual(state["changed"]["suez"], {"status": "closed", "note": "Test closure"})

    def test_reverting_a_change_removes_the_state_file(self):
        reg = self.registry()
        original = reg.get("suez").status
        reg.set_status("suez", "closed" if original == "open" else "open")
        self.assertTrue(self.state.exists())
        reg.set_status("suez", original)
        self.assertFalse(self.state.exists())

    def test_add_and_remove_custom_chokepoints(self):
        reg = self.registry()
        cp = reg.add("Danish Straits", 55.6, 12.7, radius_km=60, risk_score=0.2)
        reg.remove("taiwan")
        self.assert_reference_untouched()

        again = self.registry()
        self.assertEqual(again.get(cp.id).name, "Danish Straits")
        with self.assertRaises(KeyError):
            again.get("taiwan")

    def test_reset(self):
        reg = self.registry()
        reg.set_status("suez", "closed")
        reg.add("Test strait", 1.0, 2.0)
        reg.reset()
        self.assertFalse(self.state.exists())
        self.assertEqual(reg.changed_ids(), set())
        self.assertEqual([c.id for c in reg.all()], [c.id for c in self.registry().all()])

    def test_reference_updates_still_apply_under_runtime_changes(self):
        reg = self.registry()
        reg.set_status("suez", "closed")
        data = json.loads(self.ref.read_text())
        next(c for c in data if c["id"] == "suez")["toll_usd"] = 123
        self.ref.write_text(json.dumps(data))

        again = self.registry()
        self.assertEqual(again.get("suez").toll_usd, 123)    # from the updated reference
        self.assertEqual(again.get("suez").status, "closed")  # from the state file

    def test_stale_or_unknown_entries_in_state_are_ignored(self):
        self.state.write_text(json.dumps({
            "changed": {"gone": {"status": "closed"}, "suez": {"status": "closed", "bogus": 1}},
            "removed": ["also_gone"],
            "added": [{"id": "x", "name": "X", "lat": 0, "lon": 0, "future_field": 1}],
        }))
        reg = self.registry()
        self.assertEqual(reg.get("suez").status, "closed")
        self.assertEqual(reg.get("x").name, "X")


if __name__ == "__main__":
    unittest.main()
