"""Cross-reference the editable practice bank and runnable exercises."""

import csv
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "practice_bank"


def rows(name):
    with (BANK / f"{name}.tsv").open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="|"))


class PracticeBankTests(unittest.TestCase):
    def test_ids_references_and_dojo_links(self):
        drills, sources, sessions = (rows(name) for name in ("drills", "sources", "sessions"))
        self.assertEqual((len(drills), len(sources), len(sessions)), (103, 8, 10))
        drill_ids = {r["id"] for r in drills}
        source_ids = {r["id"] for r in sources}
        self.assertEqual(len(drill_ids), 103)
        self.assertEqual(len(source_ids), 8)
        handbook = (BANK / "handbook.html").read_text()
        handbook_ids = set(re.findall(r'<article class="problem" id="([A-Z]{2}\d{2})"', handbook))
        self.assertEqual(handbook_ids, drill_ids)
        for r in drills:
            self.assertIn(r["priority"], {"P0", "P1", "P2"})
            self.assertGreater(int(r["minutes"]), 0)
            for source in filter(None, r["source"].split(",")):
                self.assertIn(source, source_ids)
            if r["dojo"]:
                self.assertTrue((ROOT / "dojo" / r["dojo"]).is_file())
                self.assertTrue((ROOT / "dojo" / r["dojo"].replace(".py", "_eval.py")).is_file())
        for r in sessions:
            self.assertEqual(int(r["minutes"]), 90)
            self.assertTrue(set(r["drills"].split(",")) <= drill_ids)
        self.assertEqual(rows("attempts"), [])
