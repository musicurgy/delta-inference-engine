#!/usr/bin/env python3
"""The Setup tab's "This PC's health" card: windows_tips.py's tips with their fixes, saved for setup_panel.py's
catalog.  Run: python -m unittest tools.test_pc_health"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent / "v41" / "serve"))
import setup_panel  # noqa: E402
import windows_tips  # noqa: E402


class PcHealthTest(unittest.TestCase):
    def test_each_tip_gets_its_fix(self):
        with mock.patch.object(windows_tips, "page_file", return_value=None):
            items = windows_tips.health(["power plan is Balanced: switch to High performance (powercfg.cpl)",
                                         "no 'Lock pages in memory' right (large pages for the expert RAM)",
                                         "something new"])
        self.assertIn("powercfg.cpl", items[0]["how"])
        self.assertIn("secpol.msc", items[1]["how"])
        self.assertIn("README.md", items[2]["how"])

    def test_small_page_file_is_a_tip(self):
        import drive_check
        with mock.patch.object(drive_check, "memory", return_value=(64 * 2**30, 65 * 2**30)):
            self.assertIn("page file is 1 GB", windows_tips.page_file()["text"])
        with mock.patch.object(drive_check, "memory", return_value=(64 * 2**30, 96 * 2**30)):
            self.assertIsNone(windows_tips.page_file())

    def test_saved_and_read_back(self):
        with tempfile.TemporaryDirectory() as t:
            path = os.path.join(t, "local", "windows-health.json")
            with mock.patch.object(windows_tips, "HEALTH", path), mock.patch.object(setup_panel, "HERE", t):
                self.assertIsNone(setup_panel.pc_health())            # not checked yet
                windows_tips.save_health([{"text": "a", "how": "b"}, {"bad": 1}])
                self.assertEqual(setup_panel.pc_health(), [{"text": "a", "how": "b"}])
                windows_tips.save_health([])
                self.assertEqual(setup_panel.pc_health(), [])
                self.assertIn("checked", json.loads(Path(path).read_text()))


if __name__ == "__main__":
    unittest.main()
