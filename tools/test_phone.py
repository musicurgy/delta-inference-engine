#!/usr/bin/env python3
"""The Setup tab's "Use from your phone" switch and auto-pause setting: setup_panel.py writes delta-settings.json,
ds41_local.py turns it into host + api_key (+ auto_pause_mib) in this PC's config, and the page reads it back.
No server, no firewall change.  Run: python -m unittest tools.test_phone"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent / "v41" / "serve"))
import ds41_local  # noqa: E402
import setup_panel  # noqa: E402


class PhoneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = os.path.join(self.tmp.name, "delta-settings.json")
        self.local = os.path.join(self.tmp.name, "ds41-windows.json")
        self.patches = [mock.patch.object(setup_panel, "SETTINGS", self.settings),
                        mock.patch.object(setup_panel, "LOCAL_CONFIG", self.local),
                        mock.patch.object(setup_panel, "catalog", lambda m=None: {"models": [], "helper_gpu": None}),
                        mock.patch.object(setup_panel, "firewall_rule", lambda p=None: False),
                        mock.patch.object(setup_panel, "lan_ips", lambda: ["192.168.1.23"])]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_switch_on_makes_a_key_once(self):
        s = setup_panel.apply({"phone": True})
        key = s["phone_key"]
        self.assertGreaterEqual(len(key), 30)
        self.assertEqual(setup_panel.apply({"phone": False})["phone_key"], key)     # kept for the next time
        self.assertEqual(setup_panel.apply({"phone": True})["phone_key"], key)
        self.assertTrue(setup_panel.needs_restart({"phone": True}))
        self.assertFalse(setup_panel.needs_restart({"auto_pause_mib": 1024}))

    def test_config_gets_host_and_key(self):
        cfg = {"host": "127.0.0.1"}
        ds41_local.page_access(cfg, {"phone": True, "phone_key": "k" * 32, "auto_pause_mib": 2048})
        self.assertEqual((cfg["host"], cfg["api_key"], cfg["auto_pause_mib"]), ("0.0.0.0", "k" * 32, 2048))
        own = {"api_key": "mine", "host": "192.168.1.5"}                         # the user's own config wins
        ds41_local.page_access(own, {"phone": True, "phone_key": "k" * 32})
        self.assertEqual((own["host"], own["api_key"]), ("192.168.1.5", "mine"))
        off = {}
        ds41_local.page_access(off, {"phone": False, "phone_key": "k" * 32, "auto_pause_mib": 0})
        self.assertEqual(off, {})                                                 # this PC only, no key added

    def test_page_reads_the_state(self):
        st = setup_panel.phone_state({})
        self.assertFalse(st["on"])
        self.assertEqual(st["key"], "")
        Path(self.local).write_text(json.dumps({"host": "0.0.0.0", "api_key": "abc,def"}))
        st = setup_panel.phone_state({"phone": True})
        self.assertTrue(st["on"])
        self.assertEqual((st["key"], st["addresses"], st["own_config"]), ("abc", ["192.168.1.23"], False))
        self.assertIn("-Profile Private", st["firewall_add"])
        self.assertTrue(setup_panel.phone_state({})["own_config"])               # set by hand in the config

    def test_auto_pause_values(self):
        self.assertEqual(setup_panel.apply({"auto_pause_mib": 2048})["auto_pause_mib"], 2048)
        with self.assertRaises(ValueError):
            setup_panel.apply({"auto_pause_mib": 3})


if __name__ == "__main__":
    unittest.main()
