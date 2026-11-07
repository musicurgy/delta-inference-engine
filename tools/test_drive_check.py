#!/usr/bin/env python3
"""Tests for tools/v41/serve/drive_check.py (the drive and fit check before the first download): no PowerShell, no
network.  Run: python -m unittest tools.test_drive_check"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "v41" / "serve"))
import drive_check as D  # noqa: E402

GB = 1e9


def drive(letter, bus, media, free_gb, kind="fixed"):
    return {"letter": letter, "kind": kind, "bus": bus, "media": media, "name": "", "free": free_gb * GB,
            "total": 2000 * GB, "rank": D.rank(bus, media, kind)}


class DriveCheckTest(unittest.TestCase):
    def test_parse_powershell(self):
        types, auto = D.parse_ps("DRIVE|C|NVMe|SSD|Samsung 990\r\nDRIVE|H|USB|HDD|One|Touch\r\nPAGEFILE|True\r\n")
        self.assertEqual(types["C"], ("NVMe", "SSD", "Samsung 990"))
        self.assertEqual(types["H"], ("USB", "HDD", "One|Touch"))
        self.assertTrue(auto)
        self.assertEqual(D.parse_ps(""), ({}, None))

    def test_rank_and_words(self):
        self.assertEqual(D.rank("NVMe", "SSD"), 4)
        self.assertEqual(D.rank("SATA", "SSD"), 3)
        self.assertEqual(D.rank("RAID", "Unspecified"), 2)
        self.assertEqual(D.rank("SATA", "HDD"), 1)
        self.assertEqual(D.rank("USB", "SSD"), 1)
        self.assertEqual(D.kind_words(drive("H", "USB", "HDD", 1)), "USB hard disk")
        self.assertEqual(D.kind_words(drive("D", "SATA", "SSD", 1)), "SATA SSD")

    def test_recommend(self):
        c, d, e = drive("C", "NVMe", "SSD", 500), drive("D", "NVMe", "SSD", 900), drive("E", "SATA", "SSD", 2000)
        self.assertIs(D.recommend([c, d, e], "C"), c)          # the current drive is as good as the best: kept
        self.assertIs(D.recommend([c, d, e], "E"), d)          # an NVMe drive beats the SATA one it is on
        full = drive("C", "NVMe", "SSD", 50)
        self.assertIs(D.recommend([full, d], "C"), d)
        usb = drive("H", "USB", "HDD", 3000)
        self.assertIs(D.recommend([full, usb], "C"), usb)       # the only one with room
        self.assertIsNone(D.recommend([full], "C"))
        small = drive("G", "NVMe", "SSD", 300)                  # room for the smallest model only
        self.assertIs(D.recommend([full, small], "C"), small)

    def test_choose(self):
        c, d = drive("C", "NVMe", "SSD", 50), drive("D", "NVMe", "SSD", 900)
        said = []

        def answers(*a):
            it = iter(a)
            return lambda prompt: next(it)
        here = os.path.abspath("C:/x/models") if os.name == "nt" else "/x/models"
        self.assertEqual(D.choose(here, [c, d], answers(""), said.append), "D:\\" + D.NEW_FOLDER)
        self.assertEqual(D.choose(here, [c, d], answers("1", "2"), said.append), "D:\\" + D.NEW_FOLDER)
        self.assertTrue(any("too little" in s for s in said))
        self.assertEqual(D.choose(here, [c, d], answers(None), said.append), here)   # no console: no change
        self.assertEqual(D.choose(here, [c, d], answers("nonsense", ""), said.append), "D:\\" + D.NEW_FOLDER)

    def test_write_models_folder_keeps_crlf(self):
        src = Path(D.PATHS_BAT).read_bytes()
        self.assertIn(b"\r\n", src)
        with tempfile.TemporaryDirectory() as t:
            bat = os.path.join(t, "ds41-paths.bat")
            Path(bat).write_bytes(src)
            self.assertTrue(D.write_models_folder("E:\\DeepSeek-V4.1", bat))
            new = Path(bat).read_bytes()
            self.assertIn(b'if not defined DS41_MODELS set "DS41_MODELS=E:/DeepSeek-V4.1"\r\n', new)
            self.assertEqual(new.count(b"\r\n"), src.count(b"\r\n"))
            self.assertEqual(len(new.splitlines()), len(src.splitlines()))
            self.assertTrue(D.write_models_folder("F:/Other", bat))        # a second change replaces the first
            self.assertIn(b'"DS41_MODELS=F:/Other"', Path(bat).read_bytes())
            Path(bat).write_bytes(b"@echo off\r\nset DS41_MODELS=X:/y\r\n")
            self.assertFalse(D.write_models_folder("E:/z", bat))           # an edited file is left alone

    def test_has_files(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(D.has_files(t))
            self.assertFalse(D.has_files(os.path.join(t, "nope")))
            os.makedirs(os.path.join(t, ".cache", "huggingface", "download"))
            self.assertFalse(D.has_files(t))
            Path(t, ".cache", "huggingface", "download", "a.incomplete").write_bytes(b"x")
            self.assertTrue(D.has_files(t))

    def test_memory_advice(self):
        self.assertIn("too small", " ".join(D.memory_advice((128 * 2**30, 129 * 2**30), True)))
        self.assertIn("managed by Windows", " ".join(D.memory_advice((128 * 2**30, 160 * 2**30), True)))
        self.assertIn("fixed - fine", " ".join(D.memory_advice((128 * 2**30, 192 * 2**30), False)))
        self.assertEqual(D.memory_advice(None, None), [])

    def test_report_runs(self):
        out = []
        D.report("C:/m", [drive("C", "NVMe", "SSD", 500)], True, (64 * 2**30, 96 * 2**30), 50e6, out.append)
        text = "\n".join(out)
        self.assertIn("NVMe SSD", text)
        self.assertIn("about 50 MB/s", text)
        self.assertIn("the default model (375 GB) about 2 h 5 min", text)


if __name__ == "__main__":
    unittest.main()
