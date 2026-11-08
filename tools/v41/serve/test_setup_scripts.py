"""The Windows setup scripts' rules that a unit test can reach (no GPU is used and nothing is downloaded).

S8 (0.94): setup-ds41-models.bat sets DS41_NO_DRAFTER itself on a card under 14 GB, so Enable Voice.bat and
Enable Pictures.bat (which call it without first-start.bat) no longer download and build the 7.9 GB drafter that
Start DeepSeek.bat never uses there, and the space check before the downloads does not count it.

    python tools/v41/serve/test_setup_scripts.py
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SETUP = ROOT / "setup-ds41-models.bat"
FIRST_START = ROOT / "tools" / "v41" / "serve" / "first-start.bat"
LAUNCHER = ROOT / "Start DeepSeek.bat"


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def index(text: list[str], pattern: str) -> int:
    hits = [i for i, line in enumerate(text) if re.search(pattern, line)]
    if len(hits) != 1:
        raise AssertionError(f"{pattern!r}: {len(hits)} lines match, not 1")
    return hits[0]


class SmallCardDrafter(unittest.TestCase):
    NO_DRAFTER = r'^if defined VRAM_MIB if %VRAMN% LSS (\d+) set "DS41_NO_DRAFTER=1"$'

    def test_the_same_threshold_as_the_launcher(self):
        small = re.search(r'LSS (\d+) set "SMALLCARD=1"', LAUNCHER.read_text(encoding="utf-8"))
        for path in (SETUP, FIRST_START):
            with self.subTest(path.name):
                text = lines(path)
                self.assertEqual(re.match(self.NO_DRAFTER, text[index(text, self.NO_DRAFTER)]).group(1),
                                 small.group(1))

    def test_set_before_the_space_check_and_the_drafter_step(self):
        text = lines(SETUP)
        self.assertLess(index(text, r'^call "%~dp0ds41-paths\.bat"$'), index(text, r"nvidia-smi .*-i %DS41_GPU%"))
        self.assertLess(index(text, self.NO_DRAFTER), index(text, r'^"%PY%" tools\\v41\\setup_space\.py "%M%"'))
        self.assertLess(index(text, self.NO_DRAFTER), index(text, r"^if defined DS41_NO_DRAFTER goto main$"))
        # never cleared here: a DS41_NO_DRAFTER=1 from first-start.bat or the user's environment stands
        self.assertFalse([line for line in text if re.match(r'\s*set "DS41_NO_DRAFTER="', line)])

    def test_the_enable_scripts_go_through_it(self):
        for name, what in (("Enable Voice.bat", "voice"), ("Enable Pictures.bat", "pictures")):
            with self.subTest(name):
                text = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn(f'call "%~dp0setup-ds41-models.bat" {what}', text)

    @unittest.skipUnless(os.name == "nt", "runs cmd.exe")
    def test_the_check_in_cmd(self):
        """The block itself, run by cmd.exe with a stand-in nvidia-smi that prints the card's memory."""
        text = lines(SETUP)
        block = text[index(text, r'^set "VRAM_MIB="$'):index(text, self.NO_DRAFTER) + 1]
        with tempfile.TemporaryDirectory() as d:
            Path(d, "nvidia-smi.bat").write_text("@echo off\r\necho %FAKE_VRAM%\r\n", encoding="ascii")
            harness = Path(d, "check.bat")
            harness.write_text("\r\n".join(["@echo off", "setlocal", 'set "DS41_GPU=0"', *block,
                                            "echo NO_DRAFTER=[%DS41_NO_DRAFTER%]", ""]), encoding="ascii")
            for vram, preset, want in (("12288", None, "1"), ("8192", None, "1"), ("16376", None, ""),
                                       ("24576", None, ""), ("[N/A]", None, ""), ("No devices were found", None, ""),
                                       ("24576", "1", "1")):
                env = {k: v for k, v in os.environ.items() if k.upper() not in ("DS41_NO_DRAFTER", "FAKE_VRAM")}
                env.update(FAKE_VRAM=vram, PATH=d + os.pathsep + env.get("PATH", ""))
                if preset is not None:
                    env["DS41_NO_DRAFTER"] = preset
                with self.subTest(vram=vram, preset=preset):
                    r = subprocess.run(["cmd", "/d", "/c", str(harness)], cwd=d, env=env, capture_output=True,
                                       text=True, timeout=60)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    self.assertIn(f"NO_DRAFTER=[{want}]", r.stdout, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
