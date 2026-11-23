"""serve/test_wmi_guard.py - no Strata process reaches CPython 3.12+'s _wmi on Windows (Niko1221/Strata 6f6099fe6).

platform.system() / processor() / uname() ask WMI through _wmi.exec_query; when WMI refuses the account the query
outlives its 100 ms wait and the late worker thread reads a dead stack (exit 0xC0000005).  setup.py reads
platform.processor() on Windows.  Each entry point blocks _wmi before platform is imported, so platform answers from
the registry, `ver` and the environment, as before Python 3.12."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRIES = ("setup.py", "serve/server.py", "tools/v41/serve/serve_ds41.py")
GUARD = 'sys.modules.setdefault("_wmi", None)'


class WmiGuard(unittest.TestCase):
    def test_each_entry_point_blocks_wmi_before_platform(self):
        for rel in ENTRIES:
            with self.subTest(rel):
                src = (ROOT / rel).read_text(encoding="utf-8")
                self.assertIn(GUARD, src)
                at = src.find("import platform")
                if at >= 0:
                    self.assertLess(src.index(GUARD), at)

    @unittest.skipUnless(os.name == "nt", "Windows only")
    def test_importing_the_server_keeps_platform_off_wmi(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import serve.server, platform; "
                "s = platform.system(); platform.processor(); platform.uname(); "
                "print(s, sys.modules.get('_wmi') is None)")
        out = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT)], capture_output=True, text=True,
                             timeout=120, cwd=ROOT)
        self.assertEqual(out.stdout.split(), ["Windows", "True"], out.stderr[-2000:])

    @unittest.skipUnless(os.name == "nt", "Windows only")
    def test_setup_keeps_platform_off_wmi(self):
        code = ("import sys, importlib.util; spec = importlib.util.spec_from_file_location('strata_setup', sys.argv[1]); "
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); import platform; "
                "print(platform.system(), bool(platform.processor()), sys.modules.get('_wmi') is None)")
        out = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "setup.py")], capture_output=True,
                             text=True, timeout=120, cwd=ROOT)
        self.assertEqual(out.stdout.split()[-3:], ["Windows", "True", "True"], (out.stdout[-500:], out.stderr[-2000:]))


if __name__ == "__main__":
    unittest.main()
