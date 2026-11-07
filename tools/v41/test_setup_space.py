"""tools/v41/test_setup_space.py - the drafter plan setup_space.py gives first-start.bat (and its space check), which must
follow setup-ds41-models.bat's rules: 0.94's imatrix drafter first, 0.93's uniform one kept when this engine cannot
make the new one, never a loop of downloads.

    python -m unittest tools/v41/test_setup_space.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import setup_space as S  # noqa: E402


class DrafterPlan(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.m = self.d.name
        self.env = mock.patch.dict(os.environ, {"DS41_NO_DRAFTER": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.d.cleanup()

    def touch(self, name):
        open(os.path.join(self.m, name), "wb").close()

    def plan(self, usable):
        with mock.patch.object(S, "imatrix_usable", return_value=usable):
            return S.drafter_plan(self.m)

    def test_fresh_install_makes_it(self):
        self.assertEqual(self.plan(True), "make")
        self.assertEqual(self.plan(False), "make")      # the uniform one, as 0.93 made it

    def test_imx_present_is_done(self):
        self.touch(S.IMX_NAME)
        self.assertEqual(self.plan(True), "none")
        self.touch(S.UNIFORM_NAME)
        self.assertEqual(self.plan(True), "none")

    def test_093_install_upgrades_once(self):
        self.touch(S.UNIFORM_NAME)
        self.assertEqual(self.plan(True), "upgrade")

    def test_093_install_old_engine_keeps_uniform(self):
        self.touch(S.UNIFORM_NAME)
        self.assertEqual(self.plan(False), "none")      # no download loop when the engine cannot use the matrix

    def test_no_drafter_under_14_gb(self):
        with mock.patch.dict(os.environ, {"DS41_NO_DRAFTER": "1"}):
            self.assertEqual(self.plan(True), "none")

    def test_shipped_matrix_is_in_the_repo(self):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        p = os.path.join(root, "tools", "v41", "serve", "drafter-imatrix.bin")
        self.assertTrue(os.path.exists(p))
        with open(p, "rb") as f:
            self.assertEqual(int.from_bytes(f.read(4), "little"), 9)   # 3 blocks x w1 / w2 / w3


if __name__ == "__main__":
    unittest.main()
