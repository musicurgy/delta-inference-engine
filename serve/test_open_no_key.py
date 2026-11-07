"""0.94: listening beyond this PC with no API key is refused unless the user opts in ("allow_no_key": true or
STRATA_ALLOW_NO_KEY=1).    python -m unittest serve.test_open_no_key"""
import unittest

from serve.server import open_without_key, this_pc_only


class OpenWithoutKey(unittest.TestCase):
    def test_this_pc_only(self):
        for h in ("127.0.0.1", "127.0.0.2", "localhost", "::1", "[::1]", "LOCALHOST"):
            self.assertTrue(this_pc_only(h), h)
        for h in ("0.0.0.0", "::", "", "192.168.1.5", "10.0.0.2", "myhost", "example.com"):
            self.assertFalse(this_pc_only(h), h)

    def test_refused(self):
        for h in ("0.0.0.0", "::", "192.168.1.5", "myhost"):
            self.assertTrue(open_without_key(h, "", {}, {}), h)

    def test_allowed(self):
        self.assertFalse(open_without_key("127.0.0.1", "", {}, {}))
        self.assertFalse(open_without_key("0.0.0.0", "secret", {}, {}))
        self.assertFalse(open_without_key("0.0.0.0", ["k1", "k2"], {}, {}))
        self.assertFalse(open_without_key("0.0.0.0", "", {"allow_no_key": True}, {}))
        self.assertFalse(open_without_key("0.0.0.0", "", {}, {"STRATA_ALLOW_NO_KEY": "1"}))

    def test_opt_in_must_be_explicit(self):
        self.assertTrue(open_without_key("0.0.0.0", "", {"allow_no_key": "yes"}, {}))
        self.assertTrue(open_without_key("0.0.0.0", "", {"allow_no_key": 1}, {}))
        self.assertTrue(open_without_key("0.0.0.0", "", {}, {"STRATA_ALLOW_NO_KEY": "true"}))


if __name__ == "__main__":
    unittest.main()
