"""0.94: a picture given as a local file path is read only up to IMAGE_URL_MAX (a path to the model's GGUF read hundreds
of GB into RAM), and not at all when the server listens beyond this PC unless "image_local_files" allows it.
    python -m unittest serve.test_image_local_files"""
import os
import tempfile
import unittest
from unittest import mock

from serve import server


class LocalImageFiles(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.small = os.path.join(self.dir, "small.png")
        with open(self.small, "wb") as f:
            f.write(b"\x89PNG not really")
        p = mock.patch.object(server.Vision, "local_files", True)
        p.start()
        self.addCleanup(p.stop)

    def test_small_file_read(self):
        self.assertEqual(server.Vision.load(self.small), b"\x89PNG not really")
        self.assertEqual(server.Vision.load("file://" + self.small.replace("\\", "/")), b"\x89PNG not really")

    def test_big_file_refused_before_reading(self):
        big = os.path.join(self.dir, "big.gguf")
        with open(big, "wb") as f:
            f.truncate(server.IMAGE_URL_MAX + 1)          # sparse: nothing written
        with mock.patch.object(server.Path, "read_bytes", side_effect=AssertionError("read")):
            with self.assertRaisesRegex(ValueError, "over 32 MiB"):
                server.Vision.load(big)

    def test_paths_off_when_listening_beyond_this_pc(self):
        server.Vision.local_files = False
        with self.assertRaisesRegex(ValueError, "listens beyond this PC"):
            server.Vision.load(self.small)
        data = "data:image/png;base64,iVBORw0KGgo="          # the picture itself still works
        self.assertEqual(server.Vision.load(data)[:4], b"\x89PNG")

    def test_listen_address_decides(self):
        self.assertTrue(server.this_pc_only("127.0.0.1") or {}.get("image_local_files") is True)
        self.assertFalse(server.this_pc_only("0.0.0.0") or {}.get("image_local_files") is True)
        self.assertTrue(server.this_pc_only("0.0.0.0") or {"image_local_files": True}.get("image_local_files") is True)


if __name__ == "__main__":
    unittest.main()
